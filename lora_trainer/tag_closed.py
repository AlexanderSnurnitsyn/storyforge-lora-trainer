#!/usr/bin/env python3
"""
tag_closed.py — CLOSED classification pass (transformers + constrained JSON).

Self-contained. Assigns labels from your curated vocab.txt per facet (+ overflow),
with lm-format-enforcer guaranteeing valid JSON. Built for the full ~1900-scene run
on a single 12GB GPU (Windows-friendly — no vLLM, which needs Linux/WSL).

Features:
  * --ids-from manifest.jsonl : tag EXACTLY the corpus scenes (100% coverage)
  * checkpoint/resume         : the output jsonl IS the checkpoint; each scene is
    appended+flushed as it finishes, so an interrupted run resumes where it stopped
    (already-tagged scene_ids are skipped). Use --restart to start fresh.
  * progress + ETA            : live rate and time-remaining estimate.

Output ({scene_id, text, parse_ok, facets{...}, overflow[...]}) feeds
build_tagged_corpus.py unchanged.

Run:
  python tag_closed.py ../dataset_builder/coc2.db --vocab vocab.txt \
      --ids-from ../dataset_builder/corpus.manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

from tag_pilot import (clean, est_tokens, get_keyword_tools, fetch_segments,
                       build_marked, extract_json)
from tag_discover import load_vocab, all_scene_ids, FACETS
from tagger_model import load_tagger

SYSTEM = ("You are a scene classifier. For each facet, select ONLY labels from the "
          "provided list that genuinely apply to the scene. Output ONLY one JSON object.")

USER_TMPL = """Classify the scene. For each facet, pick the applicable labels FROM ITS LIST below
(0 or more). Use ONLY labels that clearly apply — do not stretch. Do NOT invent labels:
if an important descriptor is truly missing from the lists, put it in "overflow" (lowercase,
short). Ground every choice in what actually happens.

Allowed labels per facet:
{allowed}

Return EXACTLY these keys (lists of chosen labels):
{schema}

SCENE:
{scene}"""


def build_prompt(vocab):
    facet_lines = []
    for f in FACETS:
        labs = vocab.get(f, [])
        if labs:
            facet_lines.append(f"- {f}: {', '.join(labs)}")
    allowed = "\n".join(facet_lines)
    keys = [f for f in FACETS if vocab.get(f)]
    schema_str = "{" + ",".join(f'"{f}":[]' for f in keys) + ',"overflow":[]}'
    return allowed, schema_str, keys


def make_prefix_factory(tok, keys):
    """Factory -> fresh constrained-decoding fn per scene (bounds lmfe cache).
    Returns None if lm-format-enforcer is absent."""
    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn)
    except Exception:
        return None
    props = {k: {"type": "array", "items": {"type": "string"}} for k in keys}
    props["overflow"] = {"type": "array", "items": {"type": "string"}}
    schema = {"type": "object", "properties": props, "required": list(keys) + ["overflow"]}
    try:
        from lmformatenforcer import CharacterLevelParserConfig
        parser = JsonSchemaParser(schema,
                                  config=CharacterLevelParserConfig(max_consecutive_whitespaces=0))
    except Exception:
        parser = JsonSchemaParser(schema)

    def factory():
        return build_transformers_prefix_allowed_tokens_fn(tok, parser)
    return factory


def parse_json(raw):
    obj = extract_json(raw)
    if obj is not None:
        return obj
    if "{" not in raw:
        return None
    s = raw[raw.find("{"):]
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s += "]" * max(0, s.count("[") - s.count("]"))
    s += "}" * max(0, s.count("{") - s.count("}"))
    try:
        return json.loads(s)
    except Exception:
        return None


def gen(model, tok, scene_text, prompt_body, max_new_tokens, prefix_fn=None):
    import torch
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt_body.replace("{scene}", scene_text)}]
    inputs = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                     return_tensors="pt").to(model.device)
    attn = inputs.new_ones(inputs.shape)
    kw = dict(input_ids=inputs, attention_mask=attn, max_new_tokens=max_new_tokens,
              do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id)
    if prefix_fn is not None:
        kw["prefix_allowed_tokens_fn"] = prefix_fn
    with torch.inference_mode():
        out = model.generate(**kw)
    text = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
    del out, inputs, attn
    return text


def classify(model, tok, scene_text, prompt_body, max_new_tokens, prefix_fn=None):
    raw = gen(model, tok, scene_text, prompt_body, max_new_tokens, prefix_fn)
    obj = parse_json(raw)
    if obj is None and prefix_fn is None:
        raw = gen(model, tok, scene_text + "\n\n(Output ONLY the JSON object.)",
                  prompt_body, max_new_tokens)
        obj = parse_json(raw)
    return obj


def load_scene_ids(args):
    if args.ids_from:
        ids, seen = [], set()
        with open(args.ids_from, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sid = str(json.loads(line)["scene_id"])
                if sid not in seen:
                    seen.add(sid)
                    ids.append(sid)
        return ids, False
    return all_scene_ids(args.db, args.table, args.text_col, args.scene_col,
                         args.max_segments, args.seed), True


def gather_scenes(args, ids, do_filter, matcher, hits_fn):
    out, kw_reject, skipped = [], 0, 0
    for sid in ids:
        segs = fetch_segments(args.db, args.table, args.text_col,
                              args.scene_col, args.pos_col, sid)
        text, _, _ = build_marked(segs)
        if not text:
            continue
        if do_filter and hits_fn and hits_fn(text, matcher) < args.min_keyword_hits:
            kw_reject += 1
            continue
        if est_tokens(text) > args.max_scene_tokens:
            skipped += 1
            continue
        out.append((sid, text))
        if args.sample and args.sample > 0 and len(out) >= args.sample:
            break
    return out, kw_reject, skipped


def fmt_eta(sec):
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def main():
    ap = argparse.ArgumentParser(description="Closed classification (transformers).")
    ap.add_argument("db")
    ap.add_argument("--vocab", default="vocab.txt")
    ap.add_argument("--model", default="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
    ap.add_argument("--ids-from", default=None, help="manifest jsonl: tag exactly these")
    ap.add_argument("--sample", type=int, default=0, help="0 = all")
    ap.add_argument("--max-segments", type=int, default=4)
    ap.add_argument("--keywords", default=None)
    ap.add_argument("--builder-path", default=None)
    ap.add_argument("--min-keyword-hits", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="tag_closed.jsonl")
    ap.add_argument("--restart", action="store_true", help="ignore existing output, start fresh")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--max-scene-tokens", type=int, default=3600)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--no-constrain", action="store_true")
    ap.add_argument("--table", default="segments")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--scene-col", default="scene_id")
    ap.add_argument("--pos-col", default="pos")
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)
    allowed, schema_str, keys = build_prompt(vocab)
    if not keys:
        sys.exit(f"No labels in {args.vocab}. Curate it first.")
    allowed_sets = {f: {l.lower() for l in vocab.get(f, [])} for f in keys}
    prompt_body = USER_TMPL.format(allowed=allowed, schema=schema_str, scene="{scene}")
    print(f"vocab: {sum(len(vocab[f]) for f in keys)} labels / {len(keys)} facets",
          file=sys.stderr)

    matcher = hits_fn = None
    ids, do_filter = load_scene_ids(args)
    if do_filter and (args.keywords or args.builder_path):
        matcher, hits_fn, src = get_keyword_tools(args.keywords, args.builder_path)
        print(f"keyword filter: {src} (>= {args.min_keyword_hits} hits)", file=sys.stderr)
    scenes, kw_reject, skipped = gather_scenes(args, ids, do_filter, matcher, hits_fn)
    print(f"scenes to tag: {len(scenes):,}  (kw_reject {kw_reject}, skipped_long {skipped})",
          file=sys.stderr)
    if not scenes:
        sys.exit("No scenes to tag.")

    # ---- checkpoint/resume: the output jsonl is the checkpoint ----
    if args.restart and os.path.exists(args.out):
        os.remove(args.out)
    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(str(json.loads(line)["scene_id"]))
                except Exception:
                    pass
        print(f"resuming: {len(done_ids):,} already done, "
              f"{len(scenes) - len(done_ids):,} to go", file=sys.stderr)
    todo = [(sid, text) for sid, text in scenes if sid not in done_ids]
    if not todo:
        print("nothing to do — all scenes already tagged.", file=sys.stderr)
    else:
        print(f"Loading model: {args.model}", file=sys.stderr)
        model, tok = load_tagger(args.model, args.max_seq_len, not args.no_4bit)
        import torch

        prefix_factory = None if args.no_constrain else make_prefix_factory(tok, keys)
        if prefix_factory is not None:
            raw = gen(model, tok, "He nodded once and left the room.",
                      prompt_body, 256, prefix_factory())
            ok = parse_json(raw) is not None
            run = max((len(m.group(0)) for m in re.finditer(r"\s+", raw)), default=0)
            print(f"constrained JSON: {'ON, self-test PASS' if ok else 'ON, SELF-TEST FAILED'}"
                  f" ({'compact' if run <= 1 else f'whitespace run={run}'})", file=sys.stderr)
        else:
            print("constrained JSON: OFF (pip install lm-format-enforcer)", file=sys.stderr)

        fout = open(args.out, "a", encoding="utf-8")
        t0 = time.time()
        total = len(scenes)
        base = len(done_ids)
        for i, (sid, text) in enumerate(todo, start=1):
            pf = prefix_factory() if prefix_factory is not None else None
            obj = classify(model, tok, text, prompt_body, args.max_new_tokens, pf)
            if obj is None:
                rec = {"scene_id": sid, "text": text, "parse_ok": False}
            else:
                facets = {}
                for f in keys:
                    vals = obj.get(f, [])
                    if isinstance(vals, str):
                        vals = [vals]
                    keep, seen = [], set()
                    for lab in (vals or []):
                        if not isinstance(lab, str):
                            continue
                        ll = lab.strip().lower()
                        if not ll or ll in seen or ll not in allowed_sets[f]:
                            continue
                        seen.add(ll)
                        keep.append(lab)
                    facets[f] = keep
                ov = [x for x in (obj.get("overflow") or []) if isinstance(x, str)]
                rec = {"scene_id": sid, "text": text, "parse_ok": True,
                       "facets": facets, "overflow": ov}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            if i % 8 == 0:
                torch.cuda.empty_cache()
            elapsed = time.time() - t0
            rate = elapsed / i
            eta = fmt_eta((len(todo) - i) * rate)
            print(f"\r  {base + i}/{total} ({100*(base+i)/total:.0f}%) "
                  f"{rate:.1f}s/scene  ETA {eta}    ", end="", file=sys.stderr, flush=True)
        fout.close()
        print("", file=sys.stderr)

    # ---- summary over the FULL output (resume-safe) ----
    records = []
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    ok = [r for r in records if r.get("parse_ok")]
    parse_fail = len(records) - len(ok)
    facet_hits, label_counts, overflow = Counter(), Counter(), Counter()
    for r in ok:
        for f in keys:
            labs = r.get("facets", {}).get(f, [])
            if labs:
                facet_hits[f] += 1
            for lab in labs:
                label_counts[f"{f}:{str(lab).lower()}"] += 1
        for x in r.get("overflow", []):
            overflow[str(x).lower()] += 1

    print("\n" + "=" * 60)
    print(f"CLOSED  n={len(records)}  parsed={len(ok)}  parse_fail={parse_fail}")
    print("=" * 60)
    print("\n[coverage] scenes with >=1 label, per facet")
    for f in keys:
        print(f"    {f:>14}: {facet_hits[f]:>4}/{len(ok)}  "
              f"({100*facet_hits[f]/max(1,len(ok)):.0f}%)")
    print("\n[top labels assigned]")
    for fl, n in label_counts.most_common(20):
        print(f"    {fl}: {n}")
    if overflow:
        print("\n[overflow] frequent non-vocab descriptors (vocab v2 candidates):")
        for x, n in overflow.most_common(15):
            print(f"    {x}: {n}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
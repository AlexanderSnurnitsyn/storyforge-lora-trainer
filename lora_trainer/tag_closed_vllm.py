#!/usr/bin/env python3
"""
tag_closed_vllm.py — CLOSED classification on vLLM (self-contained). For WSL/Linux.

Same output as tag_closed.py ({scene_id, text, parse_ok, facets{...}, overflow[...]}),
so build_tagged_corpus.py consumes it unchanged — but with continuous batching and
guided JSON decoding (minutes, not hours; zero parse_fail). Includes checkpoint/resume
(the output jsonl is the checkpoint) and progress + ETA.

Only needs tag_pilot.py + tag_discover.py present (stdlib-importable) and `vllm`.

Run (inside the WSL venv):
  cd /mnt/d/Story_Forge/tools/lora_trainer
  python tag_closed_vllm.py ../dataset_builder/coc2.db --vocab vocab.txt \
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

from tag_pilot import (est_tokens, get_keyword_tools, fetch_segments, build_marked,
                       extract_json)
from tag_discover import load_vocab, all_scene_ids, FACETS

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
    lines = []
    for f in FACETS:
        if vocab.get(f):
            lines.append(f"- {f}: {', '.join(vocab[f])}")
    keys = [f for f in FACETS if vocab.get(f)]
    schema_str = "{" + ",".join(f'"{f}":[]' for f in keys) + ',"overflow":[]}'
    return "\n".join(lines), schema_str, keys


def build_schema(keys):
    props = {k: {"type": "array", "items": {"type": "string"}} for k in keys}
    props["overflow"] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "properties": props, "required": list(keys) + ["overflow"]}


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


def make_sampling_params(schema, max_tokens):
    """Version-tolerant structured-output SamplingParams.
    vLLM v0.12+ : structured_outputs=StructuredOutputsParams(json=schema)
    older        : guided_decoding=GuidedDecodingParams(json=schema)
    """
    from vllm import SamplingParams
    try:                                    # new API (v0.12+)
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(temperature=0.0, max_tokens=max_tokens,
                              structured_outputs=StructuredOutputsParams(json=schema))
    except Exception:                       # older API
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(temperature=0.0, max_tokens=max_tokens,
                              guided_decoding=GuidedDecodingParams(json=schema))


def process(obj, sid, text, keys, allowed_sets):
    if obj is None:
        return {"scene_id": sid, "text": text, "parse_ok": False}
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
    return {"scene_id": sid, "text": text, "parse_ok": True, "facets": facets, "overflow": ov}


def main():
    ap = argparse.ArgumentParser(description="Closed classification on vLLM.")
    ap.add_argument("db")
    ap.add_argument("--vocab", default="vocab.txt")
    ap.add_argument("--model", default="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
    ap.add_argument("--ids-from", default=None)
    ap.add_argument("--sample", type=int, default=0, help="0 = all")
    ap.add_argument("--max-segments", type=int, default=4)
    ap.add_argument("--keywords", default=None)
    ap.add_argument("--builder-path", default=None)
    ap.add_argument("--min-keyword-hits", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="tag_closed.jsonl")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--max-scene-tokens", type=int, default=2600)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--batch", type=int, default=256)
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
    schema = build_schema(keys)
    prompt_body = USER_TMPL.format(allowed=allowed, schema=schema_str, scene="{scene}")
    print(f"vocab: {sum(len(vocab[f]) for f in keys)} labels / {len(keys)} facets",
          file=sys.stderr)

    matcher = hits_fn = None
    ids, do_filter = load_scene_ids(args)
    if do_filter and (args.keywords or args.builder_path):
        matcher, hits_fn, _ = get_keyword_tools(args.keywords, args.builder_path)
    scenes, kw_reject, skipped = gather_scenes(args, ids, do_filter, matcher, hits_fn)
    print(f"scenes to tag: {len(scenes):,}  (kw_reject {kw_reject}, skipped_long {skipped})",
          file=sys.stderr)
    if not scenes:
        sys.exit("No scenes to tag.")

    # ---- checkpoint/resume ----
    if args.restart and os.path.exists(args.out):
        os.remove(args.out)
    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(str(json.loads(line)["scene_id"]))
                    except Exception:
                        pass
        print(f"resuming: {len(done_ids):,} done, {len(scenes)-len(done_ids):,} to go",
              file=sys.stderr)
    todo = [(sid, text) for sid, text in scenes if sid not in done_ids]

    if todo:
        from vllm import LLM

        llm = LLM(model=args.model, quantization="bitsandbytes",
                  load_format="bitsandbytes", dtype="bfloat16",
                  max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_mem_util, enforce_eager=True)
        tok = llm.get_tokenizer()
        sp = make_sampling_params(schema, args.max_new_tokens)

        def to_prompt(t):
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt_body.replace("{scene}", t)}]
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        fout = open(args.out, "a", encoding="utf-8")
        t0 = time.time()
        base, total = len(done_ids), len(scenes)
        for start in range(0, len(todo), args.batch):
            chunk = todo[start:start + args.batch]
            outs = llm.generate([to_prompt(t) for _, t in chunk], sp)
            for (sid, text), o in zip(chunk, outs):
                rec = process(parse_json(o.outputs[0].text), sid, text, keys, allowed_sets)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done = min(start + args.batch, len(todo))
            elapsed = time.time() - t0
            eta = fmt_eta((len(todo) - done) * (elapsed / done))
            print(f"\r  {base+done}/{total} ({100*(base+done)/total:.0f}%)  "
                  f"ETA {eta}    ", end="", file=sys.stderr, flush=True)
        fout.close()
        print("", file=sys.stderr)

    # ---- summary over the full file ----
    records = [json.loads(l) for l in open(args.out, encoding="utf-8") if l.strip()]
    ok = [r for r in records if r.get("parse_ok")]
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
    print(f"CLOSED (vLLM)  n={len(records)}  parsed={len(ok)}  parse_fail={len(records)-len(ok)}")
    print("=" * 60)
    print("\n[coverage] scenes with >=1 label, per facet")
    for f in keys:
        print(f"    {f:>14}: {facet_hits[f]:>4}/{len(ok)}  "
              f"({100*facet_hits[f]/max(1,len(ok)):.0f}%)")
    print("\n[top labels assigned]")
    for fl, n in label_counts.most_common(20):
        print(f"    {fl}: {n}")
    if overflow:
        print("\n[overflow] vocab v2 candidates:")
        for x, n in overflow.most_common(15):
            print(f"    {x}: {n}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

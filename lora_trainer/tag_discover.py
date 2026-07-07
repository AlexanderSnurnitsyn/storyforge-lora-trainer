#!/usr/bin/env python3
"""
tag_discover.py — DISCOVERY pass: harvest a candidate controlled vocabulary.

Why this exists (pilot findings):
  - coherence scoring is useless: the judge missed 100% of injected splices -> dropped.
  - POV is mis-tagged and the corpus is 2nd-person throughout (a constant) -> dropped.
  - verbatim entity extraction pulls names / places / typos -> dropped.
So we switch from EXTRACTION to CLASSIFICATION. Here we run the OPEN half: ask the
model for short, generic, GROUNDED descriptors of each scene across a few facets,
then aggregate everything into a frequency-ranked candidate vocabulary you curate by
hand. Curation is bounded — you review a list, not 6000 scenes.

Frequency doubles as a LEARNABILITY signal: an attribute in only 1-2 scenes is not a
useful control tag (the LoRA has nothing to learn the handle from). Keep labels that
are both meaningful AND frequent enough.

Next step (not this script): a CLOSED pass that classifies every scene against the
curated vocab.

Outputs:
  tag_discover.jsonl    per-scene facet labels (+ scene text, for review)
  candidate_vocab.md    per-facet  label -> count, sorted  (curate THIS)

Run (trainer env; needs unsloth + GPU):
  python tag_discover.py your.db --sample 150 \
      --builder-path ../dataset_builder --keywords ../dataset_builder/keywords.txt

Re-aggregate without the GPU (after tweaking normalisation):
  python tag_discover.py --aggregate-only tag_discover.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from collections import Counter

# reuse the pilot's vetted helpers (stdlib import; torch only loads inside load_model)
from tag_pilot import (clean, est_tokens, get_keyword_tools, fetch_segments,
                       build_marked, extract_json, load_model)

FACETS = ["situation", "interactions", "appearance", "actions",
          "setting", "tone", "dynamics"]

SYSTEM = ("You are a precise scene-descriptor tagger. Output ONLY one valid JSON "
          "object. No prose, no markdown fences.")

USER_TMPL = """Describe the scene below with short, GENERIC, lowercase descriptors — the kind a
reader would use to request a scene like this. Describe scene FEATURES, not the wording.
Do NOT output proper names, place names, or invented terms. Ground every label in what
actually happens; if a facet does not apply, return [].

Facets (0-6 short labels each, snake_case, e.g. "first_meeting", "scarred_face"):
- situation: the kind of scene / trope (farewell, confrontation, seduction, reunion...)
- interactions: specific acts between characters (embrace, kiss_on_cheek, slap, confession...)
- appearance: notable physical features present (scar, tall, red_hair, eyepatch, armor...)
- actions: notable activities (sword_fight, cooking, dancing, spellcasting, travel...)
- setting: where it happens (tavern, bedroom, forest, throne_room, ship...)
- tone: mood (tender, tense, playful, melancholic, ominous...)
- dynamics: relationship / power between characters (lovers, enemies, strangers, mentor_pupil...)

Return EXACTLY these keys:
{{"situation":[],"interactions":[],"appearance":[],"actions":[],"setting":[],"tone":[],"dynamics":[]}}

SCENE:
{scene}"""


def gen(model, tok, scene_text, max_new_tokens):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(scene=scene_text)}]
    inputs = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                     return_tensors="pt").to(model.device)
    attn = inputs.new_ones(inputs.shape)
    out = model.generate(input_ids=inputs, attention_mask=attn,
                         max_new_tokens=max_new_tokens, do_sample=False,
                         use_cache=True, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)


def tag(model, tok, scene_text, max_new_tokens):
    raw = gen(model, tok, scene_text, max_new_tokens)
    obj = extract_json(raw)
    if obj is None:
        raw = gen(model, tok, scene_text + "\n\n(Output ONLY the JSON object.)",
                  max_new_tokens)
        obj = extract_json(raw)
    return obj


# --------------------------------------------------------------------------- #
def norm_label(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s-]", " ", s)        # punctuation -> separator (not a join)
    s = re.sub(r"[\s-]+", "_", s)
    return s.strip("_")


def aggregate(records):
    """records -> {facet: Counter(label -> n_scenes)}."""
    agg = {f: Counter() for f in FACETS}
    for r in records:
        if not r.get("parse_ok", True):
            continue
        for f in FACETS:
            vals = r.get(f) or []
            if isinstance(vals, str):          # model returned a string, not a list
                vals = [vals]                  # (do NOT iterate it char-by-char)
            elif not isinstance(vals, list):
                continue
            seen = set()
            for lab in vals:
                if not isinstance(lab, str):
                    continue
                nl = norm_label(lab)
                if len(nl) < 2:                # drop single-char noise from bad parses
                    continue
                if nl not in seen:             # count once per scene
                    seen.add(nl)
                    agg[f][nl] += 1
    return agg


def write_vocab_edit(agg, path, min_count):
    """Emit the EDITABLE vocab the closed classifier consumes. INI-style facets,
    one label per line, discovery count as a trailing comment. Pre-filled with
    labels seen in >= min_count scenes; the user adds/removes freely."""
    lines = [
        "# Editable vocabulary — the CLOSED classifier reads THIS file (not the .md).",
        "# Format: [facet] headers, one label per line. '# n' = scenes seen in discovery.",
        "# Add your own labels, delete noise/synonyms. Blank lines and #-lines are ignored.",
        f"# Pre-filled with labels seen in >= {min_count} scenes.",
        "",
    ]
    for f in FACETS:
        lines.append(f"[{f}]")
        for lab, n in sorted(agg[f].items(), key=lambda x: (-x[1], x[0])):
            if n >= min_count:
                lines.append(f"{lab}    # {n}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def load_vocab(path):
    """Parse the editable vocab.txt -> {facet: [labels]}. Shared with the closed pass."""
    vocab = {}
    cur = None
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"\[(\w+)\]$", s)
        if m:
            cur = m.group(1)
            vocab.setdefault(cur, [])
            continue
        if cur is None:
            continue
        lab = s.split("#", 1)[0].strip()
        if lab:
            vocab[cur].append(lab)
    return vocab


def write_vocab(agg, n_scenes, path):
    lines = [f"# Candidate vocabulary  (from {n_scenes} scenes)\n",
             "Curate each facet: merge synonyms, drop noise, and keep labels that are "
             "both meaningful and frequent enough to be learnable "
             "(rule of thumb: a control tag needs enough example scenes to learn from).\n"]
    for f in FACETS:
        c = agg[f]
        lines.append(f"\n## {f}  ({len(c)} unique)\n")
        lines.append("| label | scenes |")
        lines.append("|---|---|")
        for lab, n in sorted(c.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"| {lab} | {n} |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def print_top(agg, k=15):
    print("\n" + "=" * 60)
    print("CANDIDATE VOCABULARY  (top labels per facet)")
    print("=" * 60)
    for f in FACETS:
        c = agg[f]
        top = ", ".join(f"{lab}:{n}" for lab, n in c.most_common(k))
        print(f"\n[{f}]  ({len(c)} unique)\n    {top or '—'}")


# --------------------------------------------------------------------------- #
def all_scene_ids(db, table, text_col, scene_col, max_segments, seed):
    con = sqlite3.connect(db)
    try:
        having = "HAVING COUNT(*) <= ?" if max_segments else ""
        q = (f"SELECT {scene_col} FROM {table} WHERE {text_col} IS NOT NULL "
             f"GROUP BY {scene_col} {having}")
        params = (max_segments,) if max_segments else ()
        ids = [r[0] for r in con.execute(q, params)]
    finally:
        con.close()
    random.Random(seed).shuffle(ids)
    return ids


def main():
    ap = argparse.ArgumentParser(description="Discovery pass: harvest candidate vocab.")
    ap.add_argument("db", nargs="?", help="SQLite DB (omit with --aggregate-only)")
    ap.add_argument("--aggregate-only", default=None,
                    help="skip the model; re-aggregate an existing tag_discover.jsonl")
    ap.add_argument("--model", default="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--max-segments", type=int, default=0,
                    help="0 = no cap; else only scenes with <= this many segments")
    ap.add_argument("--keywords", default=None)
    ap.add_argument("--builder-path", default=None)
    ap.add_argument("--min-keyword-hits", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tag_discover.jsonl")
    ap.add_argument("--vocab-out", default="candidate_vocab.md")
    ap.add_argument("--vocab-edit-out", default="vocab.txt",
                    help="editable vocab file the closed classifier reads")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--min-count", type=int, default=2,
                    help="pre-fill vocab.txt with labels seen in >= this many scenes")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--max-scene-tokens", type=int, default=3600)
    ap.add_argument("--table", default="segments")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--scene-col", default="scene_id")
    ap.add_argument("--pos-col", default="pos")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    # ---- aggregate-only path (no GPU) ----
    if args.aggregate_only:
        records = [json.loads(l) for l in open(args.aggregate_only, encoding="utf-8")
                   if l.strip()]
        agg = aggregate(records)
        write_vocab(agg, len(records), args.vocab_out)
        write_vocab_edit(agg, args.vocab_edit_out, args.min_count)
        print_top(agg)
        print(f"\nWrote {args.vocab_out} (report) and {args.vocab_edit_out} (edit THIS)")
        return

    if not args.db:
        sys.exit("Provide a DB path (or use --aggregate-only FILE).")

    kw_enabled = bool(args.keywords or args.builder_path)
    matcher = hits_fn = None
    if kw_enabled:
        matcher, hits_fn, src = get_keyword_tools(args.keywords, args.builder_path)
        print(f"keyword filter: {src} (>= {args.min_keyword_hits} hits)", file=sys.stderr)

    ids = all_scene_ids(args.db, args.table, args.text_col, args.scene_col,
                        args.max_segments, args.seed)
    print(f"Candidates: {len(ids):,}"
          + (f"  (<= {args.max_segments} segments)" if args.max_segments else ""),
          file=sys.stderr)

    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tok = load_model(args.model, args.max_seq_len, not args.no_4bit)

    records = []
    kw_reject = skipped = parse_fail = 0
    tagged = 0
    for sid in ids:
        if tagged >= args.sample:
            break
        segs = fetch_segments(args.db, args.table, args.text_col,
                              args.scene_col, args.pos_col, sid)
        clean_joined, _, _ = build_marked(segs)
        if not clean_joined:
            continue
        if kw_enabled and hits_fn(clean_joined, matcher) < args.min_keyword_hits:
            kw_reject += 1
            continue
        if est_tokens(clean_joined) > args.max_scene_tokens:
            skipped += 1
            continue

        tagged += 1
        obj = tag(model, tok, clean_joined, args.max_new_tokens)
        print(f"\r  {tagged}/{args.sample}  ", end="", file=sys.stderr, flush=True)
        rec = {"scene_id": sid, "n_segments": len(segs), "text": clean_joined,
               "parse_ok": obj is not None}
        if obj:
            for f in FACETS:
                rec[f] = obj.get(f, [])
        else:
            parse_fail += 1
        records.append(rec)
    print("", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    agg = aggregate(records)
    write_vocab(agg, sum(1 for r in records if r.get("parse_ok")), args.vocab_out)
    write_vocab_edit(agg, args.vocab_edit_out, args.min_count)
    print_top(agg)
    print(f"\nscenes tagged: {len(records)}  parse_fail: {parse_fail}  "
          f"kw_reject: {kw_reject}  skipped_long: {skipped}")
    print(f"Wrote {args.out}, {args.vocab_out} (report), {args.vocab_edit_out} (edit THIS)")


if __name__ == "__main__":
    main()

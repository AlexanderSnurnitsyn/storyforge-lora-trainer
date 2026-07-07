#!/usr/bin/env python3
"""
build_tagged_corpus.py — turn the PLAIN corpus into a TAGGED corpus, on exactly the
same scenes and splits, so the only difference between the two trained models is the
presence of tag-instructions.

Inputs:
  --manifest corpus.manifest.jsonl   (from build_corpus.py: scene_id, split, text)
  --tags     tag_closed.jsonl         (from tag_closed.py: scene_id, facets{...}, overflow)

Output ({"prompt","completion"} -> dataset.py detects "instruction" format and masks
the prompt, so only the scene is learned):
  corpus_tagged.train.jsonl
  corpus_tagged.val.jsonl

The completion text comes from the MANIFEST (not the tag file), so the scene text is
byte-identical to the plain corpus. Empty/missing tags -> a generic "Write a scene."
prompt, so every plain example has a tagged counterpart (alignment preserved).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from tag_discover import FACETS

FACET_DISPLAY = {
    "situation": "Situation", "interactions": "Interactions", "appearance": "Appearance",
    "actions": "Actions", "setting": "Setting", "tone": "Tone", "dynamics": "Dynamics",
}


def build_prompt(tags: dict, facets, include_overflow: bool) -> str:
    lines, any_el = ["Write a scene with the following elements."], False
    for f in facets:
        labs = tags.get(f) or []
        if labs:
            disp = ", ".join(str(l).replace("_", " ").strip() for l in labs if l)
            if disp:
                lines.append(f"{FACET_DISPLAY.get(f, f.title())}: {disp}")
                any_el = True
    if include_overflow and tags.get("overflow"):
        ov = ", ".join(str(x).replace("_", " ").strip() for x in tags["overflow"] if x)
        if ov:
            lines.append(f"Other: {ov}")
            any_el = True
    return "\n".join(lines) if any_el else "Write a scene."


def load_tags(path):
    """scene_id -> {facet: [labels], overflow: [...]}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("parse_ok", True):
                continue
            sid = r.get("scene_id")
            if sid is None:
                continue
            facets = r.get("facets", {}) or {}
            d = {f: facets.get(f, []) for f in FACETS}
            d["overflow"] = r.get("overflow", [])
            out[str(sid)] = d
    return out


def main():
    ap = argparse.ArgumentParser(description="Build the tagged corpus from manifest + tags.")
    ap.add_argument("--manifest", default="corpus.manifest.jsonl")
    ap.add_argument("--tags", default="tag_closed.jsonl")
    ap.add_argument("--out", default="corpus_tagged")
    ap.add_argument("--facets", nargs="*", default=FACETS,
                    help="subset/order of facets to include in the prompt")
    ap.add_argument("--include-overflow", action="store_true",
                    help="also put overflow terms in the prompt (noisier; default off)")
    args = ap.parse_args()

    tags = load_tags(args.tags)
    print(f"tags loaded for {len(tags):,} scenes", file=sys.stderr)

    out = {"train": [], "val": []}
    missing = empty = 0
    label_use = Counter()
    n = 0
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            sid, split, text = str(m["scene_id"]), m.get("split", "train"), m["text"]
            t = tags.get(sid)
            if t is None:
                missing += 1
                t = {}
            prompt = build_prompt(t, args.facets, args.include_overflow)
            if prompt == "Write a scene.":
                empty += 1
            else:
                for fct in args.facets:
                    for lab in (t.get(fct) or []):
                        label_use[f"{fct}:{str(lab).lower()}"] += 1
            out.setdefault(split, []).append({"prompt": prompt, "completion": text})
            n += 1

    for split in ("train", "val"):
        path = f"{args.out}.{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in out.get(split, []):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"examples: {n:,}  (train {len(out.get('train',[])):,} / "
          f"val {len(out.get('val',[])):,})")
    if missing:
        print(f"  ! {missing:,} manifest scenes had NO tags (not in {args.tags}) "
              f"-> generic prompt. Tag them with: tag_closed.py --ids-from {args.manifest}")
    print(f"  {empty:,} scenes got a generic 'Write a scene.' prompt (no labels)")
    print(f"  distinct control labels used: {len(label_use)}")
    print(f"  wrote {args.out}.train.jsonl, {args.out}.val.jsonl")


if __name__ == "__main__":
    main()

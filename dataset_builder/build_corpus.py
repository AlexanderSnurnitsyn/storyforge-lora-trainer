#!/usr/bin/env python3
"""
build_corpus.py — Build stage.  (Construction only; no metric *logic* lives here.)

Changes vs the original:
  * --no-split : treat each whole scene as ONE training example (scene = example),
    instead of smart-splitting into chunks. Scenes outside [min_tokens, max_tokens]
    are dropped (the long, splice-prone tail). This makes tag<->example mapping 1:1.
  * writes a MANIFEST ({scene_id, chunk_index, split, text}) so a tagged corpus can be
    built on EXACTLY the same scenes and splits (clean A/B: tags are the only diff).

Pipeline:
    read scenes -> attach scores -> keyword pre-filter -> (split | whole-scene)
    -> exact + near-dup removal -> diversity (MMR) -> train/val -> JSONL + manifest

Usage:
    # plain corpus, scene = example, fits a 2048 window with room for a tag-prompt:
    python build_corpus.py your.db --analysis dataset_analysis.csv \
        --no-split --max-tokens 1700 --max-examples 2200
"""
from __future__ import annotations

import argparse
import csv
import json
import random

import common
from config import Config
from keyword_filter import compile_matcher, distinct_hits, load_keywords
from metrics import compute_metrics
from ranking import length_score, score_scene
from splitter import smart_split


# ----------------------------------------------------------------------
def load_scores(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    scores = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            scores[row["scene_id"]] = float(row["score"])
    return scores


def dedup_and_diversify(chunks: list[dict], cfg) -> list[dict]:
    """chunks: [{text, scene_id, score, ...}] -> filtered/selected list."""
    hasher = common.MinHasher(cfg.minhash_perm, seed=cfg.seed)

    # 1) exact dedup + signatures
    seen_md5 = set()
    staged = []
    prog = common.Progress(len(chunks), label="Signing  ")
    for c in chunks:
        prog.update()
        h = common.md5(c["text"])
        if cfg.dedup_exact and h in seen_md5:
            continue
        seen_md5.add(h)
        c["_sig"] = hasher.signature(c["text"], cfg.minhash_k)
        staged.append(c)
    prog.done()

    # 2) near-dup removal via LSH band buckets (keep higher score)
    staged.sort(key=lambda x: x["score"], reverse=True)
    kept: list[dict] = []
    buckets: dict[int, list[int]] = {}
    prog = common.Progress(len(staged), label="Dedup    ")
    for c in staged:
        prog.update()
        bands = hasher.bands(c["_sig"])
        candidates = set()
        for b in bands:
            candidates.update(buckets.get(b, ()))
        is_dup = False
        for idx in candidates:
            if common.MinHasher.jaccard(c["_sig"], kept[idx]["_sig"]) >= cfg.near_dup_threshold:
                is_dup = True
                break
        if not is_dup:
            new_idx = len(kept)
            kept.append(c)
            for b in bands:
                buckets.setdefault(b, []).append(new_idx)
    prog.done()

    # 3) diversity selection (MMR) toward max_examples
    if not cfg.diversity_enabled or not cfg.max_examples or len(kept) <= cfg.max_examples:
        return kept[: cfg.max_examples] if cfg.max_examples else kept

    lam = cfg.diversity_lambda
    pool = kept[:]                       # already score-sorted desc
    selected = [pool.pop(0)]
    maxsim = [common.MinHasher.jaccard(c["_sig"], selected[0]["_sig"]) for c in pool]

    prog = common.Progress(cfg.max_examples, label="Diversity")
    prog.update()
    while pool and len(selected) < cfg.max_examples:
        best_i, best_val = 0, -1e18
        for i in range(len(pool)):
            val = lam * pool[i]["score"] - (1 - lam) * maxsim[i]
            if val > best_val:
                best_val, best_i = val, i
        chosen = pool.pop(best_i)
        chosen_sig = chosen["_sig"]
        maxsim.pop(best_i)
        selected.append(chosen)
        prog.update()
        for i, c in enumerate(pool):
            s = common.MinHasher.jaccard(c["_sig"], chosen_sig)
            if s > maxsim[i]:
                maxsim[i] = s
    prog.done()
    return selected


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build stage — construct the corpus.")
    ap.add_argument("db")
    ap.add_argument("--config", default=None)
    ap.add_argument("--analysis", default=None, help="dataset_analysis.csv from Stage 1")
    ap.add_argument("--keywords-file", default=None)
    ap.add_argument("--mode", choices=["row", "scene"], default=None)
    ap.add_argument("--out", default=None, help="output prefix")
    ap.add_argument("--max-examples", "--rows", "--limit", dest="max_examples",
                    type=int, default=None,
                    help="cap total examples in the final JSONL (train+val combined)")
    ap.add_argument("--all", action="store_true", help="disable keyword pre-filter")
    ap.add_argument("--no-split", action="store_true",
                    help="scene = one example (no smart_split); drop scenes outside "
                         "[min_tokens, max_tokens]. Recommended for the tagged-corpus A/B.")
    ap.add_argument("--min-tokens", type=int, default=None, help="override cfg.min_tokens")
    ap.add_argument("--max-tokens", type=int, default=None, help="override cfg.max_tokens")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.keywords_file:
        cfg.keywords_file = args.keywords_file
    if args.mode:
        cfg.mode = args.mode
    if args.max_examples is not None:
        cfg.max_examples = args.max_examples
    if args.all:
        cfg.keyword_filter_enabled = False
    if args.min_tokens is not None:
        cfg.min_tokens = args.min_tokens
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens
    no_split = args.no_split or not cfg.split_enabled
    prefix = args.out or cfg.out_prefix

    matcher = compile_matcher(load_keywords(cfg.keywords_file))
    scores = load_scores(args.analysis)

    rejected_kw = 0
    rejected_len = 0
    chunks: list[dict] = []
    n_scenes = 0
    total = common.count_scenes(args.db, cfg.table, cfg.text_col, cfg.scene_col, cfg.mode)
    prog = common.Progress(total, label="Building ")
    for scene in common.iter_scenes(
        args.db, cfg.table, cfg.text_col, cfg.scene_col, cfg.pos_col, cfg.mode
    ):
        n_scenes += 1
        prog.update()
        text = common.clean(scene.text)

        # keyword pre-filter (remove obviously unrelated scenes)
        if cfg.keyword_filter_enabled and distinct_hits(text, matcher) < cfg.min_keyword_hits:
            rejected_kw += 1
            continue

        # score: from analysis CSV, else compute now
        if scene.scene_id in scores:
            base_score = scores[scene.scene_id]
        else:
            m = compute_metrics(text, matcher, cfg)
            base_score = score_scene(m, cfg)["score"]

        # whole-scene (scene = example) OR smart-split into chunks
        if no_split:
            pieces = [(0, text)]
        else:
            pieces = list(enumerate(smart_split(text, cfg)))

        for idx, piece in pieces:
            tok = common.estimate_tokens(piece, cfg.chars_per_token)
            if no_split:
                # whole-scene length gate: drop the too-short and the long tail
                if tok < cfg.min_tokens or tok > cfg.max_tokens:
                    rejected_len += 1
                    continue
            else:
                if tok < max(cfg.min_tokens, cfg.target_tokens // 6):
                    continue
            len_fit = length_score(tok, cfg.target_tokens, cfg.min_tokens, cfg.max_tokens)
            chunks.append({
                "text": piece,
                "scene_id": scene.scene_id,
                "chunk_index": idx,
                "original_length": len(text),
                "original_score": round(base_score, 5),
                "estimated_tokens": tok,
                "score": round(0.7 * base_score + 0.3 * len_fit, 5),
            })
    prog.done()

    selected = dedup_and_diversify(chunks, cfg)

    # train/val split
    random.seed(cfg.seed)
    random.shuffle(selected)
    n_val = max(1, int(len(selected) * cfg.val_frac)) if selected else 0
    val, train = selected[:n_val], selected[n_val:]

    def dump_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps({"text": it["text"]}, ensure_ascii=False) + "\n")

    # manifest: scene_id + split + text, so the tagged corpus aligns 1:1
    def dump_manifest(path, train, val):
        with open(path, "w", encoding="utf-8") as f:
            for split, items in (("train", train), ("val", val)):
                for it in items:
                    f.write(json.dumps({
                        "scene_id": it["scene_id"],
                        "chunk_index": it["chunk_index"],
                        "split": split,
                        "estimated_tokens": it["estimated_tokens"],
                        "text": it["text"],
                    }, ensure_ascii=False) + "\n")

    dump_jsonl(f"{prefix}.train.jsonl", train)
    dump_jsonl(f"{prefix}.val.jsonl", val)
    dump_manifest(f"{prefix}.manifest.jsonl", train, val)

    # chunk metadata CSV (no training text)
    meta_fields = ["scene_id", "chunk_index", "original_length",
                   "original_score", "estimated_tokens", "score"]
    with open(cfg.chunks_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=meta_fields)
        w.writeheader()
        for c in selected:
            w.writerow({k: c[k] for k in meta_fields})

    total_tok = sum(c["estimated_tokens"] for c in selected)
    print(f"Scenes read: {n_scenes:,}")
    if cfg.keyword_filter_enabled:
        print(f"  rejected (keyword < {cfg.min_keyword_hits}): {rejected_kw:,}")
    if no_split:
        print(f"  rejected (length outside [{cfg.min_tokens},{cfg.max_tokens}]): {rejected_len:,}")
        print(f"  mode: NO-SPLIT (scene = example)")
    print(f"Examples produced: {len(chunks):,}  ->  kept after dedup/diversity: {len(selected):,}")
    print(f"  train {len(train):,} / val {len(val):,}")
    print(f"  ~{total_tok:,} est. tokens total")
    print(f"  wrote {prefix}.train.jsonl, {prefix}.val.jsonl, "
          f"{prefix}.manifest.jsonl, {cfg.chunks_csv}")
    if total_tok < 75_000:
        print("  ! small corpus — keep epochs to 1–2 and watch val loss")


if __name__ == "__main__":
    main()

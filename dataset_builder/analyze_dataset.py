#!/usr/bin/env python3
"""
analyze_dataset.py — Stage 1 of the pipeline ("Analyze").

Reads EVERY scene from the database and computes a full metric profile plus a
quality score. Nothing is discarded, split, or shuffled here.

Output:
    dataset_analysis.csv   (one row per scene: id + metrics + score)
    dataset_summary.md     (high-level overview)

Usage:
    python analyze_dataset.py your.db
    python analyze_dataset.py your.db --config config.json --keywords-file keywords.txt
"""
from __future__ import annotations

import argparse
import csv
import statistics as stats

import common
from config import Config
from keyword_filter import compile_matcher, load_keywords
from metrics import compute_metrics
from ranking import score_scene


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 — analyze every scene.")
    ap.add_argument("db")
    ap.add_argument("--config", default=None)
    ap.add_argument("--keywords-file", default=None)
    ap.add_argument("--mode", choices=["row", "scene"], default=None)
    ap.add_argument("--out", default=None, help="analysis CSV path")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.keywords_file:
        cfg.keywords_file = args.keywords_file
    if args.mode:
        cfg.mode = args.mode
    out_csv = args.out or cfg.analysis_csv

    matcher = compile_matcher(load_keywords(cfg.keywords_file))

    total = common.count_scenes(args.db, cfg.table, cfg.text_col, cfg.scene_col, cfg.mode)
    prog = common.Progress(total, label="Analyzing")

    rows = []
    n = 0
    for scene in common.iter_scenes(
        args.db, cfg.table, cfg.text_col, cfg.scene_col, cfg.pos_col, cfg.mode
    ):
        m = compute_metrics(scene.text, matcher, cfg)
        sc = score_scene(m, cfg)
        row = {"scene_id": scene.scene_id, "score": sc["score"]}
        row.update({f"c_{k}": v for k, v in sc["components"].items()})
        row.update(m)
        rows.append(row)
        n += 1
        prog.update()
    prog.done()

    if not rows:
        print("No scenes found — check --mode and table/column names.")
        return

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # summary ------------------------------------------------------
    tokens = [r["token_est"] for r in rows]
    scores = [r["score"] for r in rows]
    kw = [r["keyword_hits"] for r in rows]
    summary = [
        "# Dataset Analysis Summary",
        "",
        f"- Scenes analyzed: **{n:,}**",
        f"- Token estimate — mean {int(stats.mean(tokens)):,}, "
        f"median {int(stats.median(tokens)):,}, max {max(tokens):,}",
        f"- Score — mean {stats.mean(scores):.3f}, "
        f"median {stats.median(scores):.3f}, max {max(scores):.3f}",
        f"- Keyword hits — mean {stats.mean(kw):.2f}, max {max(kw)}",
        f"- Scenes over max_tokens ({cfg.target_tokens}/{cfg.max_tokens}): "
        f"{sum(1 for t in tokens if t > cfg.max_tokens):,}",
        "",
        f"Written: `{out_csv}`",
    ]
    with open("dataset_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    print(f"Analyzed {n:,} scenes -> {out_csv}")
    print(f"  token median {int(stats.median(tokens)):,}  score median {stats.median(scores):.3f}")
    print("  wrote dataset_summary.md")


if __name__ == "__main__":
    main()
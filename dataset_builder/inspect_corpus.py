#!/usr/bin/env python3
"""
inspect_corpus.py — Validation stage.

Reads the produced JSONL (train + val) and reports problems before training:
token distribution, duplicate ratio, vocabulary diversity, keyword stats,
truncation risk, and train/val balance. Writes corpus_statistics.md.

Usage:
    python inspect_corpus.py corpus
    python inspect_corpus.py corpus --config config.json
"""
from __future__ import annotations

import argparse
import json
import statistics as stats

import common
from config import Config
from keyword_filter import compile_matcher, distinct_hits, load_keywords


def read_jsonl(path: str) -> list[str]:
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line)["text"])
    except FileNotFoundError:
        pass
    return out


def pct(x, total):
    return f"{(100.0 * x / total):.1f}%" if total else "0.0%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a built corpus.")
    ap.add_argument("prefix", help="corpus prefix (expects <prefix>.train.jsonl / .val.jsonl)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--keywords-file", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.keywords_file:
        cfg.keywords_file = args.keywords_file
    matcher = compile_matcher(load_keywords(cfg.keywords_file))

    train = read_jsonl(f"{args.prefix}.train.jsonl")
    val = read_jsonl(f"{args.prefix}.val.jsonl")
    allc = train + val
    n = len(allc)
    if n == 0:
        print("No chunks found — did you run build_corpus.py?")
        return

    toks = [common.estimate_tokens(t, cfg.chars_per_token) for t in allc]
    md5s = [common.md5(t) for t in allc]
    dup = n - len(set(md5s))
    over = sum(1 for t in toks if t > cfg.max_tokens)
    kw_hits = [distinct_hits(t, matcher) for t in allc]
    zero_kw = sum(1 for h in kw_hits if h == 0)

    # corpus-wide vocabulary diversity
    vocab, total_words = set(), 0
    for t in allc:
        ws = common.words(t)
        total_words += len(ws)
        vocab.update(ws)
    ttr = (len(vocab) / total_words) if total_words else 0.0

    def histline(lo, hi):
        c = sum(1 for t in toks if lo <= t < hi)
        bar = "#" * int(40 * c / n)
        return f"  {lo:>5}-{hi:<5} {c:>6} {pct(c, n):>7} {bar}"

    edges = [0, cfg.min_tokens, cfg.target_tokens // 2, cfg.target_tokens,
             cfg.max_tokens, cfg.max_tokens * 2, 10**9]

    report = [
        "# Corpus Statistics",
        "",
        f"- Total chunks: **{n:,}**  (train {len(train):,} / val {len(val):,}, "
        f"val {pct(len(val), n)})",
        f"- Tokens — mean {int(stats.mean(toks)):,}, median {int(stats.median(toks)):,}, "
        f"min {min(toks):,}, max {max(toks):,}",
        f"- Target token window: {cfg.target_tokens} (max {cfg.max_tokens})",
        f"- Over max_tokens (truncation risk): {over:,} ({pct(over, n)})",
        f"- Exact duplicate ratio: {pct(dup, n)}",
        f"- Vocabulary diversity (corpus TTR): {ttr:.4f}  ({len(vocab):,} unique words)",
        f"- Chunks with 0 keyword hits: {zero_kw:,} ({pct(zero_kw, n)})",
        f"- Mean keyword hits/chunk: {stats.mean(kw_hits):.2f}",
        "",
        "## Token length distribution",
        "```",
    ]
    for lo, hi in zip(edges[:-1], edges[1:]):
        report.append(histline(lo, hi))
    report.append("```")
    report.append("")

    # flags --------------------------------------------------------
    flags = []
    if n < 50:
        flags.append("Very few chunks — corpus may be too small for stable training.")
    if over / n > 0.05:
        flags.append(f"{pct(over, n)} of chunks exceed max_tokens — check the splitter / max_tokens.")
    if dup / n > 0.02:
        flags.append(f"Duplicate ratio {pct(dup, n)} is high — exact dedup may be off.")
    if zero_kw / n > 0.20:
        flags.append(f"{pct(zero_kw, n)} of chunks have no keyword hits — review keyword filter.")
    if len(val) == 0:
        flags.append("No validation chunks — increase val_frac.")
    report.append("## Flags")
    report.extend([f"- ⚠️ {x}" for x in flags] or ["- ✅ No problems detected."])

    with open("corpus_statistics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print("\nwrote corpus_statistics.md")


if __name__ == "__main__":
    main()

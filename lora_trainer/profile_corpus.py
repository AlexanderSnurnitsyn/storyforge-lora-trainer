#!/usr/bin/env python3
"""
profile_corpus.py — cheap, MODEL-FREE profiling to decide the tagging strategy
BEFORE spending GPU hours.

Answers, from the raw DB only (seconds, stdlib):
  A. how many scenes pass the keyword filter (>= N distinct hits) vs not
  B. segments-per-scene x keyword pass/fail   (single vs multi among the real population)
  C. per-segment-bucket coverage: scenes, tokens, and CUMULATIVE coverage if you cap
     segments at K  -> tells you whether "just take 1-3 segments" loses much
  D. token-length distribution, single-segment vs multi-segment

Optional (needs the model, reuses tag_pilot):
  --time-model NAME : time real tagging on a few scenes per segment bucket and project
                      total runtime over the keyword-pass population, per cap.

Usage (model-free):
  python profile_corpus.py your.db --builder-path ../dataset_builder \
      --keywords ../dataset_builder/keywords.txt --min-keyword-hits 2

Add timing:
  ... --time-model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated --time-samples 5
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict

# reuse the exact same helpers the pilot uses (stdlib-only; no torch imported here)
from tag_pilot import clean, est_tokens, get_keyword_tools


# --------------------------------------------------------------------------- #
def read_profiles(db, table, text_col, scene_col, pos_col, matcher, hits_fn, min_hits):
    """Stream segments grouped by scene_id; return per-scene profiles.

    Each profile: (scene_id, n_segments, est_tokens, kw_pass: bool)
    Matches production iter_scenes: WHERE text IS NOT NULL, joined by '\\n\\n',
    cleaned, then estimated.
    """
    con = sqlite3.connect(db)
    cur = con.cursor()
    q = (f"SELECT {scene_col}, {text_col} FROM {table} "
         f"WHERE {text_col} IS NOT NULL ORDER BY {scene_col}, {pos_col}")
    rows = []

    def finalize(sid, buf):
        if not buf:
            return None
        text = clean("\n\n".join(buf))
        if not text:
            return None
        toks = est_tokens(text)
        kw_pass = (hits_fn(text, matcher) >= min_hits) if hits_fn else True
        return (sid, len(buf), toks, kw_pass)

    cur_id, buf = None, []
    for sid, txt in cur.execute(q):
        if sid != cur_id and buf:
            r = finalize(cur_id, buf)
            if r:
                rows.append(r)
            buf = []
        cur_id = sid
        if txt:
            buf.append(txt)
    if buf:
        r = finalize(cur_id, buf)
        if r:
            rows.append(r)
    con.close()
    return rows


# --------------------------------------------------------------------------- #
def pct(a, b):
    return f"{100*a/b:5.1f}%" if b else "   -  "


def median(xs):
    if not xs:
        return 0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) // 2


def percentile(xs, p):
    if not xs:
        return 0
    s = sorted(xs)
    i = min(len(s) - 1, int(p / 100 * len(s)))
    return s[i]


def seg_bucket(n, cap=6):
    return n if n < cap else cap  # cap label means ">= cap"


# --------------------------------------------------------------------------- #
def report(rows, min_hits, kw_on):
    total = len(rows)
    pass_rows = [r for r in rows if r[3]]
    fail_rows = [r for r in rows if not r[3]]

    print("\n" + "=" * 66)
    print(f"CORPUS PROFILE   scenes={total:,}")
    print("=" * 66)

    # ---- A: keyword filter ----
    print(f"\n[A] keyword filter" + (f" (>= {min_hits} distinct hits)" if kw_on else " (OFF)"))
    if kw_on:
        print(f"    pass: {len(pass_rows):,} ({pct(len(pass_rows),total)})   "
              f"fail: {len(fail_rows):,} ({pct(len(fail_rows),total)})")
    else:
        print("    no --keywords/--builder-path given; treating all scenes as 'pass'.")

    # ---- B: segments x keyword ----
    print("\n[B] segments-per-scene  x  keyword")
    print("    segs   kw-pass    kw-fail        all")
    by_pass = defaultdict(int)
    by_fail = defaultdict(int)
    for _, ns, _, p in rows:
        (by_pass if p else by_fail)[seg_bucket(ns)] += 1
    for b in sorted(set(by_pass) | set(by_fail)):
        lbl = f"{b}" if b < 6 else "6+"
        ap, af = by_pass.get(b, 0), by_fail.get(b, 0)
        print(f"    {lbl:>4}   {ap:>7,}    {af:>7,}    {ap+af:>7,}")
    multi_pass = sum(v for k, v in by_pass.items() if k >= 2)
    print(f"    -> kw-pass single-segment: {by_pass.get(1,0):,}   "
          f"multi-segment: {multi_pass:,}")

    # ---- C: coverage if you cap segment count (kw-pass population) ----
    pop = pass_rows if kw_on else rows
    print(f"\n[C] coverage by segment cap   (population = kw-pass: {len(pop):,} scenes)")
    print("    segs    scenes   %scn   medTok   p90Tok      tokens   %tok | cum%scn cum%tok")
    tot_scn = len(pop)
    tot_tok = sum(r[2] for r in pop)
    cum_s = cum_t = 0
    # group by exact seg count, but collapse the long tail at 6+
    groups = defaultdict(list)
    for _, ns, tk, _ in pop:
        groups[seg_bucket(ns)].append(tk)
    for b in sorted(groups):
        lbl = f"{b}" if b < 6 else "6+"
        toks = groups[b]
        scn = len(toks)
        tsum = sum(toks)
        cum_s += scn
        cum_t += tsum
        print(f"    {lbl:>4}   {scn:>7,}  {pct(scn,tot_scn)}  {median(toks):>6,}   "
              f"{percentile(toks,90):>6,}  {tsum:>10,}  {pct(tsum,tot_tok)} |"
              f"  {pct(cum_s,tot_scn)} {pct(cum_t,tot_tok)}")
    print("    (cum% = what you KEEP if you cap segments at that row)")

    # ---- D: token distribution single vs multi (kw-pass) ----
    print(f"\n[D] token length   single-segment vs multi-segment   (kw-pass)")
    edges = [0, 400, 800, 1200, 1600, 2000, 3000, 10**9]
    labels = ["<400", "400-800", "800-1200", "1200-1600", "1600-2000", "2000-3000", "3000+"]
    single = [0] * len(labels)
    multi = [0] * len(labels)
    for _, ns, tk, _ in pop:
        for i in range(len(labels)):
            if edges[i] <= tk < edges[i + 1]:
                (single if ns == 1 else multi)[i] += 1
                break
    print("    tokens          single     multi")
    for i, lb in enumerate(labels):
        print(f"    {lb:>12}   {single[i]:>7,}   {multi[i]:>7,}")

    return pop, groups


# --------------------------------------------------------------------------- #
def run_timing(args, pop, groups):
    """Time real tagging per segment bucket; project totals. Needs the model."""
    import time
    import random
    from tag_pilot import load_model, tag_scene, build_marked, fetch_segments

    # collect scene_ids per bucket from the kw-pass population we already scanned
    # (we only kept profiles, not ids in pop tuples -> re-derive by re-querying ids)
    # pop tuples are (sid, ns, tk, kw); sid is index 0
    by_bucket_ids = defaultdict(list)
    for sid, ns, tk, _ in pop:
        if tk <= args.max_scene_tokens:
            by_bucket_ids[seg_bucket(ns)].append(sid)

    print(f"\nLoading model for timing: {args.time_model}", file=sys.stderr)
    model, tok = load_model(args.time_model, args.max_seq_len, not args.no_4bit)
    rnd = random.Random(args.seed)

    print("\n[T] tagging speed by segment bucket  "
          f"(sample {args.time_samples}/bucket, greedy)")
    print("    segs   n   sec/scene   medTok(sample)")
    per_bucket_sec = {}
    for b in sorted(by_bucket_ids):
        ids = by_bucket_ids[b][:]
        rnd.shuffle(ids)
        ids = ids[: args.time_samples]
        times, toks = [], []
        for sid in ids:
            segs = fetch_segments(args.db, args.table, args.text_col,
                                  args.scene_col, args.pos_col, sid)
            _, marked, _ = build_marked(segs)
            if not marked:
                continue
            t0 = time.time()
            tag_scene(model, tok, marked, args.max_new_tokens)
            times.append(time.time() - t0)
            toks.append(est_tokens(marked))
        if times:
            avg = sum(times) / len(times)
            per_bucket_sec[b] = avg
            lbl = f"{b}" if b < 6 else "6+"
            print(f"    {lbl:>4}  {len(times):>2}   {avg:>8.2f}   {median(toks):>6,}")

    # project totals over the FULL kw-pass population (counts from groups)
    print("\n[T] projected full-run time over kw-pass population")
    counts = {b: len(v) for b, v in groups.items()}
    cum_sec = 0
    cum_scn = 0
    tot_scn = sum(counts.values())
    for b in sorted(counts):
        sec = per_bucket_sec.get(b)
        if sec is None:
            continue
        bucket_total = counts[b] * sec
        cum_sec += bucket_total
        cum_scn += counts[b]
        lbl = f"<={b}" if b < 6 else "all"
        print(f"    cap segments {lbl:>4}:  {cum_scn:>6,} scenes  "
              f"~{cum_sec/3600:5.1f} h   ({pct(cum_scn,tot_scn)} of scenes)")
    print("    (each row = cumulative cost+coverage if you stop at that segment cap)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Model-free corpus profiling.")
    ap.add_argument("db")
    ap.add_argument("--keywords", default=None)
    ap.add_argument("--builder-path", default=None,
                    help="path to dataset_builder/ for the PRODUCTION keyword filter")
    ap.add_argument("--min-keyword-hits", type=int, default=2)
    ap.add_argument("--table", default="segments")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--scene-col", default="scene_id")
    ap.add_argument("--pos-col", default="pos")
    # timing (optional)
    ap.add_argument("--time-model", default=None,
                    help="model name/path; enables per-segment timing + projection")
    ap.add_argument("--time-samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--max-scene-tokens", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    kw_on = bool(args.keywords or args.builder_path)
    matcher = hits_fn = None
    if kw_on:
        matcher, hits_fn, src = get_keyword_tools(args.keywords, args.builder_path)
        print(f"keyword filter: {src} (>= {args.min_keyword_hits} hits)", file=sys.stderr)

    print("Scanning DB...", file=sys.stderr)
    rows = read_profiles(args.db, args.table, args.text_col, args.scene_col,
                         args.pos_col, matcher, hits_fn, args.min_keyword_hits)
    if not rows:
        sys.exit("No scenes found — check table/column names.")

    pop, groups = report(rows, args.min_keyword_hits, kw_on)

    if args.time_model:
        run_timing(args, pop, groups)
    else:
        print("\n(add --time-model NAME to measure tagging speed per segment bucket)")


if __name__ == "__main__":
    main()

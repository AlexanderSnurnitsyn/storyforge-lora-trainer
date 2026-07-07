#!/usr/bin/env python3
"""
tag_pilot.py — PILOT: scene tagging + junction-coherence detection.

Goal: validate, on a SMALL sample, whether a local instruct model can
  (a) produce useful structured tags,
  (b) extract verbatim entities without hallucinating,
  (c) judge scene coherence and locate breaks AT SEGMENT SEAMS.

This is a DIAGNOSTIC, not a corpus pass. The production `iter_scenes`
concatenates segments and `clean()` collapses newlines, so segment seams are
lost in the final text. This script therefore reads `segments` directly,
inserts ⟦Jn⟧ seam markers for the model to judge, and records each seam's
character offset in the clean (marker-free) text so a later build step can do
extractive truncation without regenerating anything.

It prints three decision summaries:
  1. coherence_score histogram        -> how much "scenario 1" (spliced scenes) exists
  2. % multi-seg scenes with breaks   -> do breaks really land on segment seams
  3. % entities NOT found in source   -> entity-hallucination rate (validates grounding)

Run in the TRAINER environment (needs unsloth + torch + a CUDA GPU):

    python tag_pilot.py path/to/your.db --sample 50 --min-segments 2

Output: tag_pilot.jsonl  (one record per scene) + printed summaries.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from collections import Counter

# --- text cleaning: mirror the production common.clean() exactly -------------
_WS = re.compile(r"[ \t]+")
_NL3 = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _NL3.sub("\n\n", text)
    return text.strip()


def est_tokens(text: str, cpt: float = 4.0) -> int:
    return max(1, int(len(text) / cpt))


# --------------------------------------------------------------------------- #
# Keyword matching (optional) — prefer the production module for EXACT parity
# with the real corpus filter; fall back to a faithful built-in matcher.
# --------------------------------------------------------------------------- #
def _load_keywords_builtin(path):
    kws, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w or w.startswith("#"):
                continue
            wl = w.lower()
            if wl not in seen:
                seen.add(wl)
                kws.append(wl)
    return kws


def _compile_matcher_builtin(keywords):
    if not keywords:
        return None
    pat = r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b"
    return re.compile(pat, re.IGNORECASE)


def _distinct_hits_builtin(text, matcher):
    if matcher is None:
        return 0
    return len({m.group(0).lower() for m in matcher.finditer(text)})


def get_keyword_tools(keywords_path, builder_path):
    """Return (matcher, distinct_hits_fn, source_label).

    If builder_path is given, import the production keyword_filter for exact
    parity (keywords_path=None there means the module's own default keywords.txt).
    Otherwise build a faithful in-house matcher from keywords_path.
    """
    if builder_path:
        sys.path.insert(0, builder_path)
        try:
            import keyword_filter as kf  # type: ignore
            matcher = kf.compile_matcher(kf.load_keywords(keywords_path))
            return matcher, kf.distinct_hits, "production"
        except Exception as e:  # noqa: BLE001
            print(f"  (! could not import production keyword_filter from "
                  f"{builder_path}: {e}\n     falling back to built-in matcher)",
                  file=sys.stderr)
    if not keywords_path:
        sys.exit("Keyword filtering requested but no --keywords file given "
                 "(and no importable production module).")
    matcher = _compile_matcher_builtin(_load_keywords_builtin(keywords_path))
    return matcher, _distinct_hits_builtin, "built-in"


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
def segment_count_distribution(db, table, text_col, scene_col):
    """Counter: {n_segments_in_scene: how_many_scenes}. Shows base rate of
    multi-segment scenes (only those can have a seam artifact)."""
    con = sqlite3.connect(db)
    try:
        q = (f"SELECT cnt, COUNT(*) FROM ("
             f"  SELECT {scene_col} AS sid, COUNT(*) AS cnt FROM {table} "
             f"  WHERE {text_col} IS NOT NULL GROUP BY {scene_col}"
             f") GROUP BY cnt ORDER BY cnt")
        return Counter({row[0]: row[1] for row in con.execute(q)})
    finally:
        con.close()


def candidate_scene_ids(db, table, text_col, scene_col, min_segments, seed):
    """All scene_ids with >= min_segments, shuffled (consumed lazily so the
    keyword filter can stream until enough scenes are collected)."""
    con = sqlite3.connect(db)
    try:
        q = (f"SELECT {scene_col} FROM {table} WHERE {text_col} IS NOT NULL "
             f"GROUP BY {scene_col} HAVING COUNT(*) >= ?")
        ids = [row[0] for row in con.execute(q, (min_segments,))]
    finally:
        con.close()
    random.Random(seed).shuffle(ids)
    return ids


def fetch_segments(db, table, text_col, scene_col, pos_col, scene_id):
    con = sqlite3.connect(db)
    try:
        q = (f"SELECT {text_col} FROM {table} "
             f"WHERE {scene_col} = ? AND {text_col} IS NOT NULL ORDER BY {pos_col}")
        return [r[0] for r in con.execute(q, (scene_id,))]
    finally:
        con.close()


def build_marked(segments):
    """Returns (clean_joined, marked_for_model, seam_offsets).

    seam_offsets[i] is the char index in `clean_joined` where segment (i+1)
    begins — i.e. the start of the text AFTER junction J(i+1). To keep only the
    coherent prefix up to a broken junction Jn, slice clean_joined[:offset]
    where offset is the seam *before* segment n.
    """
    segs = [clean(s) for s in segments if s and s.strip()]
    if not segs:
        return "", "", []
    sep = "\n\n"
    clean_joined = sep.join(segs)

    seam_offsets = []
    running = len(segs[0])
    for s in segs[1:]:
        seam_offsets.append(running + len(sep))  # index where this segment starts
        running += len(sep) + len(s)

    parts = [segs[0]]
    for n, s in enumerate(segs[1:], start=1):
        parts.append(f"{sep}\u27e6J{n}\u27e6{sep}{s}")
    marked = "".join(parts)
    return clean_joined, marked, seam_offsets


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM = (
    "You are a precise literary-metadata extractor. "
    "Output ONLY one valid JSON object matching the schema. "
    "No prose, no markdown fences, no commentary."
)

USER_TMPL = """The text below is one scene. Internal segment seams are marked \u27e6J1\u27e6, \u27e6J2\u27e6, ...
The markers are NEUTRAL: most scenes read straight through them. Only flag a seam if the
text genuinely fails to connect across it (an action contradicts what just happened, the
subject/location silently changes, a non-sequitur).

Extract metadata. Rules:
- genre, setting, pacing, pov: choose the single best fit.
- tone, content_elements: short generic terms for what HAPPENS (e.g. "duel", "betrayal",
  "ritual magic"). Describe only what is in the text; do NOT invent.
- characters: do NOT extract personal names — we train style, not the cast. Skip them.
- places: NAMED locations, VERBATIM.
- invented_terms: invented vocabulary unique to this world — named magic systems,
  mechanics, factions, items, species, titles. VERBATIM. THIS is the field that matters
  for controllability. Do NOT put plain character names here.
- For places/invented_terms: copy exact spelling; if unsure, omit; NEVER invent.
- coherence_score: 5 fully coherent; 4 minor awkwardness; 3 noticeable but readable;
  2 a clear contradiction/non-sequitur; 1 incoherent / multiple breaks.
- broken_junctions: numbers of the \u27e6Jn\u27e6 seams the text does NOT connect across.
  [] if none (or if there are no seams).
- Judge only the text as written; do NOT speculate about hidden conditions or missing context.

Schema (return EXACTLY these keys):
{{"genre":"","tone":[],"setting":"","pov":"first|second|third-limited|third-omniscient|unclear","pacing":"slow|moderate|fast","content_elements":[],"places":[],"invented_terms":[],"coherence_score":0,"coherence_notes":"","broken_junctions":[]}}

SCENE:
{scene}"""

SCHEMA_KEYS = {"genre", "tone", "setting", "pov", "pacing", "content_elements",
               "places", "invented_terms",
               "coherence_score", "coherence_notes", "broken_junctions"}


# --------------------------------------------------------------------------- #
# JSON extraction (defensive)
# --------------------------------------------------------------------------- #
def extract_json(s: str):
    """Pull the first balanced {...} block, tolerating ```json fences/prose."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    start = s.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


# --------------------------------------------------------------------------- #
# Model (Unsloth, inference only)
# --------------------------------------------------------------------------- #
def load_model(model_name, max_seq_len, load_in_4bit):
    from unsloth import FastLanguageModel  # noqa: must precede transformers
    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def generate(model, tok, scene_text, max_new_tokens):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(scene=scene_text)}]
    inputs = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    attn = inputs.new_ones(inputs.shape)   # pad==eos -> set mask explicitly
    out = model.generate(
        input_ids=inputs,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,                    # greedy: deterministic extraction
        use_cache=True,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)


def tag_scene(model, tok, scene_text, max_new_tokens):
    """One generation + one retry on parse failure."""
    raw = generate(model, tok, scene_text, max_new_tokens)
    obj = extract_json(raw)
    if obj is None:
        raw = generate(model, tok,
                       scene_text + "\n\n(Reminder: output ONLY the JSON object.)",
                       max_new_tokens)
        obj = extract_json(raw)
    return obj, raw


# --------------------------------------------------------------------------- #
# Entity grounding check
# --------------------------------------------------------------------------- #
def unfound_entities(entities, source):
    src = source.lower()
    miss = []
    for e in entities or []:
        if not isinstance(e, str):
            continue
        probe = e.strip().strip("\"'").lower()
        if probe and probe not in src:
            miss.append(e)
    return miss


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Pilot: tagging + junction coherence.")
    ap.add_argument("db")
    ap.add_argument("--model", default="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--min-segments", type=int, default=2,
                    help="only sample scenes with >= this many segments "
                         "(seam artifacts can only occur when >=2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tag_pilot.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--max-scene-tokens", type=int, default=2800,
                    help="skip scenes whose estimated tokens exceed this (prompt budget)")
    ap.add_argument("--keywords", default=None,
                    help="path to keywords.txt; enables the keyword pre-filter so the "
                         "sample matches the real (keyword-filtered) corpus population")
    ap.add_argument("--min-keyword-hits", type=int, default=2,
                    help="distinct keyword hits required to keep a scene (match config)")
    ap.add_argument("--builder-path", default=None,
                    help="path to dataset_builder/ — if given, uses the PRODUCTION "
                         "keyword_filter module for exact parity (recommended)")
    ap.add_argument("--inject-controls", type=int, default=0,
                    help="after tagging, splice N pairs of unrelated scenes and tag them; "
                         "validates the coherence judge (it SHOULD flag every splice)")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--table", default="segments")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--scene-col", default="scene_id")
    ap.add_argument("--pos-col", default="pos")
    args = ap.parse_args()

    # base-rate context: how many scenes can even have a seam?
    dist = segment_count_distribution(args.db, args.table, args.text_col, args.scene_col)
    total_scenes = sum(dist.values())
    multi = sum(v for k, v in dist.items() if k >= 2)
    print(f"Scenes total: {total_scenes:,}   multi-segment (>=2): {multi:,} "
          f"({100*multi/max(1,total_scenes):.1f}%)", file=sys.stderr)
    head = ", ".join(f"{k}:{dist[k]}" for k in sorted(dist)[:8])
    print(f"  segment-count distribution (first buckets): {head}", file=sys.stderr)

    ids = candidate_scene_ids(args.db, args.table, args.text_col, args.scene_col,
                              args.min_segments, args.seed)
    if not ids:
        sys.exit("No scenes matched --min-segments. Lower it or check the DB.")

    kw_enabled = bool(args.keywords or args.builder_path)
    matcher = hits_fn = None
    kw_source = "off"
    if kw_enabled:
        matcher, hits_fn, kw_source = get_keyword_tools(args.keywords, args.builder_path)
    print(f"Candidates (min_segments={args.min_segments}): {len(ids):,}   "
          f"keyword filter: {kw_source}"
          + (f" (>= {args.min_keyword_hits} hits)" if kw_enabled else ""),
          file=sys.stderr)

    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tok = load_model(args.model, args.max_seq_len, not args.no_4bit)

    records = []
    parse_fail = skipped = kw_reject = 0
    target = args.sample
    tagged = 0
    for sid in ids:
        if tagged >= target:
            break
        segs = fetch_segments(args.db, args.table, args.text_col,
                              args.scene_col, args.pos_col, sid)
        clean_joined, marked, seams = build_marked(segs)
        if not clean_joined:
            continue
        if kw_enabled and hits_fn(clean_joined, matcher) < args.min_keyword_hits:
            kw_reject += 1
            continue
        if est_tokens(marked) > args.max_scene_tokens:
            skipped += 1
            continue

        tagged += 1
        obj, raw = tag_scene(model, tok, marked, args.max_new_tokens)
        print(f"\r  {tagged}/{target}  ", end="", file=sys.stderr, flush=True)

        if obj is None:
            parse_fail += 1
            records.append({"scene_id": sid, "n_segments": len(segs),
                            "parse_ok": False, "text": clean_joined, "raw": raw[:600]})
            continue

        places = obj.get("places", [])
        terms = obj.get("invented_terms", [])
        all_ents = list(places) + list(terms)
        miss = unfound_entities(all_ents, clean_joined)
        records.append({
            "scene_id": sid,
            "n_segments": len(segs),
            "n_junctions": len(seams),
            "seam_offsets": seams,            # char index where each later segment starts
            "parse_ok": True,
            "control": False,
            "text": clean_joined,             # included so the viewer can show it
            "missing_schema_keys": sorted(SCHEMA_KEYS - set(obj.keys())),
            "tags": {k: obj.get(k) for k in
                     ("genre", "tone", "setting", "pov", "pacing", "content_elements")},
            "places": places,
            "invented_terms": terms,
            "entities_unfound": miss,
            "coherence_score": obj.get("coherence_score"),
            "coherence_notes": obj.get("coherence_notes", ""),
            "broken_junctions": obj.get("broken_junctions", []),
        })
    print("", file=sys.stderr)
    if kw_enabled:
        print(f"  (keyword filter rejected {kw_reject:,} candidates along the way)",
              file=sys.stderr)

    # ----- control injection: splice two unrelated scenes, expect a flag -----
    control_results = []
    if args.inject_controls > 0:
        real_texts = [r["text"] for r in records if r.get("parse_ok") and r.get("text")]
        rnd = random.Random(args.seed + 1)
        rnd.shuffle(real_texts)
        pairs = min(args.inject_controls, len(real_texts) // 2)
        print(f"Injecting {pairs} control splices (two unrelated scenes each)...",
              file=sys.stderr)
        for j in range(pairs):
            a, b = real_texts[2 * j], real_texts[2 * j + 1]
            spliced = f"{a}\n\n\u27e6J1\u27e6\n\n{b}"
            if est_tokens(spliced) > args.max_scene_tokens:
                # truncate each half so the splice still fits
                half = (args.max_scene_tokens // 2) * 4
                spliced = f"{a[:half]}\n\n\u27e6J1\u27e6\n\n{b[:half]}"
            obj, raw = tag_scene(model, tok, spliced, args.max_new_tokens)
            print(f"\r  control {j+1}/{pairs}  ", end="", file=sys.stderr, flush=True)
            cs = obj.get("coherence_score") if obj else None
            bj = obj.get("broken_junctions", []) if obj else []
            caught = bool(bj) or (isinstance(cs, int) and cs <= 3)
            rec = {"scene_id": f"__control_{j}", "control": True, "parse_ok": obj is not None,
                   "expected_break": 1, "coherence_score": cs, "broken_junctions": bj,
                   "caught": caught, "text": spliced}
            control_results.append(rec)
            records.append(rec)
        print("", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------ decision summaries ------------------
    ok = [r for r in records if r.get("parse_ok") and not r.get("control")]
    print("\n" + "=" * 60)
    print(f"PILOT SUMMARY  (n={len(records)}, parsed={len(ok)}, "
          f"parse_fail={parse_fail}, skipped_long={skipped})")
    if kw_enabled:
        print(f"  keyword filter: {kw_source}, >= {args.min_keyword_hits} hits "
              f"(rejected {kw_reject:,} candidates)")
    print("=" * 60)

    print("\n[1] coherence_score histogram  (decides: how much scenario-1 exists)")
    ch = Counter(r["coherence_score"] for r in ok)
    for s in (5, 4, 3, 2, 1):
        c = ch.get(s, 0)
        bar = "#" * c
        print(f"    {s}: {c:3d} {bar}")
    other = [k for k in ch if k not in (1, 2, 3, 4, 5)]
    if other:
        print(f"    out-of-range/None: {sum(ch[k] for k in other)}")
    low = sum(ch.get(s, 0) for s in (1, 2))
    if ok:
        print(f"    -> low (<=2): {low}/{len(ok)} = {100*low/len(ok):.0f}%")

    print("\n[2] seam-break rate  (decides: do breaks land on segment seams)")
    with_seams = [r for r in ok if r.get("n_junctions", 0) >= 1]
    broke = [r for r in with_seams if r.get("broken_junctions")]
    if with_seams:
        print(f"    scenes with >=1 seam: {len(with_seams)}")
        print(f"    of those, >=1 broken seam: {len(broke)} "
              f"= {100*len(broke)/len(with_seams):.0f}%")
    else:
        print("    (no multi-segment scenes parsed)")

    print("\n[3] entity grounding  (decides: is verbatim extraction trustworthy)")
    n_place = sum(len(r.get("places", [])) for r in ok)
    n_term = sum(len(r.get("invented_terms", [])) for r in ok)
    tot_ent = n_place + n_term
    tot_miss = sum(len(r.get("entities_unfound", [])) for r in ok)
    if tot_ent:
        print(f"    extracted: {tot_ent}  (places {n_place}, invented_terms {n_term})")
        print(f"    not found in source: {tot_miss} = {100*tot_miss/tot_ent:.0f}%")
        print(f"    -> invented_terms are the control signal; names are not extracted.")
    else:
        print("    (no entities extracted)")

    if control_results:
        caught = sum(1 for r in control_results if r.get("caught"))
        n = len(control_results)
        print("\n[4] CONTROL check  (decides: is the coherence judge trustworthy AT ALL)")
        print(f"    injected splices: {n}   flagged as broken: {caught} "
              f"= {100*caught/n:.0f}%")
        if caught < n:
            print("    !! judge MISSED known-broken scenes — coherence_score is NOT")
            print("       reliable; the 'all 5' result above is likely non-discrimination.")
        else:
            print("    judge caught every splice — clean coherence results are credible.")

    missing_keys = sum(1 for r in ok if r.get("missing_schema_keys"))
    if missing_keys:
        print(f"\n[!] {missing_keys}/{len(ok)} scenes had missing schema keys "
              f"(check prompt strictness before scaling).")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

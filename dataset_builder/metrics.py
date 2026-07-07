#!/usr/bin/env python3
"""
metrics.py — compute a complete metric profile for a single scene/chunk.

Stage 1 ("Analyze") reads every scene and calls compute_metrics(). Nothing is
discarded or modified here — this module only measures.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Pattern

import common
from keyword_filter import distinct_hits, keyword_density, keyword_ratio

# dialogue heuristics -------------------------------------------------
_DQ = re.compile(r"[\"“«][^\"”»\n]{1,2000}?[\"”»]")          # "double" / «guillemets»
_DASH_LINE = re.compile(r"(?m)^\s*[—–\-]\s+\S")               # — dash-led dialogue lines
_PUNCT = re.compile(r"[.,;:!?…—–\"'“”«»()\-]")
_TERMINAL = re.compile(r"[.!?…][\"'”’»\)\]]*$")


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _dialogue_chars(text: str) -> int:
    total = sum(len(m.group(0)) for m in _DQ.finditer(text))
    for line in text.split("\n"):
        if _DASH_LINE.match(line):
            total += len(line)
    return total


def _repeated_ngram_score(tokens: list[str], n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    uniq = len(set(grams))
    return _safe_div(len(grams) - uniq, len(grams))   # 0 = no repeats, ->1 = very repetitive


def _repeated_sentence_score(sents: list[str]) -> float:
    if not sents:
        return 0.0
    norm = [re.sub(r"\s+", " ", s.strip().lower()) for s in sents]
    return _safe_div(len(norm) - len(set(norm)), len(norm))


def _malformed_paragraphs(paras: list[str]) -> int:
    """Count paragraphs that look broken: very long with no sentence punctuation."""
    bad = 0
    for p in paras:
        if len(p) > 1500 and not re.search(r"[.!?…]", p):
            bad += 1
    return bad


def compute_metrics(text: str, matcher: Pattern, cfg) -> dict:
    text = common.clean(text)
    chars = len(text)
    toks_est = common.estimate_tokens(text, cfg.chars_per_token)

    paras = common.paragraphs(text)
    sents = common.sentences(text)
    wlist = common.words(text)
    wcount = len(wlist)

    # keyword ------------------------------------------------------
    kw_hits = distinct_hits(text, matcher)
    kw_density = keyword_density(text, matcher)
    kw_ratio = keyword_ratio(text, matcher)

    # structure ----------------------------------------------------
    avg_sent_len = _safe_div(wcount, len(sents))                 # words / sentence
    avg_para_len = _safe_div(wcount, len(paras))                 # words / paragraph
    dlg = _dialogue_chars(text)
    dialogue_ratio = min(1.0, _safe_div(dlg, chars))
    narration_ratio = max(0.0, 1.0 - dialogue_ratio)
    punct_density = _safe_div(len(_PUNCT.findall(text)), wcount)

    # vocabulary ---------------------------------------------------
    unique_words = len(set(wlist))
    lexical_diversity = _safe_div(unique_words, wcount)          # type-token ratio
    repeated_phrase = _repeated_ngram_score(wlist, 4)
    repeated_sentence = _repeated_sentence_score(sents)

    # readability --------------------------------------------------
    avg_word_len = _safe_div(sum(len(w) for w in wlist), wcount)
    terminated = sum(1 for s in sents if _TERMINAL.search(s.strip()))
    punct_consistency = _safe_div(terminated, len(sents))
    double_space = text.count("  ")
    whitespace_quality = max(0.0, 1.0 - _safe_div(double_space, max(1, chars // 80)))
    malformed = _malformed_paragraphs(paras)

    # training -----------------------------------------------------
    over = max(0, toks_est - cfg.max_tokens)
    truncation_risk = min(1.0, _safe_div(over, max(1, cfg.max_tokens)))
    context_util = min(1.0, _safe_div(toks_est, cfg.target_tokens))
    recommended_splits = max(1, -(-toks_est // max(1, cfg.target_tokens)))  # ceil

    return {
        # basic
        "char_count": chars,
        "token_est": toks_est,
        "paragraph_count": len(paras),
        "sentence_count": len(sents),
        "word_count": wcount,
        # keyword
        "keyword_hits": kw_hits,
        "keyword_density": round(kw_density, 5),
        "keyword_ratio": round(kw_ratio, 4),
        # structure
        "avg_sentence_length": round(avg_sent_len, 2),
        "avg_paragraph_length": round(avg_para_len, 2),
        "dialogue_ratio": round(dialogue_ratio, 4),
        "narration_ratio": round(narration_ratio, 4),
        "punctuation_density": round(punct_density, 4),
        # vocabulary
        "unique_word_count": unique_words,
        "lexical_diversity": round(lexical_diversity, 4),
        "repeated_phrase_score": round(repeated_phrase, 4),
        "repeated_sentence_score": round(repeated_sentence, 4),
        # readability
        "avg_word_length": round(avg_word_len, 3),
        "punctuation_consistency": round(punct_consistency, 4),
        "whitespace_quality": round(whitespace_quality, 4),
        "malformed_paragraphs": malformed,
        # training
        "truncation_risk": round(truncation_risk, 4),
        "context_utilization": round(context_util, 4),
        "recommended_split_count": recommended_splits,
    }

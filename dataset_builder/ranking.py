#!/usr/bin/env python3
"""
ranking.py — convert a metric profile into one quality score in [0, 1].

Design principle: rank, don't threshold. Every sub-score lives in [0, 1];
the final score is a configurable weighted sum (weights live in config.py).
"""
from __future__ import annotations


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def length_score(token_est: int, target: int, lo: int, hi: int) -> float:
    """Triangular peak at `target`, decaying toward `lo` and `hi`.

    Scenes longer than `hi` are not punished to zero — they get split later
    into target-sized chunks — so we floor the penalty at 0.3 above `hi`.
    """
    if token_est <= 0:
        return 0.0
    if token_est <= target:
        if target <= lo:
            return 1.0
        return _clamp((token_est - lo) / (target - lo))
    # above target
    if token_est <= hi:
        if hi <= target:
            return 1.0
        return _clamp((hi - token_est) / (hi - target))
    return 0.3  # over max: still splittable into good chunks


def _dialogue_score(ratio: float, target: float) -> float:
    """Closeness to the desired dialogue/narration balance."""
    if target <= 0:
        return 1.0
    return _clamp(1.0 - abs(ratio - target) / max(target, 1.0 - target))


def _paragraph_score(metrics: dict, target_para_tokens: int) -> float:
    paras = metrics["paragraph_count"]
    if paras <= 1:
        return 0.3  # wall-of-text: weak structure
    avg_words = metrics["avg_paragraph_length"]
    avg_tokens = avg_words * 1.3  # rough words->tokens
    closeness = _clamp(1.0 - abs(avg_tokens - target_para_tokens) / max(1, target_para_tokens))
    malformed_pen = 0.5 if metrics["malformed_paragraphs"] else 1.0
    return _clamp(closeness * malformed_pen + 0.1)


def _punctuation_score(metrics: dict) -> float:
    return _clamp(0.5 * metrics["punctuation_consistency"] + 0.5 * metrics["whitespace_quality"])


def score_scene(metrics: dict, cfg) -> dict:
    """Return {'score': float, 'components': {...}} for transparency."""
    w = cfg.weights

    kw = _clamp(metrics["keyword_hits"] / max(1, cfg.target_keyword_hits))
    length = length_score(metrics["token_est"], cfg.target_tokens, cfg.min_tokens, cfg.max_tokens)
    lexical = _clamp(metrics["lexical_diversity"])
    dialogue = _dialogue_score(metrics["dialogue_ratio"], cfg.target_dialogue_ratio)
    paragraph = _paragraph_score(metrics, cfg.target_paragraph_tokens)
    punctuation = _punctuation_score(metrics)
    repetition = 1.0 - _clamp(
        0.6 * metrics["repeated_phrase_score"] + 0.4 * metrics["repeated_sentence_score"]
    )

    components = {
        "keyword": kw,
        "length": length,
        "lexical": lexical,
        "dialogue": dialogue,
        "paragraph": paragraph,
        "punctuation": punctuation,
        "repetition": repetition,
    }

    total_w = sum(w.values()) or 1.0
    score = sum(w.get(k, 0.0) * v for k, v in components.items()) / total_w
    return {"score": round(score, 5), "components": {k: round(v, 4) for k, v in components.items()}}

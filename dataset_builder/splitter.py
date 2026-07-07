#!/usr/bin/env python3
"""
splitter.py — split over-long scenes while preserving semantic structure.

Boundary priority (highest first):
    chapter -> section -> blank line -> paragraph -> dialogue block
            -> sentence -> comma -> hard cut

The splitter always attempts the highest possible boundary, and only divides a
sentence (or hard-cuts) as a last resort. Resulting atoms are then greedily
packed toward `target_tokens`, never exceeding `max_tokens`.
"""
from __future__ import annotations

import re
from typing import Callable, List

import common

# Boundary detectors, in priority order. Each returns a list of pieces.
_CHAPTER = re.compile(r"(?im)(?=^\s*(?:#{1,6}\s+|chapter\b|глава\b|глава\s|part\b|часть\b))")
_SECTION = re.compile(r"(?m)^\s*(?:\*\s*\*\s*\*|-{3,}|—{3,}|\*{3,}|#{1,6}\s*$)\s*$")
_BLANK = re.compile(r"\n{2,}")
_PARA = re.compile(r"\n")
_DIALOGUE = re.compile(r"(?m)(?=^\s*[—–\-\"“«])")
_COMMA = re.compile(r"(?<=,)\s+")


def _split_keep(text: str, pattern: re.Pattern) -> List[str]:
    parts = [p for p in pattern.split(text) if p and p.strip()]
    return parts if len(parts) > 1 else [text]


def _explode(piece: str, max_tokens: int, cpt: float) -> List[str]:
    """Recursively break `piece` until every atom fits in max_tokens."""
    if common.estimate_tokens(piece, cpt) <= max_tokens:
        return [piece]

    # Cascade through boundary levels.
    for pattern in (_CHAPTER, _SECTION, _BLANK, _PARA, _DIALOGUE):
        parts = _split_keep(piece, pattern)
        if len(parts) > 1:
            out: List[str] = []
            for p in parts:
                out.extend(_explode(p, max_tokens, cpt))
            return out

    # Sentences.
    sents = common.sentences(piece)
    if len(sents) > 1:
        out = []
        for s in sents:
            out.extend(_explode(s, max_tokens, cpt))
        return out

    # Commas.
    parts = _split_keep(piece, _COMMA)
    if len(parts) > 1:
        out = []
        for p in parts:
            out.extend(_explode(p, max_tokens, cpt))
        return out

    # Hard cut (last resort).
    max_chars = int(max_tokens * cpt)
    return [piece[i:i + max_chars] for i in range(0, len(piece), max_chars)]


def smart_split(text: str, cfg) -> List[str]:
    """Return one-or-more chunks, each <= max_tokens, clustered near target."""
    text = common.clean(text)
    if not cfg.split_enabled or common.estimate_tokens(text, cfg.chars_per_token) <= cfg.max_tokens:
        return [text]

    atoms = _explode(text, cfg.max_tokens, cfg.chars_per_token)

    chunks: List[str] = []
    cur = ""
    cur_tok = 0
    for a in atoms:
        a_tok = common.estimate_tokens(a, cfg.chars_per_token)
        if cur and (cur_tok + a_tok > cfg.max_tokens or cur_tok >= cfg.target_tokens):
            chunks.append(cur.strip())
            cur, cur_tok = "", 0
        cur = (cur + "\n\n" + a) if cur else a
        cur_tok += a_tok
    if cur.strip():
        chunks.append(cur.strip())

    return [c for c in chunks if c.strip()]

#!/usr/bin/env python3
"""
common.py — shared helpers for the dataset pipeline.

Intentionally dependency-free (stdlib only) so every stage runs out of the box.
Holds: text cleaning, paragraph/sentence/word splitting, token estimation,
DB scene iteration, and a small MinHash implementation used for near-duplicate
detection and diversity selection.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

# ----------------------------------------------------------------------
# Text normalisation / segmentation
# ----------------------------------------------------------------------

_WS = re.compile(r"[ \t]+")
_NL3 = re.compile(r"\n{3,}")
_PARA = re.compile(r"\n{2,}")
_WORD = re.compile(r"\w+", re.UNICODE)

# A sentence terminator: . ! ? … possibly followed by closing quotes/brackets,
# then whitespace. We capture the END index so callers can slice without
# losing any characters.
_SENT_END = re.compile(r"[.!?…]+[\"'”’»\)\]]*(?=\s)")


def clean(text: str) -> str:
    """Normalise newlines and collapse runaway whitespace (non-destructive)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _NL3.sub("\n\n", text)
    return text.strip()


def paragraphs(text: str) -> List[str]:
    return [p.strip() for p in _PARA.split(text) if p.strip()]


def sentences(text: str) -> List[str]:
    """Split into sentences while preserving all characters."""
    out: List[str] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        start = end
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Cheap token estimate (chars / N). Pluggable target for real tokenizers."""
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Database access
# ----------------------------------------------------------------------

@dataclass
class Scene:
    scene_id: str
    text: str


def iter_scenes(
    db: str,
    table: str = "segments",
    text_col: str = "text",
    scene_col: str = "scene_id",
    pos_col: str = "pos",
    mode: str = "scene",
) -> Iterator[Scene]:
    """Yield Scene(scene_id, text) from the SQLite database.

    mode='row'   -> each row is already a full scene (id = rowid)
    mode='scene' -> a scene = many segments concatenated by pos within scene_id
    """
    con = sqlite3.connect(db)
    cur = con.cursor()
    try:
        if mode == "row":
            q = f"SELECT rowid, {text_col} FROM {table} WHERE {text_col} IS NOT NULL"
            for rid, txt in cur.execute(q):
                if txt:
                    yield Scene(str(rid), txt)
        else:
            q = (
                f"SELECT {scene_col}, {text_col} FROM {table} "
                f"WHERE {text_col} IS NOT NULL ORDER BY {scene_col}, {pos_col}"
            )
            current_id = None
            buf: List[str] = []
            for sid, txt in cur.execute(q):
                if sid != current_id and buf:
                    yield Scene(str(current_id), "\n\n".join(buf))
                    buf = []
                current_id = sid
                if txt:
                    buf.append(txt)
            if buf:
                yield Scene(str(current_id), "\n\n".join(buf))
    finally:
        con.close()


def count_scenes(
    db: str,
    table: str = "segments",
    text_col: str = "text",
    scene_col: str = "scene_id",
    mode: str = "scene",
) -> int:
    """Cheap total count so progress bars can show a percentage / ETA."""
    con = sqlite3.connect(db)
    cur = con.cursor()
    try:
        if mode == "row":
            q = f"SELECT COUNT(*) FROM {table} WHERE {text_col} IS NOT NULL"
        else:
            q = f"SELECT COUNT(DISTINCT {scene_col}) FROM {table} WHERE {text_col} IS NOT NULL"
        return cur.execute(q).fetchone()[0]
    finally:
        con.close()


# ----------------------------------------------------------------------
# Progress reporting (stderr, stdlib only)
# ----------------------------------------------------------------------

class Progress:
    """Lightweight progress line printed to stderr (keeps stdout/JSONL clean).

    Updates at most every `interval` seconds. Shows count/total, %, rate, ETA
    when a total is known; otherwise just a running count and rate.
    """

    def __init__(self, total: Optional[int] = None, label: str = "",
                 interval: float = 0.2, stream=sys.stderr):
        self.total = total
        self.label = label
        self.interval = interval
        self.stream = stream
        self.n = 0
        self._start = time.time()
        self._last = 0.0

    def update(self, k: int = 1) -> None:
        self.n += k
        now = time.time()
        final = self.total is not None and self.n >= self.total
        if not final and (now - self._last) < self.interval:
            return
        self._last = now
        elapsed = max(1e-9, now - self._start)
        rate = self.n / elapsed
        if self.total:
            pct = 100.0 * self.n / self.total
            eta = (self.total - self.n) / rate if rate > 0 else 0
            msg = (f"\r{self.label} {self.n:,}/{self.total:,} "
                   f"({pct:4.1f}%) {rate:,.0f}/s ETA {eta:4.0f}s   ")
        else:
            msg = f"\r{self.label} {self.n:,} ({rate:,.0f}/s)   "
        self.stream.write(msg)
        self.stream.flush()

    def done(self) -> None:
        if self.n:
            self.stream.write("\n")
            self.stream.flush()


# ----------------------------------------------------------------------
# MinHash (for near-dup detection + diversity selection)
# ----------------------------------------------------------------------

_MERSENNE = (1 << 61) - 1  # large prime modulus


def shingles(text: str, k: int = 5) -> List[int]:
    """k-word shingles hashed to ints. Returns [] for very short texts."""
    toks = words(text)
    if len(toks) < k:
        return [hash_str(" ".join(toks))] if toks else []
    out = []
    for i in range(len(toks) - k + 1):
        out.append(hash_str(" ".join(toks[i:i + k])))
    return out


def hash_str(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


class MinHasher:
    """Deterministic MinHash with `num_perm` linear hash permutations."""

    def __init__(self, num_perm: int = 64, seed: int = 1):
        import random
        rnd = random.Random(seed)
        self.num_perm = num_perm
        self.a = [rnd.randrange(1, _MERSENNE) for _ in range(num_perm)]
        self.b = [rnd.randrange(0, _MERSENNE) for _ in range(num_perm)]

    def signature(self, text: str, k: int = 5) -> Sequence[int]:
        sh = shingles(text, k)
        if not sh:
            return tuple([0] * self.num_perm)
        sig = []
        for a, b in zip(self.a, self.b):
            m = min(((a * h + b) % _MERSENNE) for h in sh)
            sig.append(m)
        return tuple(sig)

    @staticmethod
    def jaccard(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
        if not sig_a or not sig_b:
            return 0.0
        same = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
        return same / len(sig_a)

    def bands(self, sig: Sequence[int], rows: int = 4) -> List[int]:
        """LSH band hashes — chunks sharing a band hash are candidate dups."""
        out = []
        for i in range(0, len(sig), rows):
            band = tuple(sig[i:i + rows])
            out.append(hash_str(repr(band)))
        return out
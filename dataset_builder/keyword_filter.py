#!/usr/bin/env python3
"""
keyword_filter.py

Generic keyword matching utilities shared by the dataset preparation tools.

The module is intentionally domain-agnostic. It simply detects configurable
keywords or phrases inside text and reports how many DISTINCT keywords were
matched.

Keywords are stored in a plain text file (default: keywords.txt located next
to this script).

One keyword or phrase per line.

Example:

dragon
castle
magic
ancient ruins

Blank lines and lines beginning with '#' are ignored.

Matching is:

- case insensitive
- whole word
- supports multi-word phrases

Typical usage:

    from keyword_filter import (
        load_keywords,
        compile_matcher,
        distinct_hits,
        matched_keywords,
    )

    matcher = compile_matcher(load_keywords())

    hits = distinct_hits(text, matcher)
    words = matched_keywords(text, matcher)
"""

from __future__ import annotations

import os
import re
from typing import List, Pattern, Set


# ----------------------------------------------------------------------
# Default keywords
#
# Only used if keywords.txt is missing.
# Replace with your own file for real projects.
# ----------------------------------------------------------------------

DEFAULT_KEYWORDS = [
    "example",
    "sample",
    "keyword",
]


# ----------------------------------------------------------------------
# Keyword loading
# ----------------------------------------------------------------------

def load_keywords(path: str | None = None) -> List[str]:
    """
    Load keywords from a text file.

    Parameters
    ----------
    path
        Optional path to a keyword file.

    Returns
    -------
    list[str]
        Lowercase keyword list.
    """

    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "keywords.txt",
        )

    if os.path.exists(path):

        keywords = []

        with open(path, encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                keywords.append(line.lower())

        if keywords:
            return keywords

    return [k.lower() for k in DEFAULT_KEYWORDS]


# ----------------------------------------------------------------------
# Matcher compilation
# ----------------------------------------------------------------------

def compile_matcher(keywords: List[str]) -> Pattern:
    """
    Compile a regex matcher.

    Whole-word matching.

    Multi-word phrases are supported.
    """

    if not keywords:
        raise ValueError("Keyword list is empty.")

    escaped = [re.escape(k) for k in keywords]

    pattern = (
        r"(?<!\w)"
        r"(?:"
        + "|".join(escaped)
        + r")"
        r"(?!\w)"
    )

    return re.compile(pattern, re.IGNORECASE)


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def matched_keywords(text: str, matcher: Pattern) -> Set[str]:
    """
    Return the set of matched keywords.
    """

    return {
        m.group(0).lower()
        for m in matcher.finditer(text)
    }


def distinct_hits(text: str, matcher: Pattern) -> int:
    """
    Number of distinct matched keywords.
    """

    return len(matched_keywords(text, matcher))


def keyword_density(text: str, matcher: Pattern) -> float:
    """
    Distinct keyword count divided by word count.

    Returns
    -------
    float
    """

    words = re.findall(r"\w+", text)

    if not words:
        return 0.0

    return distinct_hits(text, matcher) / len(words)


def keyword_ratio(text: str, matcher: Pattern) -> float:
    """
    Ratio of matched keywords to total configured keywords.

    Useful for ranking.
    """

    matches = matched_keywords(text, matcher)

    pattern = matcher.pattern

    total_keywords = pattern.count("|") + 1

    if total_keywords == 0:
        return 0.0

    return len(matches) / total_keywords


# ----------------------------------------------------------------------
# Convenience helper
# ----------------------------------------------------------------------

def analyze_keywords(text: str, matcher: Pattern) -> dict:
    """
    Return keyword statistics for one text.

    Example output:

    {
        "hits": 12,
        "density": 0.021,
        "ratio": 0.32,
        "matched": [
            "dragon",
            "magic",
            ...
        ]
    }
    """

    matched = sorted(matched_keywords(text, matcher))

    return {
        "hits": len(matched),
        "density": keyword_density(text, matcher),
        "ratio": keyword_ratio(text, matcher),
        "matched": matched,
    }
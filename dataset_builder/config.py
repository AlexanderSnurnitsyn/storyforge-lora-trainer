#!/usr/bin/env python3
"""
config.py — all adjustable parameters in one place.

Load defaults with Config.load() or override from a JSON file:

    cfg = Config.load("config.json")

The training pipeline should never require editing source code — only the
config file and keywords.txt.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # ---- database ----
    table: str = "segments"
    text_col: str = "text"
    scene_col: str = "scene_id"
    pos_col: str = "pos"
    mode: str = "scene"            # "row" | "scene"

    # ---- keywords ----
    keywords_file: str | None = None      # None -> keywords.txt next to scripts
    min_keyword_hits: int = 2
    keyword_filter_enabled: bool = True

    # ---- length targets (estimated tokens) ----
    target_tokens: int = 1400
    min_tokens: int = 200
    max_tokens: int = 1800
    chars_per_token: float = 4.0

    # ---- ranking ----
    weights: dict = field(default_factory=lambda: {
        "keyword": 0.30,
        "length": 0.25,
        "lexical": 0.15,
        "dialogue": 0.10,
        "paragraph": 0.10,
        "punctuation": 0.05,
        "repetition": 0.05,   # applied as (1 - repetition_score)
    })
    target_keyword_hits: int = 4          # hits >= this -> full keyword sub-score
    target_dialogue_ratio: float = 0.35   # ideal share of dialogue
    target_paragraph_tokens: int = 60     # ideal avg paragraph size (tokens)

    # ---- splitting ----
    split_enabled: bool = True

    # ---- dedup / diversity ----
    dedup_exact: bool = True
    near_dup_threshold: float = 0.85      # MinHash jaccard above -> duplicate
    minhash_perm: int = 64
    minhash_k: int = 5                    # shingle size (words)
    diversity_enabled: bool = True
    diversity_lambda: float = 0.5         # MMR: score vs. diversity tradeoff

    # ---- selection / output ----
    max_examples: int = 0                 # 0 = keep all (after dedup)
    val_frac: float = 0.02
    seed: int = 42
    out_prefix: str = "corpus"
    analysis_csv: str = "dataset_analysis.csv"
    chunks_csv: str = "corpus_chunks.csv"

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            base = asdict(cfg)
            for k, v in data.items():
                if k == "weights" and isinstance(v, dict):
                    base["weights"].update(v)
                elif k in base:
                    base[k] = v
                else:
                    raise KeyError(f"Unknown config key: {k}")
            cfg = cls(**base)
        return cfg

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

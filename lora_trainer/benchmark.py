"""
benchmark.py
============
Сравнение базовой модели и LoRA на наборе промптов из prompts/benchmark.txt.

CLI:
    python benchmark.py

Для каждого промпта генерирует ответ базы и LoRA, сохраняет всё в
output/benchmark/benchmark_<timestamp>.md.
"""

from __future__ import annotations

import gc
from datetime import datetime
from pathlib import Path

import unsloth  # noqa: F401  (первым)

from config import load_config, apply_run_paths
from inference import generate, load_pipeline


def load_prompts(path: Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Нет файла промптов: {path}")
    blocks = [b.strip() for b in path.read_text(encoding="utf-8").split("\n\n")]
    prompts = [b for b in blocks if b and not b.startswith("#")]
    if not prompts:
        raise ValueError(f"{path.name} не содержит промптов.")
    return prompts


def _free():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_benchmark(cfg) -> Path:
    prompts = load_prompts(cfg.paths.prompts)
    print(f"Промптов: {len(prompts)}")

    print("Генерация базовой моделью...")
    bm, bt = load_pipeline(cfg, with_lora=False)
    base = [generate(bm, bt, p, cfg) for p in prompts]
    del bm
    _free()

    print("Генерация LoRA...")
    lm, lt = load_pipeline(cfg, with_lora=True)
    lora = [generate(lm, lt, p, cfg) for p in prompts]
    del lm
    _free()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.paths.benchmark_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"benchmark_{ts}.md"

    lines = [f"# StoryForge benchmark — {ts}\n",
             f"- Базовая модель: `{cfg.model.base_model}`",
             f"- LoRA: `{cfg.export.lora_dir}`",
             f"- Генерация: temp={cfg.gen.temperature}, top_p={cfg.gen.top_p}, "
             f"max_new_tokens={cfg.gen.max_new_tokens}\n"]
    for i, (p, ba, la) in enumerate(zip(prompts, base, lora), 1):
        lines += [f"\n## Промпт {i}\n", f"> {p}\n",
                  "### Базовая модель\n", ba + "\n",
                  "### LoRA\n", la + "\n", "\n---"]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт сохранён: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="StoryForge benchmark")
    ap.add_argument("--run", default=None, help="имя прогона из runs/ (по умолчанию самый свежий)")
    ap.add_argument("--config", default=None, help="путь к конфигу")
    _a = ap.parse_args()
    _cfg = load_config(_a.config)
    apply_run_paths(_cfg, _a.run, create=False)
    run_benchmark(_cfg)

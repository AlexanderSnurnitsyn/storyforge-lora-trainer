"""
compare_checkpoints.py
======================
Прогоняет ОДИН промпт через несколько стадий обучения, чтобы найти момент
до переобучения (когда стиль уже есть, но имена из корпуса ещё не «протекают»).

Сравнивает: базовую модель + все output/checkpoints/checkpoint-* + финальную LoRA.

Запуск:
    python compare_checkpoints.py --prompt "Опиши осенний город..."
    python compare_checkpoints.py --prompt "..." --steps 25 75 125   # только эти шаги
    python compare_checkpoints.py --prompt "..." --no-base           # без базовой

Результат печатается и сохраняется в output/benchmark/checkpoints_<timestamp>.md
"""

from __future__ import annotations

# Unsloth первым.
import unsloth  # noqa: F401

import argparse
import gc
import re
from datetime import datetime
from pathlib import Path

from config import load_config, apply_run_paths
from inference import generate, load_pipeline


def _free():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_checkpoints(cfg, steps=None) -> list[tuple[str, Path]]:
    ckpt_dir = Path(cfg.train.output_dir)
    found = []
    if ckpt_dir.exists():
        for p in ckpt_dir.glob("checkpoint-*"):
            m = re.search(r"checkpoint-(\d+)$", p.name)
            if m and (p / "adapter_model.safetensors").exists():
                found.append((int(m.group(1)), p))
    found.sort(key=lambda x: x[0])
    if steps:
        wanted = set(steps)
        found = [(s, p) for s, p in found if s in wanted]
    return [(f"checkpoint-{s}", p) for s, p in found]


def run(cfg, prompt: str, steps=None, include_base=True) -> Path:
    # Список того, что сравниваем: (метка, путь_к_адаптеру | None для базы)
    stages: list[tuple[str, Path | None]] = []
    if include_base:
        stages.append(("БАЗА", None))
    stages.extend((label, path) for label, path in find_checkpoints(cfg, steps))
    final = Path(cfg.export.lora_dir)
    if (final / "adapter_model.safetensors").exists():
        stages.append(("LoRA (финал)", final))

    if len(stages) <= (1 if include_base else 0):
        raise SystemExit("Не найдено чекпоинтов в output/checkpoints и финальной LoRA. "
                         "Сначала обучи модель (train.py).")

    print(f"Промпт: {prompt}\nСтадий для сравнения: {len(stages)}\n")
    results: list[tuple[str, str]] = []
    for label, path in stages:
        print(f"--- генерирую: {label} ---")
        model, tok = load_pipeline(cfg, with_lora=path is not None, lora_path=path)
        text = generate(model, tok, prompt, cfg)
        results.append((label, text))
        print(text + "\n")
        del model
        _free()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.paths.benchmark_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"checkpoints_{ts}.md"
    lines = [f"# Сравнение чекпоинтов — {ts}\n",
             f"**Промпт:** {prompt}\n",
             f"Генерация: temp={cfg.gen.temperature}, top_p={cfg.gen.top_p}, "
             f"max_new_tokens={cfg.gen.max_new_tokens}\n"]
    for label, text in results:
        lines += [f"\n## {label}\n", text + "\n", "\n---"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт сохранён: {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Сравнение стадий обучения на одном промпте")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--steps", nargs="*", type=int, default=None,
                    help="только эти шаги, напр. --steps 25 75 125")
    ap.add_argument("--no-base", action="store_true", help="не включать базовую модель")
    ap.add_argument("--run", default=None, help="имя прогона из runs/ (по умолчанию самый свежий)")
    ap.add_argument("--config", default=None, help="путь к конфигу")
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_run_paths(cfg, args.run, create=False)
    run(cfg, args.prompt, steps=args.steps, include_base=not args.no_base)


if __name__ == "__main__":
    main()

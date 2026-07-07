"""
runs.py
=======
Просмотр обученных прогонов (LoRA-версий) в папке runs/.

    python runs.py                 # таблица всех прогонов
    python runs.py --show <name>   # подробности одного прогона (run.json)

Каждый прогон — это runs/<name>/ с адаптером, чекпоинтами, логами и
метаданными run.json. Скрипты обучения/инференса/экспорта адресуют прогон
флагом --run <name>.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import load_config


def _load_meta(run_dir: Path) -> dict:
    meta_file = run_dir / "run.json"
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def list_runs(cfg) -> None:
    runs_dir = Path(cfg.run.runs_dir)
    if not runs_dir.is_absolute():
        from config import PROJECT_ROOT
        runs_dir = PROJECT_ROOT / runs_dir
    if not runs_dir.exists() or not any(runs_dir.iterdir()):
        print(f"Прогонов пока нет ({runs_dir}). Обучи модель: python train.py")
        return

    rows = []
    for d in sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        meta = _load_meta(d)
        has_lora = (d / "lora" / "adapter_model.safetensors").exists()
        n_ckpt = len(list((d / "checkpoints").glob("checkpoint-*"))) if (d / "checkpoints").exists() else 0
        lora = meta.get("lora", {})
        tcfg = meta.get("train_config", {})
        corpus = (meta.get("corpus") or {}).get("train", {})
        metrics = meta.get("metrics", {})
        rows.append({
            "name": d.name,
            "created": meta.get("created", "")[:16].replace("T", " "),
            "r": lora.get("r", "—"),
            "epochs": tcfg.get("epochs", "—"),
            "examples": corpus.get("examples", "—"),
            "loss": round(metrics["train_loss"], 3) if "train_loss" in metrics else "—",
            "lora": "да" if has_lora else "нет",
            "ckpt": n_ckpt,
        })

    name_w = max(len(r["name"]) for r in rows + [{"name": "ПРОГОН"}])
    head = (f"{'ПРОГОН':<{name_w}}  {'ДАТА':<16}  {'r':>3}  {'эпох':>5}  "
            f"{'сцен':>5}  {'loss':>6}  {'LoRA':>4}  {'чек':>3}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['name']:<{name_w}}  {r['created']:<16}  {str(r['r']):>3}  "
              f"{str(r['epochs']):>5}  {str(r['examples']):>5}  {str(r['loss']):>6}  "
              f"{r['lora']:>4}  {r['ckpt']:>3}")
    print(f"\nВсего прогонов: {len(rows)}")
    print("Использовать:  python inference.py --run <ПРОГОН> --prompt \"...\" --compare")


def show_run(cfg, name: str) -> None:
    runs_dir = Path(cfg.run.runs_dir)
    if not runs_dir.is_absolute():
        from config import PROJECT_ROOT
        runs_dir = PROJECT_ROOT / runs_dir
    run_dir = runs_dir / name
    if not run_dir.exists():
        raise SystemExit(f"Прогон не найден: {run_dir}")
    meta = _load_meta(run_dir)
    if not meta:
        print(f"У прогона {name} нет run.json (обучался старой версией?).")
    else:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nПапка: {run_dir}")
    lora = run_dir / "lora" / "adapter_model.safetensors"
    print("Адаптер:", "есть" if lora.exists() else "НЕТ")


def main() -> None:
    ap = argparse.ArgumentParser(description="Список и просмотр прогонов LoRA")
    ap.add_argument("--show", default=None, metavar="NAME", help="подробности прогона")
    ap.add_argument("--config", default=None, help="путь к конфигу")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.show:
        show_run(cfg, args.show)
    else:
        list_runs(cfg)


if __name__ == "__main__":
    main()

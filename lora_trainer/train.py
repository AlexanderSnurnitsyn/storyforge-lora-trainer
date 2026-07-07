"""
train.py  (Unsloth)
===================
Запуск:  python train.py

Последовательность та же, что раньше: проверки окружения -> CUDA -> файлы/диск
-> модель (Unsloth) -> корпус -> обучение -> сохранение LoRA + статистики.
"""

from __future__ import annotations

# Unsloth ДОЛЖЕН импортироваться первым, до transformers/trl, иначе патчи
# не применятся и будет предупреждение/замедление.
import unsloth  # noqa: F401  (важен сам факт раннего импорта)

import json
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from config import load_config, apply_run_paths, config_to_dict


def _ver_tuple(v: str) -> tuple[int, ...]:
    parts = []
    for chunk in v.split(".")[:3]:
        num = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts)


def check_libraries(cfg) -> None:
    print("[1/8] Проверка версий библиотек...")
    problems = []
    for pkg, min_v in cfg.min_versions.items():
        try:
            cur = version(pkg)
        except PackageNotFoundError:
            problems.append(f"  ОТСУТСТВУЕТ: {pkg} (нужно >= {min_v})")
            continue
        if _ver_tuple(cur) < _ver_tuple(min_v):
            problems.append(f"  УСТАРЕЛА: {pkg}=={cur} (нужно >= {min_v})")
        else:
            print(f"  ok  {pkg}=={cur}")
    if problems:
        print("\n".join(problems))
        sys.exit("Останов: несовместимые/отсутствующие библиотеки. "
                 "См. README (раздел установки Unsloth для Windows).")


def check_cuda() -> None:
    print("[2/8] Проверка CUDA...")
    import torch
    if not torch.cuda.is_available():
        sys.exit("Останов: CUDA недоступна. Обучение требует GPU.")
    print(f"  CUDA ok, устройств: {torch.cuda.device_count()}, "
          f"torch.version.cuda={torch.version.cuda}")


def check_paths_and_disk(cfg) -> None:
    print("[3/8] Проверка файлов и диска...")
    if not Path(cfg.data.train_file).exists():
        sys.exit(f"Останов: нет train-корпуса: {cfg.data.train_file}")
    print(f"  train: {cfg.data.train_file}")
    print(f"  val:   {cfg.data.val_file}"
          if Path(cfg.data.val_file).exists()
          else f"  val:   отсутствует ({cfg.data.val_file})")
    for d in (cfg.train.output_dir, cfg.export.lora_dir, cfg.paths.logs):
        Path(d).mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(cfg.train.output_dir).free / (1024 ** 3)
    print(f"  Свободно на диске: {free_gb:.1f} ГБ")
    if free_gb < cfg.min_free_disk_gb:
        sys.exit(f"Останов: мало места ({free_gb:.1f} < {cfg.min_free_disk_gb} ГБ).")


def main(config_path=None, run_name=None) -> None:
    cfg = load_config(config_path)
    name, run_root = apply_run_paths(cfg, run_name, create=True)
    print("=" * 64)
    print("StoryForge LoRA (Unsloth) — обучение")
    print(f"Прогон: {name}")
    print(f"Папка:  {run_root}")
    print("=" * 64)

    check_libraries(cfg)
    check_cuda()
    check_paths_and_disk(cfg)

    from transformers import set_seed
    set_seed(cfg.train.seed)

    print("[4/8] Конфигурация загружена.")
    import dataset as ds_mod
    import model as model_mod
    from trainer import build_trainer

    print("[5/8] Загрузка модели...")
    model, tokenizer, _dtype = model_mod.load_for_training(cfg)

    print("[6/8] Загрузка корпуса...")
    train_ds, val_ds, corpus_stats = ds_mod.prepare(cfg, tokenizer)
    collator = ds_mod.DataCollatorCausalLM(tokenizer)

    print("[7/8] Обучение...")
    trainer, resume_path = build_trainer(
        cfg, model, tokenizer, train_ds, val_ds, collator
    )
    train_result = trainer.train(resume_from_checkpoint=resume_path)

    print("[8/8] Сохранение результатов...")
    Path(cfg.export.lora_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.export.lora_dir))
    tokenizer.save_pretrained(str(cfg.export.lora_dir))
    print(f"  LoRA сохранена в {cfg.export.lora_dir}")

    metrics = dict(train_result.metrics)
    if val_ds is not None and cfg.train.eval_strategy != "no":
        try:
            metrics.update(trainer.evaluate())
        except Exception as exc:
            print(f"  Предупреждение: финальная валидация не выполнена: {exc}")

    import datetime as _dt
    final_stats = {
        "run_name": name,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "base_model": cfg.model.base_model,
        "engine": "unsloth",
        "lora": {"r": cfg.lora.r, "alpha": cfg.lora.alpha,
                 "dropout": cfg.lora.dropout,
                 "target_modules": cfg.lora.target_modules},
        "train_config": {"epochs": cfg.train.epochs, "lr": cfg.train.lr,
                         "effective_batch":
                             cfg.train.per_device_batch_size * cfg.train.grad_accum,
                         "max_seq_len": cfg.data.max_seq_len},
        "corpus": corpus_stats,
        "metrics": metrics,
        "resumed_from": resume_path,
        "lora_dir": str(cfg.export.lora_dir),
    }
    Path(cfg.paths.stats_file).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.paths.stats_file, "w", encoding="utf-8") as fh:
        json.dump(final_stats, fh, ensure_ascii=False, indent=2)

    try:
        import yaml
        snap = run_root / "config.snapshot.yaml"
        with open(snap, "w", encoding="utf-8") as fh:
            yaml.safe_dump(config_to_dict(cfg), fh, allow_unicode=True, sort_keys=False)
        print(f"  Снимок конфига:      {snap}")
    except Exception:
        pass
    print(f"  Итоговая статистика: {cfg.paths.stats_file}")
    print(f"  Лог обучения:        {Path(cfg.paths.logs) / 'train_log.jsonl'}")
    print("Готово.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="StoryForge LoRA (Unsloth) — обучение")
    ap.add_argument("--config", default=None,
                    help="путь к YAML/JSON конфигу (по умолчанию config.yaml)")
    ap.add_argument("--run", default=None,
                    help="имя прогона (по умолчанию из config.run.name или авто)")
    _a = ap.parse_args()
    main(_a.config, _a.run)

"""
trainer.py  (Unsloth) — совместим с transformers 4.x И 5.x
==========================================================
Только настройка обучения. Модель приходит готовой из model.py (Unsloth).

Совместимость с transformers 5.x: в v5 из TrainingArguments убрали ряд
параметров (group_by_length, save_safetensors, warmup_ratio -> warmup_steps
принимает float, evaluation_strategy -> eval_strategy, и др.). Чтобы не падать
на каждом, желаемые аргументы собираются в dict и ФИЛЬТРУЮТСЯ по реальной
сигнатуре TrainingArguments установленной версии — неподдерживаемые молча
отбрасываются (с предупреждением). Так код переживает и 4.x, и 5.x.

Ключевое отличие от transformers-варианта:
    gradient_checkpointing в TrainingArguments ВЫКЛЮЧЕН — им управляет Unsloth
    (use_gradient_checkpointing="unsloth" в model.py). Включать дважды нельзя.

Публичный API:
    find_last_checkpoint(output_dir)                              -> str | None
    build_trainer(cfg, model, tok, train_ds, val_ds, collator)   -> (Trainer, resume)
    MetricsCallback                                              -> логирование
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


def find_last_checkpoint(output_dir: Path) -> str | None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    try:
        return get_last_checkpoint(str(output_dir))
    except Exception:
        return None


class MetricsCallback(TrainerCallback):
    """loss / LR / VRAM / скорость / время эпох -> logs/train_log.jsonl."""

    def __init__(self, log_dir: Path):
        self.log_path = Path(log_dir) / "train_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        self._epoch_start = None
        self._train_start = None

    def _append(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _vram_gb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        dur = time.time() - self._epoch_start if self._epoch_start else 0.0
        self._append({"event": "epoch_end",
                      "epoch": round(state.epoch, 3) if state.epoch else None,
                      "step": state.global_step,
                      "epoch_seconds": round(dur, 1),
                      "vram_peak_gb": self._vram_gb()})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"event": "log", "step": state.global_step,
                  "epoch": round(state.epoch, 3) if state.epoch else None,
                  "vram_peak_gb": self._vram_gb()}
        for key in ("loss", "eval_loss", "learning_rate", "grad_norm",
                    "train_samples_per_second", "train_steps_per_second"):
            if key in logs:
                record[key] = logs[key]
        self._append(record)

    def on_train_end(self, args, state, control, **kwargs):
        total = time.time() - self._train_start if self._train_start else 0.0
        self._append({"event": "train_end",
                      "total_seconds": round(total, 1),
                      "total_steps": state.global_step,
                      "vram_peak_gb": self._vram_gb()})


def _supported_training_args(desired: dict) -> dict:
    """Оставляет только те ключи, которые принимает TrainingArguments текущей
    версии transformers. Про отброшенные печатает предупреждение."""
    try:
        valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    except (ValueError, TypeError):
        return desired
    kept, dropped = {}, []
    for k, v in desired.items():
        if k in valid:
            kept[k] = v
        else:
            dropped.append(k)
    if dropped:
        print("  [trainer] параметры не поддерживаются этой версией "
              f"transformers и отброшены: {', '.join(sorted(dropped))}")
    return kept


def build_trainer(cfg, model, tokenizer, train_ds, val_ds, collator):
    t = cfg.train
    do_eval = val_ds is not None and t.eval_strategy != "no"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Желаемые аргументы. warmup задаём и как ratio (4.x), и как steps-float (5.x);
    # фильтр ниже оставит тот, что реально существует в установленной версии.
    desired = dict(
        output_dir=str(t.output_dir),
        num_train_epochs=t.epochs,
        per_device_train_batch_size=t.per_device_batch_size,
        per_device_eval_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.grad_accum,
        learning_rate=t.lr,
        weight_decay=t.weight_decay,
        warmup_ratio=t.warmup_ratio,       # 4.x
        warmup_steps=t.warmup_ratio,       # 5.x (принимает float как долю)
        lr_scheduler_type=t.lr_scheduler,
        optim=t.optimizer,
        max_grad_norm=t.max_grad_norm,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        # Checkpointing делает Unsloth — здесь ВЫКЛ.
        gradient_checkpointing=False,
        group_by_length=t.group_by_length,     # уйдёт на 5.x (отфильтруется)
        logging_steps=t.logging_steps,
        eval_strategy=t.eval_strategy if do_eval else "no",
        eval_steps=t.eval_steps if do_eval else None,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        seed=t.seed,
        data_seed=t.seed,
        report_to=t.report_to,
        save_safetensors=True,                 # уйдёт на 5.x (safetensors и так дефолт)
        remove_unused_columns=False,
    )

    # warmup: на 5.x warmup_ratio нет; если оставить оба, warmup_steps как float
    # задаёт долю. На 4.x наоборот — warmup_steps=<доля<1> некорректен, там нужен
    # warmup_ratio. Разрулим по факту наличия ключей после фильтрации.
    valid = set()
    try:
        valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    except Exception:
        pass
    if "warmup_ratio" in valid:
        desired.pop("warmup_steps", None)      # 4.x: используем ratio
    else:
        desired.pop("warmup_ratio", None)      # 5.x: используем steps-as-float

    args = TrainingArguments(**_supported_training_args(desired))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if do_eval else None,
        data_collator=collator,
        callbacks=[MetricsCallback(cfg.paths.logs)],
    )

    resume_path = None
    if t.resume:
        resume_path = find_last_checkpoint(t.output_dir)
        print(f"  Возобновление с: {resume_path}" if resume_path
              else "  Чекпоинтов нет — обучение с нуля.")
    return trainer, resume_path
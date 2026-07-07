"""
model.py  (Unsloth)
===================
Только загрузка модели через Unsloth и подготовка под LoRA.
Никакой логики обучения / работы с корпусом.

ВАЖНО: `from unsloth import FastLanguageModel` должен выполняться РАНЬШЕ,
чем импортируется transformers (Unsloth патчит его при импорте). В train.py
unsloth импортируется первой строкой — этого достаточно.

Публичный API:
    detect_precision()              -> ("bf16"|"fp16"|"fp32", dtype)
    check_vram(min_gb)              -> float
    load_for_training(cfg)          -> (model, tokenizer, dtype)
"""

from __future__ import annotations

from unsloth import FastLanguageModel  # noqa: E402  (должен быть до transformers)
import torch  # noqa: E402


def detect_precision() -> tuple[str, "torch.dtype"]:
    if not torch.cuda.is_available():
        return "fp32", torch.float32
    if torch.cuda.is_bf16_supported():
        return "bf16", torch.bfloat16
    return "fp16", torch.float16


def check_vram(min_gb: float = 0.0) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна — обучение на CPU не поддерживается.")
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    print(f"  GPU: {props.name}, VRAM: {total_gb:.1f} ГБ")
    if min_gb and total_gb < min_gb:
        print(f"  ВНИМАНИЕ: VRAM ({total_gb:.1f} ГБ) меньше рекомендуемых "
              f"{min_gb:.1f} ГБ. При OOM снижай max_seq_len / batch / r.")
    return total_gb


def _print_trainable(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total else 0.0
    print(f"  Обучаемых параметров: {trainable:,} из {total:,} ({pct:.4f}%)")


def load_for_training(cfg):
    print("Загрузка модели (Unsloth)...")
    precision_name, dtype = detect_precision()
    print(f"  Точность вычислений: {precision_name}")
    check_vram(min_gb=8.0)

    # dtype=None -> Unsloth сам выберет bf16 на Ampere (3060).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.base_model,
        max_seq_length=cfg.data.max_seq_len,
        dtype=None,
        load_in_4bit=cfg.model.load_in_4bit,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        target_modules=cfg.lora.target_modules,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        # Встроенный экономный checkpointing Unsloth (НЕ включай его же в Trainer).
        use_gradient_checkpointing="unsloth",
        random_state=cfg.train.seed,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    _print_trainable(model)
    return model, tokenizer, dtype

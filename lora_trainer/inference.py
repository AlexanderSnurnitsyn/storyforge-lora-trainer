"""
inference.py  (Unsloth)
=======================
Проверка модели. Публичный API совпадает с прежним, поэтому benchmark.py
работает без изменений.

CLI:
    python inference.py --prompt "..."             # базовая модель
    python inference.py --prompt "..." --lora      # LoRA
    python inference.py --prompt "..." --compare   # обе рядом

Публичный API:
    load_pipeline(cfg, with_lora) -> (model, tokenizer)
    generate(model, tokenizer, prompt, cfg) -> str
"""

from __future__ import annotations

# Unsloth первым.
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import argparse

from config import load_config, apply_run_paths


def load_pipeline(cfg, with_lora: bool, lora_path=None):
    # lora_path: явный путь к адаптеру/чекпоинту (приоритетнее, чем with_lora)
    if lora_path:
        src = str(lora_path)
    elif with_lora:
        src = str(cfg.export.lora_dir)
    else:
        src = cfg.model.base_model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=src,
        max_seq_length=cfg.data.max_seq_len,
        dtype=None,
        load_in_4bit=cfg.model.load_in_4bit,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    FastLanguageModel.for_inference(model)   # 2x быстрее генерация
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _build_inputs(tokenizer, prompt: str, device):
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").to(device)


def generate(model, tokenizer, prompt: str, cfg) -> str:
    import torch
    inputs = _build_inputs(tokenizer, prompt, model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.gen.max_new_tokens,
            temperature=cfg.gen.temperature,
            top_p=cfg.gen.top_p,
            top_k=cfg.gen.top_k,
            repetition_penalty=cfg.gen.repetition_penalty,
            do_sample=cfg.gen.do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="StoryForge inference (Unsloth)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="база vs LoRA (или vs --lora-path)")
    parser.add_argument("--lora-path", default=None,
                        help="явный путь к адаптеру/чекпоинту (важнее --run)")
    parser.add_argument("--run", default=None,
                        help="имя прогона из runs/ (по умолчанию самый свежий)")
    parser.add_argument("--config", default=None, help="путь к конфигу")
    args = parser.parse_args()
    cfg = load_config(args.config)
    # если нужен адаптер и не задан явный путь — резолвим прогон
    if not args.lora_path and (args.lora or args.compare or args.run):
        apply_run_paths(cfg, args.run, create=False)

    if args.compare:
        print("\n=== БАЗОВАЯ МОДЕЛЬ ===")
        bm, bt = load_pipeline(cfg, with_lora=False)
        print(generate(bm, bt, args.prompt, cfg))
        import gc, torch
        del bm; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        label = f"LoRA ({args.lora_path})" if args.lora_path else "LoRA"
        print(f"\n=== {label} ===")
        lm, lt = load_pipeline(cfg, with_lora=True, lora_path=args.lora_path)
        print(generate(lm, lt, args.prompt, cfg))
    else:
        use_lora = args.lora or bool(args.lora_path)
        model, tok = load_pipeline(cfg, with_lora=use_lora, lora_path=args.lora_path)
        tag = f"LoRA ({args.lora_path})" if args.lora_path else ("LoRA" if use_lora else "БАЗОВАЯ МОДЕЛЬ")
        print(f"\n=== {tag} ===")
        print(generate(model, tok, args.prompt, cfg))


if __name__ == "__main__":
    main()

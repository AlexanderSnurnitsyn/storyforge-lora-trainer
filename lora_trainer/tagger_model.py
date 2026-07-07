#!/usr/bin/env python3
"""
tagger_model.py — load ANY HF instruct model for tagging via plain transformers
+ bitsandbytes 4-bit. Deliberately NOT Unsloth: Unsloth patches generation and
can silently drop `prefix_allowed_tokens_fn`, which breaks constrained JSON
decoding. The tagger is pure inference and needs no training optimisations, so a
clean transformers load is both more reliable (logits processors honored) and
model-agnostic (swap Llama / Qwen / Hermes / Dolphin freely via --model).

Heavy imports are lazy so this module stays importable without torch installed.
"""
from __future__ import annotations


def load_tagger(model_name, max_seq_len=8192, load_in_4bit=True, cache_dir=None):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir,
                                        trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = None
    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        dtype=compute,             # explicit bf16: keeps embeddings/lm_head off fp32
        device_map={"": 0},        # whole model on GPU 0 — NO silent CPU/RAM offload
        cache_dir=cache_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    name = getattr(model.config, "_name_or_path", model_name)
    dev = next(model.parameters()).device
    try:
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  tagger loaded: {name}  on {dev}  4bit={load_in_4bit}  "
              f"VRAM~{alloc:.1f}GB", flush=True)
    except Exception:
        print(f"  tagger loaded: {name}  on {dev}  4bit={load_in_4bit}")
    return model, tok

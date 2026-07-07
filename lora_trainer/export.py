"""
export.py  (Unsloth)
====================
Экспорт результатов через встроенные функции Unsloth — без ручного llama.cpp
в коде (Unsloth сам соберёт llama.cpp при GGUF-экспорте).

CLI:
    python export.py --merge      # слить LoRA в полную модель (16-bit) -> output/merged
    python export.py --gguf       # экспорт в GGUF (квант из config) -> output/gguf
    python export.py --merge --gguf

ПРИМЕЧАНИЕ по GGUF на Windows: Unsloth собирает llama.cpp на лету, для этого
нужен установленный компилятор (Visual Studio C++ + CMake). Если сборка падает —
проще сделать merge здесь, а GGUF-конвертацию выполнить отдельно готовым
llama.cpp, либо собрать GGUF на Linux/WSL/Colab.
"""

from __future__ import annotations

# Unsloth первым.
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import argparse
import sys
from pathlib import Path

from config import load_config, apply_run_paths


def _load_lora_full(cfg):
    """Грузит обученный адаптер поверх базы в 16-bit (для merge/gguf)."""
    lora_dir = Path(cfg.export.lora_dir)
    if not lora_dir.exists():
        sys.exit(f"Нет LoRA: {lora_dir}. Сначала выполни train.py.")
    # Unsloth прочитает base_model из adapter_config.json и наложит адаптер.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(lora_dir),
        max_seq_length=cfg.data.max_seq_len,
        dtype=None,
        load_in_4bit=True,            # 16-bit для корректного merge
        trust_remote_code=cfg.model.trust_remote_code,
    )
    return model, tokenizer


def merge(cfg) -> Path:
    print("Merge LoRA -> полная модель (16-bit)...")
    model, tokenizer = _load_lora_full(cfg)
    out = Path(cfg.export.merged_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(out), tokenizer, save_method="merged_4bit_forced")
    print(f"Готово: {out}")
    return out


def export_gguf(cfg) -> None:
    print(f"Экспорт в GGUF ({cfg.export.gguf_quant})...")
    model, tokenizer = _load_lora_full(cfg)
    out = Path(cfg.export.gguf_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained_gguf(
            str(out), tokenizer, quantization_method=cfg.export.gguf_quant
        )
    except Exception as exc:
        print(f"GGUF-экспорт не удался: {exc}\n"
              "Скорее всего не собрался llama.cpp (нужен Visual Studio C++ + CMake).\n"
              "Варианты: сделать merge здесь и сконвертировать готовым llama.cpp, "
              "либо собрать GGUF на WSL/Colab.")
        return
    print(f"Готово: {out}")
    _write_ollama_modelfile(cfg, out)


def _write_ollama_modelfile(cfg, gguf_dir: Path) -> None:
    """Готовит Modelfile, чтобы подключить GGUF в ollama одной командой."""
    ggufs = sorted(gguf_dir.glob("*.gguf"))
    if not ggufs:
        return
    modelfile = gguf_dir / "Modelfile"
    modelfile.write_text(
        f"FROM ./{ggufs[0].name}\n"
        f"PARAMETER temperature {cfg.gen.temperature}\n"
        f"PARAMETER top_p {cfg.gen.top_p}\n",
        encoding="utf-8",
    )
    print(f"Modelfile создан: {modelfile}\n"
          f"  Подключить в ollama:  ollama create my-storyforge -f \"{modelfile}\"")


def main() -> None:
    parser = argparse.ArgumentParser(description="StoryForge LoRA export (Unsloth)")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--gguf", action="store_true")
    parser.add_argument("--run", default=None, help="имя прогона из runs/ (по умолчанию самый свежий)")
    parser.add_argument("--config", default=None, help="путь к конфигу")
    args = parser.parse_args()
    if not (args.merge or args.gguf):
        parser.print_help()
        return
    cfg = load_config(args.config)
    apply_run_paths(cfg, args.run, create=False)
    if args.merge:
        merge(cfg)
    if args.gguf:
        export_gguf(cfg)


if __name__ == "__main__":
    main()

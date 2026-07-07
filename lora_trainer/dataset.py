"""
dataset.py  (предтокенизация — под обычный transformers.Trainer)
================================================================
Всё, что связано с корпусом, и ничего больше. БЕЗ trl/SFTTrainer.

Датасет отдаётся УЖЕ токенизированным: колонки input_ids / labels / length.
Паддинг батча и маска лейблов — в DataCollatorCausalLM.
Маскирование промпта для instruction/chat (учим только ответ) делается здесь
через labels = -100 на токенах промпта. Для text учим всю строку.

Публичный API:
    prepare(cfg, tokenizer) -> (train_ds, val_ds, stats)     # РОВНО 3 значения
    DataCollatorCausalLM
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from datasets import Dataset


# --------------------------------------------------------------------------- #
# Чтение и проверка JSONL
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл корпуса не найден: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}: строка {i} — некорректный JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path.name}: строка {i} — JSON не объект")
            rows.append(obj)
    if not rows:
        raise ValueError(f"{path.name}: корпус пуст")
    return rows


def detect_format(records: list[dict[str, Any]], text_field: str = "text") -> str:
    keys = set().union(*(r.keys() for r in records[:50]))
    if "messages" in keys:
        return "chat"
    if "instruction" in keys or {"prompt", "completion"} & keys:
        return "instruction"
    if text_field in keys:
        return "text"
    raise ValueError(
        "Не удалось определить формат корпуса. Ожидался один из: "
        f"{{'{text_field}'}} | {{'instruction','output'}} | {{'messages'}}. "
        f"Ключи: {sorted(keys)}. Укажи формат явно в config.data.fmt."
    )


def _extract(rec: dict[str, Any], fmt: str, cfg) -> tuple[str, str] | None:
    """Возвращает (prompt, target). Для text prompt='' (учим всё)."""
    if fmt == "text":
        body = str(rec.get(cfg.data.text_field, "") or "").strip()
        return ("", body) if body else None
    if fmt == "instruction":
        if "prompt" in rec or "completion" in rec:
            instr = str(rec.get("prompt", "") or "").strip()
            extra, out = "", str(rec.get("completion", "") or "").strip()
        else:
            instr = str(rec.get("instruction", "") or "").strip()
            extra = str(rec.get("input", "") or "").strip()
            out = str(rec.get("output", "") or "").strip()
        if not instr or not out:
            return None
        user = f"{instr}\n\n{extra}".strip() if extra else instr
        return (user, out)
    if fmt == "chat":
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            return None
        return ("__CHAT__", json.dumps(msgs, ensure_ascii=False))
    raise ValueError(f"Неизвестный формат: {fmt}")


def load_and_validate(path: Path, fmt: str, cfg) -> tuple[list[dict[str, Any]], str]:
    raw = _read_jsonl(path)
    if fmt == "auto":
        fmt = detect_format(raw, cfg.data.text_field)
    cleaned, dropped = [], 0
    for rec in raw:
        ex = _extract(rec, fmt, cfg)
        if ex is None or len(ex[1]) < cfg.data.min_chars:
            dropped += 1
            continue
        cleaned.append({"_prompt": ex[0], "_target": ex[1]})
    if not cleaned:
        raise ValueError(f"{Path(path).name}: после очистки нет валидных записей (формат={fmt})")
    print(f"  [{Path(path).name}] формат={fmt}, валидных={len(cleaned)}, отброшено={dropped}")
    return cleaned, fmt


# --------------------------------------------------------------------------- #
# Токенизация с маскированием лейблов
# --------------------------------------------------------------------------- #
def _tok_text(target: str, tok, cfg) -> dict[str, list[int]]:
    ids = tok(target, add_special_tokens=True)["input_ids"]
    if cfg.data.add_eos and (not ids or ids[-1] != tok.eos_token_id):
        ids = ids + [tok.eos_token_id]
    return {"input_ids": ids, "labels": list(ids)}


def _tok_prompt_target(prompt: str, target: str, tok, cfg) -> dict[str, list[int]]:
    """instruction/chat: промпт маскируется (-100), учится только ответ."""
    if prompt == "__CHAT__":
        messages = json.loads(target)
    else:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
    if getattr(tok, "chat_template", None):
        full_ids = tok.apply_chat_template(messages, tokenize=True,
                                           add_generation_prompt=False)
        prompt_ids = tok.apply_chat_template(messages[:-1], tokenize=True,
                                             add_generation_prompt=True)
    else:
        user = "\n".join(m["content"] for m in messages[:-1] if m["role"] != "assistant")
        ptxt = f"### Запрос:\n{user}\n\n### Ответ:\n"
        prompt_ids = tok(ptxt, add_special_tokens=True)["input_ids"]
        full_ids = tok(ptxt + messages[-1]["content"], add_special_tokens=True)["input_ids"]
    if cfg.data.add_eos and (not full_ids or full_ids[-1] != tok.eos_token_id):
        full_ids = full_ids + [tok.eos_token_id]
    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    return {"input_ids": full_ids, "labels": labels}


def build_dataset(records, fmt, tok, cfg):
    """Возвращает (Dataset, n_truncated)."""
    max_len = cfg.data.max_seq_len
    n_trunc = {"count": 0}

    def _map(rec):
        if fmt == "text":
            enc = _tok_text(rec["_target"], tok, cfg)
        else:
            enc = _tok_prompt_target(rec["_prompt"], rec["_target"], tok, cfg)
        ids, labels = enc["input_ids"], enc["labels"]
        full = len(ids)
        if full > max_len:
            n_trunc["count"] += 1
            ids, labels = ids[:max_len], labels[:max_len]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}

    ds = Dataset.from_list(records)
    ds = ds.map(_map, remove_columns=ds.column_names, num_proc=cfg.data.num_proc,
                desc="Токенизация")

    if cfg.data.drop_overlong:
        before = len(ds)
        ds = ds.filter(lambda x: x["length"] <= max_len)
        print(f"  Отброшено длиннее окна: {before - len(ds)}")
    return ds, n_trunc["count"]


def corpus_stats(ds) -> dict[str, Any]:
    lengths = sorted(ds["length"])
    if not lengths:
        return {}
    pct = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))]
    return {
        "examples": len(lengths),
        "tokens_total": sum(lengths),
        "tokens_min": lengths[0],
        "tokens_max": lengths[-1],
        "tokens_mean": round(statistics.fmean(lengths), 1),
        "tokens_median": int(statistics.median(lengths)),
        "tokens_p95": pct(0.95),
        "tokens_p99": pct(0.99),
    }


# --------------------------------------------------------------------------- #
# Коллатор: паддинг батча, labels паддятся -100
# --------------------------------------------------------------------------- #
class DataCollatorCausalLM:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = 8):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_id = (tokenizer.pad_token_id
                       if tokenizer.pad_token_id is not None
                       else tokenizer.eos_token_id)

    def __call__(self, features):
        import torch
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        input_ids, labels, attn = [], [], []
        for f in features:
            ids, lab = list(f["input_ids"]), list(f["labels"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Основной вход — РОВНО 3 значения (под текущий train.py)
# --------------------------------------------------------------------------- #
def prepare(cfg, tokenizer):
    print("Подготовка корпуса...")
    train_recs, fmt = load_and_validate(cfg.data.train_file, cfg.data.fmt, cfg)

    val_path = Path(cfg.data.val_file)
    if val_path.exists():
        val_recs, _ = load_and_validate(val_path, fmt, cfg)
    else:
        print(f"  Валидационный файл не найден ({val_path.name}); без отдельной валидации.")
        val_recs = []

    train_ds, n_trunc = build_dataset(train_recs, fmt, tokenizer, cfg)
    if val_recs:
        val_ds, _ = build_dataset(val_recs, fmt, tokenizer, cfg)
    else:
        val_ds = None

    if n_trunc:
        print(f"  ВНИМАНИЕ: {n_trunc} примеров длиннее окна {cfg.data.max_seq_len} "
              "— усечены по хвосту. Подними max_seq_len или дроби сцены.")

    stats = {
        "format": fmt,
        "train": corpus_stats(train_ds),
        "val": corpus_stats(val_ds) if val_ds is not None else None,
        "truncated_examples": n_trunc,
    }
    print(f"  Статистика train: {stats['train']}")
    if stats["val"]:
        print(f"  Статистика val:   {stats['val']}")
    return train_ds, val_ds, stats
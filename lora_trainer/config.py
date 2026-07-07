"""
config.py  (Unsloth-вариант, с загрузкой из YAML/JSON)
======================================================
Значения по умолчанию (схема) живут здесь, в dataclass'ах.
Реальные настройки берутся из внешнего файла и переопределяют дефолты,
так что в этот .py лазить больше не нужно.

Приоритет источника конфига:
    1) явный путь:        load_config("config.test.yaml")  / флаг --config
    2) переменная среды:  STORYFORGE_CONFIG=...
    3) файл рядом:         config.yaml -> config.yml -> config.json
    4) если файла нет — чистые дефолты из этого файла.

Поддерживается наследование: в YAML можно указать
    extends: config.yaml
и переопределить только нужные ключи (см. config.test.yaml).
"""

from __future__ import annotations

import datetime
import json
import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

_SECTIONS = ("model", "data", "lora", "train", "gen", "export", "paths", "run")


@dataclass
class ModelConfig:
    base_model: str = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
    trust_remote_code: bool = False
    load_in_4bit: bool = True


@dataclass
class DataConfig:
    train_file: Path = PROJECT_ROOT / "corpus" / "corpus.train.jsonl"
    val_file: Path = PROJECT_ROOT / "corpus" / "corpus.val.jsonl"
    max_seq_len: int = 1536
    fmt: str = "auto"
    text_field: str = "text"
    min_chars: int = 1
    drop_overlong: bool = False
    add_eos: bool = True
    num_proc: int = 1


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class TrainSettings:
    output_dir: Path = PROJECT_ROOT / "output" / "checkpoints"
    epochs: float = 3.0
    per_device_batch_size: int = 2
    grad_accum: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    optimizer: str = "adamw_8bit"
    max_grad_norm: float = 1.0
    logging_steps: int = 5
    eval_strategy: str = "no"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 25
    save_total_limit: int = 3
    resume: bool = True
    seed: int = 42
    group_by_length: bool = True
    report_to: str = "none"


@dataclass
class GenSettings:
    max_new_tokens: int = 400
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class ExportSettings:
    lora_dir: Path = PROJECT_ROOT / "output" / "lora"
    merged_dir: Path = PROJECT_ROOT / "output" / "merged"
    gguf_dir: Path = PROJECT_ROOT / "output" / "gguf"
    gguf_quant: str = "q4_k_m"


@dataclass
class Paths:
    logs: Path = PROJECT_ROOT / "logs"
    prompts: Path = PROJECT_ROOT / "prompts" / "benchmark.txt"
    benchmark_out: Path = PROJECT_ROOT / "output" / "benchmark"
    stats_file: Path = PROJECT_ROOT / "logs" / "final_stats.json"


@dataclass
class RunConfig:
    # Имя прогона. null/None -> авто-имя (модель + r + epochs + дата).
    name: str | None = None
    # Корневая папка для всех прогонов.
    runs_dir: Path = PROJECT_ROOT / "runs"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraSettings = field(default_factory=LoraSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    gen: GenSettings = field(default_factory=GenSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    paths: Paths = field(default_factory=Paths)
    run: RunConfig = field(default_factory=RunConfig)

    min_versions: dict = field(
        default_factory=lambda: {
            "torch": "2.4.0",
            "unsloth": "2025.1.1",
            "transformers": "4.45.0",
            "datasets": "3.0.0",
        }
    )
    min_free_disk_gb: float = 10.0


def _coerce(default, value):
    if isinstance(default, Path):
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p)
    return value


def _apply_section(section_obj, overrides) -> None:
    valid = {f.name for f in fields(section_obj)}
    for key, val in (overrides or {}).items():
        if key not in valid:
            raise ValueError(
                f"Неизвестный параметр '{key}' в секции "
                f"'{type(section_obj).__name__}'. Допустимые: {sorted(valid)}"
            )
        setattr(section_obj, key, _coerce(getattr(section_obj, key), val))


def _apply_all(cfg, data) -> None:
    for name, overrides in (data or {}).items():
        if name in _SECTIONS:
            _apply_section(getattr(cfg, name), overrides)
        elif name in {"min_versions", "min_free_disk_gb"}:
            setattr(cfg, name, overrides)
        else:
            raise ValueError(
                f"Неизвестная секция '{name}' в конфиге. Допустимые: "
                f"{list(_SECTIONS) + ['min_versions', 'min_free_disk_gb']}"
            )


def _read_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "Для YAML-конфига нужен pyyaml: pip install pyyaml "
                "(или используй config.json)."
            )
        return yaml.safe_load(text) or {}
    if suffix == ".json":
        return json.loads(text) if text.strip() else {}
    raise ValueError(f"Неподдерживаемый формат конфига: {path.name} "
                     "(нужен .yaml/.yml/.json)")


def _resolve_default_path():
    env = os.environ.get("STORYFORGE_CONFIG")
    if env:
        return Path(env)
    for name in ("config.yaml", "config.yml", "config.json"):
        cand = PROJECT_ROOT / name
        if cand.exists():
            return cand
    return None


def load_config(path=None, _depth: int = 0) -> Config:
    """Грузит конфиг из файла поверх дефолтов. Поддерживает extends."""
    if _depth > 10:
        raise ValueError("Слишком глубокая цепочка extends (цикл?).")

    resolved = Path(path) if path else _resolve_default_path()
    if resolved is None:
        return Config()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Конфиг не найден: {resolved}")

    data = _read_file(resolved)

    parent = data.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        cfg = load_config(parent_path, _depth + 1)
    else:
        cfg = Config()

    _apply_all(cfg, data)
    return cfg


CONFIG = load_config()


# --------------------------------------------------------------------------- #
# Управление прогонами (runs)
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text)).strip("-").lower()


def generate_run_name(cfg) -> str:
    """Авто-имя прогона: <модель>-r<r>-e<epochs>-<дата>."""
    tag = _slug(str(cfg.model.base_model).split("/")[-1])[:28]
    ep = f"{cfg.train.epochs:g}"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    return f"{tag}-r{cfg.lora.r}-e{ep}-{ts}"


def latest_run(runs_dir: Path) -> str | None:
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return None
    dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0].name


def apply_run_paths(cfg, run_name=None, create: bool = False):
    """Переназначает все выходные пути под выбранный прогон runs/<name>/.

    create=True  (обучение): если имя не задано — генерируется авто-имя.
    create=False (чтение):   если имя не задано — берётся самый свежий прогон.
    Возвращает (name, run_root).
    """
    runs_dir = cfg.run.runs_dir
    if not Path(runs_dir).is_absolute():
        runs_dir = PROJECT_ROOT / runs_dir
    runs_dir = Path(runs_dir)

    name = run_name or cfg.run.name
    if not name:
        if create:
            name = generate_run_name(cfg)
        else:
            name = latest_run(runs_dir)
            if not name:
                raise SystemExit(
                    f"Не найдено ни одного прогона в {runs_dir}. "
                    "Сначала обучи модель: python train.py"
                )
            print(f"  (прогон не указан — беру самый свежий: {name})")

    cfg.run.name = name
    root = runs_dir / name
    cfg.train.output_dir = root / "checkpoints"
    cfg.export.lora_dir = root / "lora"
    cfg.export.merged_dir = root / "merged"
    cfg.export.gguf_dir = root / "gguf"
    cfg.paths.logs = root / "logs"
    cfg.paths.benchmark_out = root / "benchmark"
    cfg.paths.stats_file = root / "run.json"
    return name, root


def config_to_dict(cfg) -> dict:
    """Сериализует конфиг в обычный dict (Path -> str) для run.json/snapshot."""
    def conv(o):
        if is_dataclass(o):
            return {f.name: conv(getattr(o, f.name)) for f in fields(o)}
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        return o
    return conv(cfg)

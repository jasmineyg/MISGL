"""Strict configuration objects and YAML loading for MISGL training."""

from dataclasses import dataclass, fields, replace
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Type, TypeVar

from ruamel.yaml import YAML


_ConfigType = TypeVar("_ConfigType")


def _require_bool(name: str, value: Any) -> None:
    if type(value) is not bool:
        raise TypeError("{} must be a boolean".format(name))


def _require_int(name: str, value: Any, minimum: int = 1) -> None:
    if type(value) is not int:
        raise TypeError("{} must be an integer".format(name))
    if value < minimum:
        raise ValueError("{} must be at least {}".format(name, minimum))


def _require_number(name: str, value: Any, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{} must be a number".format(name))
    if not math.isfinite(float(value)):
        raise ValueError("{} must be finite".format(name))
    if value < minimum:
        raise ValueError("{} must be at least {}".format(name, minimum))


def _require_probability(name: str, value: Any) -> None:
    _require_number(name, value)
    if value >= 1.0:
        raise ValueError("{} must be smaller than 1".format(name))


def _require_text(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError("{} must be a string".format(name))
    if not value.strip():
        raise ValueError("{} must not be empty".format(name))


@dataclass(frozen=True)
class ModelConfig:
    encoder_dim: int
    classifier_dim: int
    dropout: float
    gat_heads: int
    gat_attention_dropout: float
    gat_feature_dropout: float
    gat_negative_slope: float
    gat_residual: bool

    def __post_init__(self) -> None:
        _require_int("model.encoder_dim", self.encoder_dim)
        _require_int("model.classifier_dim", self.classifier_dim)
        _require_probability("model.dropout", self.dropout)
        _require_int("model.gat_heads", self.gat_heads)
        if self.encoder_dim % self.gat_heads != 0:
            raise ValueError("model.encoder_dim must be divisible by model.gat_heads")
        _require_probability("model.gat_attention_dropout", self.gat_attention_dropout)
        _require_probability("model.gat_feature_dropout", self.gat_feature_dropout)
        _require_number("model.gat_negative_slope", self.gat_negative_slope)
        _require_bool("model.gat_residual", self.gat_residual)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int
    max_grad_norm: float
    label_smoothing: float
    loss: str
    focal_gamma: float

    def __post_init__(self) -> None:
        _require_int("training.batch_size", self.batch_size)
        _require_int("training.epochs", self.epochs)
        _require_number("training.lr", self.lr)
        if self.lr == 0:
            raise ValueError("training.lr must be greater than 0")
        _require_number("training.weight_decay", self.weight_decay)
        _require_int("training.patience", self.patience)
        _require_number("training.max_grad_norm", self.max_grad_norm)
        if self.max_grad_norm == 0:
            raise ValueError("training.max_grad_norm must be greater than 0")
        _require_probability("training.label_smoothing", self.label_smoothing)
        _require_text("training.loss", self.loss)
        if self.loss not in {"bce", "focal", "weighted_bce"}:
            raise ValueError(
                "training.loss must be one of: bce, focal, weighted_bce"
            )
        _require_number("training.focal_gamma", self.focal_gamma)


@dataclass(frozen=True)
class MILHeadConfig:
    enabled: bool
    attention_dim: int
    structure_dim: int
    structure_gate_dim: int
    structure_dropout: float
    structure_residual_init: float
    attention_loss_weight: float
    attention_loss_eps: float

    def __post_init__(self) -> None:
        _require_bool("mil_head.enabled", self.enabled)
        _require_int("mil_head.attention_dim", self.attention_dim)
        _require_int("mil_head.structure_dim", self.structure_dim)
        _require_int("mil_head.structure_gate_dim", self.structure_gate_dim)
        _require_probability(
            "mil_head.structure_dropout", self.structure_dropout
        )
        _require_number(
            "mil_head.structure_residual_init", self.structure_residual_init
        )
        _require_number("mil_head.attention_loss_weight", self.attention_loss_weight)
        _require_probability("mil_head.attention_loss_eps", self.attention_loss_eps)
        if self.attention_loss_eps == 0:
            raise ValueError("mil_head.attention_loss_eps must be greater than 0")


@dataclass(frozen=True)
class POSHeadConfig:
    enabled: bool
    top_k: int
    hidden_dim: int
    dropout: float
    epochs: int
    patience: int
    lr: float
    weight_decay: float

    def __post_init__(self) -> None:
        _require_bool("pos_head.enabled", self.enabled)
        _require_int("pos_head.top_k", self.top_k)
        _require_int("pos_head.hidden_dim", self.hidden_dim)
        _require_probability("pos_head.dropout", self.dropout)
        _require_int("pos_head.epochs", self.epochs)
        _require_int("pos_head.patience", self.patience)
        _require_number("pos_head.lr", self.lr)
        if self.lr == 0:
            raise ValueError("pos_head.lr must be greater than 0")
        _require_number("pos_head.weight_decay", self.weight_decay)


@dataclass(frozen=True)
class Config:
    datasets: Tuple[str, ...]
    run_name: str
    data_dir: str
    output_dir: str
    device: str
    cuda_device: Optional[str]
    seed: int
    folds: int
    model: ModelConfig
    training: TrainingConfig
    mil_head: MILHeadConfig
    pos_head: POSHeadConfig

    def __post_init__(self) -> None:
        if not isinstance(self.datasets, tuple):
            raise TypeError("datasets must be a tuple")
        if not self.datasets:
            raise ValueError("datasets must not be empty")
        for dataset in self.datasets:
            _require_text("datasets entry", dataset)
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("datasets must not contain duplicates")
        _require_text("run_name", self.run_name)
        if "/" in self.run_name or "\\" in self.run_name:
            raise ValueError("run_name must not contain path separators")
        _require_text("data_dir", self.data_dir)
        _require_text("output_dir", self.output_dir)
        _require_text("device", self.device)
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be either cpu or cuda")
        if self.cuda_device is not None:
            _require_text("cuda_device", self.cuda_device)
        if self.device == "cuda" and self.cuda_device is None:
            raise ValueError("cuda_device is required when device is cuda")
        _require_int("seed", self.seed, minimum=0)
        _require_int("folds", self.folds, minimum=3)
        if self.pos_head.enabled and not self.mil_head.enabled:
            raise ValueError("pos_head requires mil_head to be enabled")


def _build_config_section(
    config_type: Type[_ConfigType], data: Any, section_name: str
) -> _ConfigType:
    if not isinstance(data, Mapping):
        raise TypeError("{} must be a mapping".format(section_name))

    expected = {item.name for item in fields(config_type)}
    actual = set(data)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise ValueError(
            "unknown {} keys: {}".format(section_name, ", ".join(sorted(unknown)))
        )
    if missing:
        raise ValueError(
            "missing {} keys: {}".format(section_name, ", ".join(sorted(missing)))
        )
    return config_type(**dict(data))


def load_config(path: str) -> Config:
    """Load exactly one YAML file into a validated immutable configuration."""

    config_path = Path(path)
    yaml = YAML(typ="safe")
    with config_path.open("r", encoding="utf-8") as config_file:
        raw = yaml.load(config_file)

    if not isinstance(raw, Mapping):
        raise TypeError("configuration root must be a mapping")

    expected = {item.name for item in fields(Config)}
    actual = set(raw)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise ValueError("unknown config keys: {}".format(", ".join(sorted(unknown))))
    if missing:
        raise ValueError("missing config keys: {}".format(", ".join(sorted(missing))))

    datasets = raw["datasets"]
    if not isinstance(datasets, (list, tuple)) or isinstance(datasets, str):
        raise TypeError("datasets must be a sequence of strings")

    return Config(
        datasets=tuple(datasets),
        run_name=raw["run_name"],
        data_dir=raw["data_dir"],
        output_dir=raw["output_dir"],
        device=raw["device"],
        cuda_device=raw["cuda_device"],
        seed=raw["seed"],
        folds=raw["folds"],
        model=_build_config_section(ModelConfig, raw["model"], "model"),
        training=_build_config_section(
            TrainingConfig, raw["training"], "training"
        ),
        mil_head=_build_config_section(MILHeadConfig, raw["mil_head"], "mil_head"),
        pos_head=_build_config_section(POSHeadConfig, raw["pos_head"], "pos_head"),
    )


def apply_overrides(
    config: Config,
    datasets: Optional[Sequence[str]] = None,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    mil_head: Optional[bool] = None,
    pos_head: Optional[bool] = None,
) -> Config:
    """Apply explicit CLI overrides and revalidate the resulting head state."""

    mil_config = (
        config.mil_head
        if mil_head is None
        else replace(config.mil_head, enabled=mil_head)
    )
    pos_config = (
        config.pos_head
        if pos_head is None
        else replace(config.pos_head, enabled=pos_head)
    )

    return replace(
        config,
        datasets=config.datasets if datasets is None else tuple(datasets),
        data_dir=config.data_dir if data_dir is None else data_dir,
        output_dir=config.output_dir if output_dir is None else output_dir,
        device=config.device if device is None else device,
        mil_head=mil_config,
        pos_head=pos_config,
    )

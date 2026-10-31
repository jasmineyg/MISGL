"""Training settings and the paper defaults."""

from dataclasses import dataclass
import math
from typing import Any, Optional, Tuple


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
    """Residual GAT backbone and the first-stage classifier."""

    node_dim: int
    classifier_dim: int
    dropout: float
    gat_heads: int
    gat_attention_dropout: float
    gat_feature_dropout: float
    gat_negative_slope: float
    gat_residual: bool

    def __post_init__(self) -> None:
        _require_int("model.node_dim", self.node_dim)
        _require_int("model.classifier_dim", self.classifier_dim)
        _require_probability("model.dropout", self.dropout)
        _require_int("model.gat_heads", self.gat_heads)
        if self.node_dim % self.gat_heads != 0:
            raise ValueError("model.node_dim must be divisible by model.gat_heads")
        _require_probability("model.gat_attention_dropout", self.gat_attention_dropout)
        _require_probability("model.gat_feature_dropout", self.gat_feature_dropout)
        _require_number("model.gat_negative_slope", self.gat_negative_slope)
        _require_bool("model.gat_residual", self.gat_residual)


@dataclass(frozen=True)
class TrainingConfig:
    """First-stage optimization; the Context-Head has its own settings below."""

    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int
    max_grad_norm: float
    loss: str

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
        _require_text("training.loss", self.loss)
        if self.loss != "bce":
            raise ValueError("The paper implementation uses BCE only")


@dataclass(frozen=True)
class InternalHeadConfig:
    """Structural encoding, gated attention and residual fusion (Section 4.2)."""

    enabled: bool
    attention_dim: int
    structure_dim: int
    structure_gate_dim: int
    residual_scale_init: float

    def __post_init__(self) -> None:
        _require_bool("internal_head.enabled", self.enabled)
        _require_int("internal_head.attention_dim", self.attention_dim)
        _require_int("internal_head.structure_dim", self.structure_dim)
        if self.structure_dim != 32:
            raise ValueError(
                "internal_head.structure_dim must be 32 for the final encoder"
            )
        _require_int("internal_head.structure_gate_dim", self.structure_gate_dim)
        _require_number("internal_head.residual_scale_init", self.residual_scale_init)


@dataclass(frozen=True)
class ContextHeadConfig:
    """Coarse-graph Top-K and second-stage optimization (Sections 4.3-4.4)."""

    enabled: bool
    top_k: int
    hidden_dim: int
    dropout: float
    epochs: int
    patience: int
    lr: float
    weight_decay: float

    def __post_init__(self) -> None:
        _require_bool("context_head.enabled", self.enabled)
        _require_int("context_head.top_k", self.top_k)
        _require_int("context_head.hidden_dim", self.hidden_dim)
        _require_probability("context_head.dropout", self.dropout)
        _require_int("context_head.epochs", self.epochs)
        _require_int("context_head.patience", self.patience)
        _require_number("context_head.lr", self.lr)
        if self.lr == 0:
            raise ValueError("context_head.lr must be greater than 0")
        _require_number("context_head.weight_decay", self.weight_decay)


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
    internal_head: InternalHeadConfig
    context_head: ContextHeadConfig

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


def default_config() -> Config:
    """Return the paper hyperparameters; no external configuration file is needed."""
    return Config(
        datasets=("synthetic_milinst_both_useful_v2",),
        run_name="nasgl",
        data_dir="data/syn3",
        output_dir="outputs",
        device="cuda",
        cuda_device="0",
        seed=1024,
        folds=10,
        model=ModelConfig(
            node_dim=256,
            classifier_dim=64,
            dropout=0.5,
            gat_heads=4,
            gat_attention_dropout=0.5,
            gat_feature_dropout=0.5,
            gat_negative_slope=0.2,
            gat_residual=True,
        ),
        training=TrainingConfig(
            batch_size=64,
            epochs=200,
            lr=0.001,
            weight_decay=0.0,
            patience=30,
            max_grad_norm=2.0,
            loss="bce",
        ),
        internal_head=InternalHeadConfig(
            enabled=True,
            attention_dim=128,
            structure_dim=32,
            structure_gate_dim=64,
            residual_scale_init=0.1,
        ),
        context_head=ContextHeadConfig(
            enabled=True,
            top_k=16,
            hidden_dim=256,
            dropout=0.5,
            epochs=300,
            patience=50,
            lr=0.001,
            weight_decay=0.0005,
        ),
    )

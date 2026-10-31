"""Two-stage NASGL training, BCE loss and classification metrics."""

import copy
import os
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from MISGL.config import Config
from MISGL.context_head import ContextHead
from MISGL.model import SubgraphEncoder


def binary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits.reshape(-1)
    targets = targets.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have equal shape, got {logits.shape} and {targets.shape}"
        )
    return F.binary_cross_entropy_with_logits(logits, targets)


METRIC_NAMES = (
    "acc",
    "precision",
    "recall",
    "f1",
    "balanced_acc",
    "roc_auc",
    "pr_auc",
    "tn",
    "fp",
    "fn",
    "tp",
)


def binary_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> Dict[str, Union[float, int]]:
    """Return the full metric set; both classes must be present."""
    logits = logits.detach().cpu().reshape(-1)
    labels = labels.detach().cpu().to(torch.long).reshape(-1)
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have equal shape")
    if logits.numel() == 0:
        raise ValueError("cannot evaluate an empty split")

    y_true = labels.numpy()
    if not np.array_equal(np.unique(y_true), np.asarray([0, 1])):
        raise ValueError("every evaluated split must contain both binary classes")

    probabilities = torch.sigmoid(logits).numpy()
    predictions = (probabilities > 0.5).astype(np.int64)
    tn, fp, fn, tp = metrics.confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "acc": float(metrics.accuracy_score(y_true, predictions)),
        "precision": float(
            metrics.precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(metrics.recall_score(y_true, predictions, zero_division=0)),
        "f1": float(metrics.f1_score(y_true, predictions, zero_division=0)),
        "balanced_acc": float(metrics.balanced_accuracy_score(y_true, predictions)),
        "roc_auc": float(metrics.roc_auc_score(y_true, probabilities)),
        "pr_auc": float(metrics.average_precision_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def summarize(
    fold_metrics: List[Dict[str, Union[float, int]]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate numeric fold metrics with sample standard deviation."""
    if not fold_metrics:
        raise ValueError("cannot summarize zero folds")
    summary = {}
    for name in METRIC_NAMES:
        values = np.asarray(
            [float(item[name]) for item in fold_metrics], dtype=np.float64
        )
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return summary


def runtime_device(config: Config) -> torch.device:
    if config.device == "cuda":
        if config.cuda_device is None:
            raise ValueError("cuda_device must be set when device is 'cuda'")
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
    return torch.device(config.device)


def _move_batch(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _is_better(
    accuracy: float,
    loss: float,
    best_accuracy: float,
    best_loss: float,
    accuracy_delta: float,
    loss_delta: float,
) -> bool:
    if accuracy > best_accuracy + accuracy_delta:
        return True
    return (
        abs(accuracy - best_accuracy) <= accuracy_delta
        and loss < best_loss - loss_delta
    )


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def encode_subgraphs(
    model: SubgraphEncoder,
    loader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = []
    z_int = []
    labels = []
    sample_indices = []
    model.eval()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = _move_batch(cpu_batch, device)
            output = model(batch)
            logits.append(output.logits.detach().cpu().reshape(-1))
            z_int.append(output.z_int.detach().cpu())
            labels.append(batch["label"].detach().cpu().reshape(-1))
            sample_indices.append(batch["sample_index"].detach().cpu().reshape(-1))
    if not logits:
        raise ValueError("cannot predict an empty dataset")
    return (
        torch.cat(logits),
        torch.cat(z_int),
        torch.cat(labels),
        torch.cat(sample_indices),
    )


def evaluate_splits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    split_positions: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    return {
        name: binary_metrics(logits[positions], labels[positions])
        for name, positions in split_positions.items()
    }


def train_internal_stage(
    config: Config,
    input_dim: int,
    train_loader,
    val_loader,
    device: torch.device,
) -> Tuple[SubgraphEncoder, Dict[str, float]]:
    """Stage 1: train the residual GAT and Internal-Head using subgraph labels."""
    model = SubgraphEncoder(config, input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    best_state = None
    best_accuracy = float("-inf")
    best_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(config.training.epochs):
        model.train()
        for cpu_batch in train_loader:
            batch = _move_batch(cpu_batch, device)
            optimizer.zero_grad()
            output = model(batch)
            loss = binary_loss(output.logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.max_grad_norm
            )
            optimizer.step()

        val_logits, _, val_labels, _ = encode_subgraphs(model, val_loader, device)
        val_loss = float(binary_loss(val_logits, val_labels).item())
        val_accuracy = float(binary_metrics(val_logits, val_labels)["acc"])
        if _is_better(
            val_accuracy,
            val_loss,
            best_accuracy,
            best_loss,
            accuracy_delta=0.0,
            loss_delta=1.0e-4,
        ):
            best_state = copy.deepcopy(model.state_dict())
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.patience:
                break

    if best_state is None:
        raise RuntimeError("encoder training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "epoch": int(best_epoch),
        "val_acc": best_accuracy,
        "val_loss": best_loss,
    }


def train_context_stage(
    config: Config,
    z_int: torch.Tensor,
    labels: torch.Tensor,
    split_positions: Dict[str, np.ndarray],
    a_context: torch.Tensor,
    device: torch.device,
) -> Tuple[ContextHead, torch.Tensor, Dict[str, float]]:
    """Stage 2: train Context-Head using the frozen internal representations."""
    z_int = z_int.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.float32)
    masks = {}
    for name, positions in split_positions.items():
        mask = torch.zeros(labels.numel(), dtype=torch.bool, device=device)
        mask[torch.as_tensor(positions, dtype=torch.long, device=device)] = True
        masks[name] = mask

    model = ContextHead(
        internal_dim=z_int.size(1),
        hidden_dim=config.context_head.hidden_dim,
        dropout=config.context_head.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.context_head.lr,
        weight_decay=config.context_head.weight_decay,
    )

    best_state = None
    best_accuracy = float("-inf")
    best_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    for epoch in range(config.context_head.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(z_int, a_context)
        loss = binary_loss(
            logits[masks["train"]],
            labels[masks["train"]],
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_logits = model(z_int, a_context)[masks["val"]]
        val_labels = labels[masks["val"]]
        val_loss = float(binary_loss(val_logits, val_labels).item())
        val_accuracy = float(binary_metrics(val_logits, val_labels)["acc"])
        if _is_better(
            val_accuracy,
            val_loss,
            best_accuracy,
            best_loss,
            accuracy_delta=1.0e-8,
            loss_delta=1.0e-8,
        ):
            best_state = copy.deepcopy(model.state_dict())
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.context_head.patience:
                break

    if best_state is None:
        raise RuntimeError("Context-Head training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits = model(z_int, a_context).detach().cpu()
    return (
        model,
        logits,
        {
            "epoch": int(best_epoch),
            "val_acc": best_accuracy,
            "val_loss": best_loss,
        },
    )

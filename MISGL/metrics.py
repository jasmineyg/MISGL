"""Strict binary-classification metrics."""

from typing import Dict, List, Union

import numpy as np
import torch
from sklearn import metrics


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
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "acc": float(metrics.accuracy_score(y_true, predictions)),
        "precision": float(metrics.precision_score(y_true, predictions, zero_division=0)),
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
        values = np.asarray([float(item[name]) for item in fold_metrics], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return summary

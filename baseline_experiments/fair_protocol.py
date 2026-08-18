"""Shared immutable MISGL evaluation protocol for external baselines."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROTOCOL_NAME = "single_grouped_stratified_10fold_8_1_1"
PROTOCOL_SEED = 1024
NUM_FOLDS = 10


def load_dataset_entry(execution_manifest: str | os.PathLike[str], dataset_key: str) -> Dict[str, Any]:
    """Resolve one canonical dataset entry without duplicating the MISGL registry."""
    manifest_path = Path(execution_manifest)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("seed") != PROTOCOL_SEED or manifest.get("folds") != NUM_FOLDS:
        raise ValueError(f"Unexpected execution protocol in {manifest_path}")
    matches = [entry for entry in manifest["entries"] if entry["dataset_key"] == dataset_key]
    if not matches:
        raise KeyError(f"Unknown dataset key: {dataset_key}")
    entry = dict(matches[0])
    basename = Path(entry["data_name"]).name
    candidates = [
        Path(entry["data_dir"]) / f"{entry['data_name']}_processed.pkl",
        Path(entry["data_dir"]) / f"{basename}_processed.pkl",
    ]
    existing = next((candidate for candidate in candidates if candidate.is_file()), None)
    if existing is None:
        raise FileNotFoundError(
            "Dataset pickle not found; checked: " + ", ".join(str(path) for path in candidates)
        )
    entry["pkl_path"] = str(existing)
    entry["execution_manifest"] = str(manifest_path)
    return entry


def load_fixed_folds(split_manifest: str | os.PathLike[str], sample_count: int) -> List[Dict[str, np.ndarray]]:
    """Load the exact MISGL folds: test=f, val=f+1, train=remaining eight."""
    path = Path(split_manifest)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "cv_seed": PROTOCOL_SEED,
        "cv_num_folds": NUM_FOLDS,
        "cv_val_policy": "adjacent",
        "protocol": "grouped_stratified_cv_8_1_1",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}, got {payload.get(key)!r}")
    raw_folds = payload.get("folds", [])
    if len(raw_folds) != NUM_FOLDS:
        raise ValueError(f"{path}: expected {NUM_FOLDS} folds, got {len(raw_folds)}")

    fold_indices = [np.asarray(fold["sample_indices"], dtype=np.int64) for fold in raw_folds]
    all_indices = np.concatenate(fold_indices)
    if len(all_indices) != sample_count:
        raise ValueError(f"{path}: folds contain {len(all_indices)} samples, data has {sample_count}")
    if len(np.unique(all_indices)) != sample_count:
        raise ValueError(f"{path}: folds overlap or contain duplicate sample indices")
    if not np.array_equal(np.sort(all_indices), np.arange(sample_count, dtype=np.int64)):
        raise ValueError(f"{path}: folds do not cover exactly [0, {sample_count})")

    splits: List[Dict[str, np.ndarray]] = []
    for fold_id in range(NUM_FOLDS):
        val_fold = (fold_id + 1) % NUM_FOLDS
        train_folds = [idx for idx in range(NUM_FOLDS) if idx not in (fold_id, val_fold)]
        train = np.concatenate([fold_indices[idx] for idx in train_folds]).astype(np.int64)
        val = fold_indices[val_fold].copy()
        test = fold_indices[fold_id].copy()
        if set(train).intersection(val) or set(train).intersection(test) or set(val).intersection(test):
            raise ValueError(f"{path}: leakage detected in fold {fold_id}")
        splits.append(
            {
                "fold_id": fold_id,
                "fold_seed": PROTOCOL_SEED + fold_id,
                "train_folds": np.asarray(train_folds, dtype=np.int64),
                "val_fold": np.asarray([val_fold], dtype=np.int64),
                "test_fold": np.asarray([fold_id], dtype=np.int64),
                "train_indices": train,
                "val_indices": val,
                "test_indices": test,
            }
        )
    return splits


def split_digest(split: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("train_indices", "val_indices", "test_indices"):
        values = np.ascontiguousarray(split[key], dtype=np.int64)
        digest.update(key.encode("ascii"))
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def seed_everything(seed: int, torch_module: Any | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is None:
        return
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed(seed)
        torch_module.cuda.manual_seed_all(seed)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> Dict[str, float | int]:
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities_array = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(labels_array) != len(probabilities_array) or len(labels_array) == 0:
        raise ValueError("labels and probabilities must be non-empty and have equal length")
    # A score exactly at the decision boundary is assigned to the negative
    # class, matching the saved MISGL predictions, RGMIL's original rule, and
    # SubGNN argmax tie behavior.
    predictions = (probabilities_array > 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels_array, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics: Dict[str, float | int] = {
        "n": int(len(labels_array)),
        "acc": float(accuracy_score(labels_array, predictions)),
        "precision": float(precision_score(labels_array, predictions, zero_division=0)),
        "recall": float(recall_score(labels_array, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "f1_macro": float(f1_score(labels_array, predictions, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(labels_array, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(labels_array)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels_array, probabilities_array))
        metrics["pr_auc"] = float(average_precision_score(labels_array, probabilities_array))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def aggregate_fold_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    metric_names = [
        "acc",
        "precision",
        "recall",
        "specificity",
        "f1",
        "f1_macro",
        "balanced_acc",
        "roc_auc",
        "pr_auc",
    ]
    aggregate: Dict[str, Dict[str, float]] = {}
    for metric in metric_names:
        values = np.asarray([record[metric] for record in records], dtype=np.float64)
        aggregate[metric] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return aggregate


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def save_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)


def dataset_fingerprint(path: str | os.PathLike[str]) -> Dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    return {
        "path": str(source),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def runtime_info(torch_module: Any | None = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    if torch_module is not None:
        info.update(
            {
                "torch": torch_module.__version__,
                "cuda_available": bool(torch_module.cuda.is_available()),
                "cuda_version": torch_module.version.cuda,
                "device_name": (
                    torch_module.cuda.get_device_name(0) if torch_module.cuda.is_available() else "cpu"
                ),
            }
        )
    return info

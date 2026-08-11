"""Unified fixed-CV training for the MIL and POS heads."""

import copy
import logging
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from MISGL.config import Config
from MISGL.data import DatasetBundle
from MISGL.keys import LABEL, SAMPLE_INDEX
from MISGL.losses import binary_loss, model_loss
from MISGL.metrics import binary_metrics, summarize
from MISGL.models.encoder import MISGLModel
from MISGL.models.pos_head import POSHead
from MISGL.position_graph import (
    build_position_adjacency,
    row_normalize,
    to_torch_sparse,
)


SPLITS = ("train", "val", "test")


def _runtime_device(config: Config) -> torch.device:
    if config.device == "cuda":
        if config.cuda_device is None:
            raise ValueError("cuda_device must be set when device is 'cuda'")
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
    return torch.device(config.device)


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _positive_class_weight(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    if negative_count == 0 or positive_count == 0:
        raise ValueError("every training fold must contain both binary classes")
    return float(negative_count) / float(positive_count)


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


def _set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _predict_all(
    model: MISGLModel,
    loader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = []
    embeddings = []
    labels = []
    sample_indices = []
    model.eval()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = _move_batch(cpu_batch, device)
            output = model(batch)
            logits.append(output.logits.detach().cpu().reshape(-1))
            embeddings.append(output.embedding.detach().cpu())
            labels.append(batch[LABEL].detach().cpu().reshape(-1))
            sample_indices.append(batch[SAMPLE_INDEX].detach().cpu().reshape(-1))
    if not logits:
        raise ValueError("cannot predict an empty dataset")
    return (
        torch.cat(logits),
        torch.cat(embeddings),
        torch.cat(labels),
        torch.cat(sample_indices),
    )


def _split_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    split_positions: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    return {
        name: binary_metrics(logits[positions], labels[positions])
        for name, positions in split_positions.items()
    }


def train_encoder(
    config: Config,
    input_dim: int,
    train_loader,
    val_loader,
    train_labels: np.ndarray,
    device: torch.device,
) -> Tuple[MISGLModel, Dict[str, float]]:
    """Train the node encoder and optional MIL head for one fold."""
    model = MISGLModel(config, input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    pos_weight = (
        _positive_class_weight(train_labels)
        if config.training.loss == "weighted_bce"
        else None
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
            loss = model_loss(output, batch[LABEL], config, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()

        val_logits, _, val_labels, _ = _predict_all(model, val_loader, device)
        val_loss = float(binary_loss(val_logits, val_labels, config, pos_weight=pos_weight).item())
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


def train_pos_head(
    config: Config,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    split_positions: Dict[str, np.ndarray],
    normalized_adjacency: torch.Tensor,
    device: torch.device,
) -> Tuple[POSHead, torch.Tensor, Dict[str, float]]:
    """Train the POS head on frozen MIL embeddings for one fold."""
    embeddings = embeddings.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.float32)
    masks = {}
    for name, positions in split_positions.items():
        mask = torch.zeros(labels.numel(), dtype=torch.bool, device=device)
        mask[torch.as_tensor(positions, dtype=torch.long, device=device)] = True
        masks[name] = mask

    train_labels = labels[masks["train"]].detach().cpu().numpy()
    pos_weight = (
        _positive_class_weight(train_labels)
        if config.training.loss == "weighted_bce"
        else None
    )
    model = POSHead(
        embedding_dim=embeddings.size(1),
        hidden_dim=config.pos_head.hidden_dim,
        dropout=config.pos_head.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.pos_head.lr,
        weight_decay=config.pos_head.weight_decay,
    )

    best_state = None
    best_accuracy = float("-inf")
    best_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    for epoch in range(config.pos_head.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(embeddings, normalized_adjacency)
        loss = binary_loss(
            logits[masks["train"]],
            labels[masks["train"]],
            config,
            pos_weight=pos_weight,
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_logits = model(embeddings, normalized_adjacency)[masks["val"]]
        val_labels = labels[masks["val"]]
        val_loss = float(binary_loss(val_logits, val_labels, config, pos_weight=pos_weight).item())
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
            if stale_epochs >= config.pos_head.patience:
                break

    if best_state is None:
        raise RuntimeError("POS-head training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits = model(embeddings, normalized_adjacency).detach().cpu()
    return model, logits, {
        "epoch": int(best_epoch),
        "val_acc": best_accuracy,
        "val_loss": best_loss,
    }


def _run_directory(config: Config, dataset_name: str) -> Path:
    mode = "mil" if not config.pos_head.enabled else "mil_pos"
    if not config.mil_head.enabled:
        mode = "baseline"
    return Path(config.output_dir) / dataset_name / f"{config.run_name}_{mode}"


def run_dataset(config: Config, dataset_name: str, device: torch.device) -> Dict:
    bundle = DatasetBundle.load(
        config,
        dataset_name,
        require_position_graph=config.pos_head.enabled,
    )
    output_dir = _run_directory(config, dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_adjacency = None
    if config.pos_head.enabled:
        position_adjacency = build_position_adjacency(
            bundle.original_graph,
            bundle.assignment_matrix,
            bundle.subgraph_ids,
            top_k=config.pos_head.top_k,
        )
        normalized_adjacency = to_torch_sparse(
            row_normalize(position_adjacency),
            device,
        )

    fold_results = []
    for fold in range(config.folds):
        _set_seed(config.seed + fold, device)
        split_positions = bundle.split(fold)
        loaders = bundle.loaders(fold, config.training.batch_size)
        train_labels = bundle.labels[split_positions["train"]]
        model, encoder_selection = train_encoder(
            config,
            bundle.feature_dim,
            loaders["train"],
            loaders["val"],
            train_labels,
            device,
        )

        encoder_logits, embeddings, labels, sample_indices = _predict_all(
            model,
            bundle.all_loader(config.training.batch_size),
            device,
        )
        expected_indices = torch.as_tensor(bundle.sample_indices, dtype=torch.long)
        if not torch.equal(sample_indices, expected_indices):
            raise RuntimeError("all_loader changed the canonical sample order")
        encoder_metrics = _split_metrics(encoder_logits, labels, split_positions)

        final_logits = encoder_logits
        pos_model = None
        pos_selection = None
        if config.pos_head.enabled:
            pos_model, final_logits, pos_selection = train_pos_head(
                config,
                embeddings,
                labels,
                split_positions,
                normalized_adjacency,
                device,
            )
        final_metrics = _split_metrics(final_logits, labels, split_positions)

        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": bundle.feature_dim,
                "selection": encoder_selection,
            },
            fold_dir / "encoder.pt",
        )
        if pos_model is not None:
            torch.save(
                {"state_dict": pos_model.state_dict(), "selection": pos_selection},
                fold_dir / "pos_head.pt",
            )

        fold_result = {
            "fold": fold,
            "encoder_selection": encoder_selection,
            "pos_selection": pos_selection,
            "encoder": encoder_metrics,
            "final": final_metrics,
        }
        fold_results.append(fold_result)
        logging.info(
            "%s fold=%d test_acc=%.4f test_f1=%.4f",
            dataset_name,
            fold,
            final_metrics["test"]["acc"],
            final_metrics["test"]["f1"],
        )

    result = {
        "dataset": dataset_name,
        "folds": fold_results,
        "summary": {
            split: summarize([item["final"][split] for item in fold_results])
            for split in SPLITS
        },
    }
    torch.save(
        {"config": asdict(config), "result": result},
        output_dir / "metrics.pt",
    )
    return result


def run(config: Config) -> Dict[str, Dict]:
    """Run all configured datasets; any error stops the run."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = _runtime_device(config)
    results = {
        dataset_name: run_dataset(config, dataset_name, device)
        for dataset_name in config.datasets
    }
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": asdict(config), "results": results},
        Path(config.output_dir) / f"{config.run_name}_summary.pt",
    )
    return results

#!/usr/bin/env python3
"""Execute one dataset/branch/fold with strict stage-level reuse."""

import argparse
import copy
import json
import logging
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

import run_coarse_gcn_paper as core
from MISGL.models.encoder import MISGLEncoder
from MISGL.utils import coarse_graph, reproducibility
from MISGL.utils.load_data import GraphDataLoaderWrapper
from offline_attention import export_test_positive_attention


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--branch", choices=("mean", "mil"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_entry(path, dataset, branch, fold):
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    matches = [
        item
        for item in manifest["entries"]
        if item["dataset_key"] == dataset
        and item["branch"] == branch
        and int(item["fold"]) == int(fold)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest entry, found {len(matches)}")
    return manifest, matches[0]


def config_path(work, entry):
    return os.path.join(
        work,
        "results/paper_5x10_20260813/configs",
        f"{entry['kind']}_{entry['branch']}_seed1024.yml",
    )


def core_args(work, entry, protocol, device):
    return SimpleNamespace(
        hparam_path=config_path(work, entry),
        data_name=entry["data_name"],
        processed_data_dir=entry["data_dir"],
        device=device,
        out_dir=entry["canonical_dir"],
        fold_idx=int(entry["fold"]),
        all_folds=False,
        create_split_if_missing=False,
        top_k=int(protocol["coarse_top_k"]),
        stage1_epochs=None,
        stage1_patience=None,
        stage2_epochs=int(protocol["epochs"]),
        stage2_patience=int(protocol["patience"]),
        stage2_lr=float(protocol["learning_rate"]),
        stage2_weight_decay=float(protocol["weight_decay"]),
        stage2_hidden_dim=protocol.get("hidden_dim"),
        stage2_dropout=float(protocol["dropout"]),
        seed=int(entry["seed"]),
        synthetic_smoke=False,
    )


def branch_is_mil(hparams):
    return bool((getattr(hparams, "branch_b", {}) or {}).get("use", False))


def validate_split(result_path, split_meta):
    with open(result_path, "r", encoding="utf-8") as handle:
        result = json.load(handle)
    old = result.get("split", {})
    for key in ("train_indices", "val_indices", "test_indices"):
        # Split identity is membership-based; frozen payloads retain orig_indices.
        old_indices = sorted(int(v) for v in old.get(key, []))
        current_indices = sorted(int(v) for v in split_meta[key])
        if old_indices != current_indices:
            raise ValueError(f"Legacy split mismatch for {key}: {result_path}")
    return result


def validate_stage1_checkpoint(path, expected_mil, seed):
    checkpoint = torch.load(path, map_location="cpu")
    hparams = checkpoint.get("hparams", {})
    actual_mil = bool((hparams.get("branch_b") or {}).get("use", False))
    if actual_mil != expected_mil:
        raise ValueError(f"Stage-1 branch mismatch in {path}")
    if int(hparams.get("cv_seed", seed)) != int(seed):
        raise ValueError(f"Stage-1 seed mismatch in {path}")
    if int(hparams.get("cv_num_folds", 10)) != 10:
        raise ValueError(f"Stage-1 fold-count mismatch in {path}")
    return checkpoint


def normalize_embedding_payload(payload, branch):
    feature_key = "z_mil" if branch == "mil" else "z_mean"
    features = payload.get(feature_key)
    if features is None:
        features = payload.get("z_mil")
    if features is None:
        raise KeyError(f"Embedding payload lacks {feature_key}")
    logits = payload.get("stage1_logits")
    if logits is None:
        raise KeyError("Embedding payload lacks stage1_logits")
    logits = logits.reshape(-1).detach().cpu()
    labels = payload["labels"].reshape(-1).to(torch.int64).detach().cpu()
    orig_indices = payload["orig_indices"].reshape(-1).to(torch.int64).detach().cpu()
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).to(torch.int64)
    return {
        feature_key: features.detach().cpu(),
        "labels": labels,
        "orig_indices": orig_indices,
        "logits": logits,
        "probabilities": probs,
        "predictions": preds,
    }


def validate_embedding_coverage(payload, loader):
    actual = sorted(int(v) for v in payload["orig_indices"].tolist())
    expected = sorted(int(v) for v in loader.cv_orig_indices)
    if actual != expected:
        raise ValueError("All-bag embedding cache does not cover the canonical CV population")


def find_coarse_source(manifest, entry):
    for candidate in manifest["entries"]:
        if candidate["dataset_key"] != entry["dataset_key"]:
            continue
        legacy_result = candidate["stage1"].get("legacy_result")
        if not legacy_result or not os.path.isfile(legacy_result):
            continue
        with open(legacy_result, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        path = result.get("paths", {}).get("coarse_adj")
        if path and os.path.isfile(path):
            return path
    return None


def load_or_build_coarse(manifest, entry, args, hparams, loader):
    source = find_coarse_source(manifest, entry)
    reusing_stage1 = entry["stage1"].get("status") == "reuse"
    if source:
        adjacency, metadata = coarse_graph.load_coarse_adjacency(source)
        if core.cache_metadata_matches(metadata, core.active_subgraph_ids(loader), args.top_k):
            logging.info("Reusing coarse adjacency: %s", source)
            return adjacency, source
        if reusing_stage1:
            raise ValueError(
                "Refusing cross-version reuse: the reused Stage-1 bundle's coarse graph "
                f"is incompatible with the active dataset: {source}"
            )
        logging.warning("Ignoring incompatible legacy coarse adjacency: %s", source)
    elif reusing_stage1:
        raise ValueError(
            "Refusing cross-version reuse: a reused Stage-1 bundle has no compatible "
            f"coarse graph source for {entry['dataset_key']}"
        )
    cache_args = copy.copy(args)
    cache_args.out_dir = os.path.join(os.path.dirname(os.path.dirname(entry["canonical_dir"])), "shared_coarse")
    adjacency, _metadata, path = core.load_or_build_coarse_adjacency(cache_args, hparams, loader)
    return adjacency, path


def load_model(checkpoint, hparams):
    model = MISGLEncoder(hparams, data_name=hparams.data_name).to(torch.device(hparams.device))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def ensure_stage1(entry, args, hparams, loader, train_loader, val_loader):
    os.makedirs(entry["canonical_dir"], exist_ok=True)
    local_checkpoint = os.path.join(entry["canonical_dir"], "stage1_checkpoint.pt")
    local_embeddings = os.path.join(entry["canonical_dir"], "stage1_embeddings.pt")
    source_checkpoint = local_checkpoint if os.path.isfile(local_checkpoint) else entry["stage1"].get("checkpoint")
    source_embeddings = local_embeddings if os.path.isfile(local_embeddings) else entry["stage1"].get("embeddings")
    expected_mil = entry["branch"] == "mil"
    checkpoint = None
    model = None

    if source_checkpoint:
        checkpoint = validate_stage1_checkpoint(source_checkpoint, expected_mil, entry["seed"])
        logging.info("Reusing Stage-1 checkpoint: %s", source_checkpoint)
    else:
        reproducibility.set_seed(int(entry["seed"]), cuda_deterministic=False)
        model, best_val = core.train_stage1(hparams, train_loader, val_loader, loader._dataset_raw)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "best_val": best_val,
            "hparams": hparams.values(),
            "protocol": "single_10fold_seed1024",
        }
        torch.save(checkpoint, local_checkpoint)
        source_checkpoint = local_checkpoint
        logging.info("Saved Stage-1 checkpoint: %s", local_checkpoint)

    if source_embeddings:
        raw_payload = torch.load(source_embeddings, map_location="cpu")
        logging.info("Reusing all-bag embedding cache: %s", source_embeddings)
    else:
        if model is None:
            model = load_model(checkpoint, hparams)
        raw_payload = core.export_z_mil(model, hparams, loader)
        normalized = normalize_embedding_payload(raw_payload, entry["branch"])
        torch.save(normalized, local_embeddings)
        source_embeddings = local_embeddings
        logging.info("Saved all-bag embedding cache: %s", local_embeddings)

    payload = normalize_embedding_payload(raw_payload, entry["branch"])
    validate_embedding_coverage(payload, loader)
    predictions_path = os.path.join(entry["canonical_dir"], "stage1_predictions.pt")
    torch.save(
        {key: value for key, value in payload.items() if key not in ("z_mean", "z_mil")},
        predictions_path,
    )
    provenance = {
        "checkpoint": source_checkpoint,
        "embeddings": source_embeddings,
        "predictions": predictions_path,
        "reused_checkpoint": source_checkpoint != local_checkpoint,
        "reused_embeddings": source_embeddings != local_embeddings,
    }
    with open(os.path.join(entry["canonical_dir"], "stage1_provenance.json"), "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False)
    return checkpoint, model, payload, provenance


def ensure_attention(entry, checkpoint, model, hparams, loader, test_loader):
    if entry["branch"] != "mil":
        return None
    raw_path = os.path.join(entry["canonical_dir"], "test_positive_attention.pt")
    metrics_path = os.path.join(entry["canonical_dir"], "attention_metrics.json")
    if os.path.isfile(raw_path) and os.path.isfile(metrics_path):
        logging.info("Reusing offline test attention: %s", raw_path)
        return {"raw": raw_path, "metrics": metrics_path}
    if model is None:
        model = load_model(checkpoint, hparams)
    raw_path, metrics_path = export_test_positive_attention(
        model=model,
        test_loader=test_loader,
        hparams=hparams,
        dataset_raw=loader._dataset_raw,
        output_dir=entry["canonical_dir"],
        data_name=entry["data_name"],
        cv_seed=entry["seed"],
        fold_idx=entry["fold"],
    )
    logging.info("Saved offline test attention: %s", raw_path)
    return {"raw": raw_path, "metrics": metrics_path}


def load_stage2_model(checkpoint, protocol, device):
    input_dim = int(checkpoint["input_dim"])
    hidden_dim = int(protocol.get("hidden_dim") or input_dim)
    model = core.OneLayerCoarseGCN(
        input_dim,
        hidden_dim=hidden_dim,
        dropout=float(protocol["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def ensure_stage2(entry, args, protocol, payload, masks, coarse_adj):
    local_checkpoint = os.path.join(entry["canonical_dir"], "stage2_checkpoint.pt")
    source_checkpoint = local_checkpoint if os.path.isfile(local_checkpoint) else entry["stage2"].get("checkpoint")
    feature_key = "z_mil" if entry["branch"] == "mil" else "z_mean"
    features = payload[feature_key]
    labels = payload["labels"]
    device = torch.device(args.device)
    if source_checkpoint:
        checkpoint = torch.load(source_checkpoint, map_location="cpu")
        model = load_stage2_model(checkpoint, protocol, device)
        adj_norm = core.scipy_to_torch_sparse(core.normalize_for_gcn(coarse_adj), device)
        metrics, logits, z_pos = core.evaluate_stage2(
            model,
            features.to(device=device, dtype=torch.float32),
            labels.to(device=device, dtype=torch.float32),
            {key: value.to(device=device) for key, value in masks.items()},
            adj_norm,
        )
        best_epoch = int(checkpoint.get("best_epoch", -1))
        logging.info("Reusing Stage-2 checkpoint: %s", source_checkpoint)
    else:
        reproducibility.set_seed(int(entry["seed"]), cuda_deterministic=False)
        output = core.train_stage2(features, labels, masks, coarse_adj, args)
        model = output["model"]
        metrics = output["final_metrics"]
        logits = output["logits"]
        z_pos = output["z_pos"]
        best_epoch = int(output["best_epoch"])
        checkpoint = {
            "state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "input_dim": int(features.size(1)),
            "protocol": protocol,
        }
        torch.save(checkpoint, local_checkpoint)
        source_checkpoint = local_checkpoint
        logging.info("Saved Stage-2 checkpoint: %s", local_checkpoint)

    logits = logits.reshape(-1).detach().cpu()
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).to(torch.int64)
    predictions_path = os.path.join(entry["canonical_dir"], "stage2_predictions.pt")
    torch.save(
        {
            "orig_indices": payload["orig_indices"],
            "labels": payload["labels"],
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
            "z_pos": z_pos.detach().cpu(),
        },
        predictions_path,
    )
    return {
        "checkpoint": source_checkpoint,
        "predictions": predictions_path,
        "reused_checkpoint": source_checkpoint != local_checkpoint,
        "best_epoch": best_epoch,
        "metrics": metrics,
    }


def main():
    cli = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest, entry = load_entry(cli.manifest, cli.dataset, cli.branch, cli.fold)
    work = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # On the server scripts live directly in the repository root.
    if os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_coarse_gcn_paper.py")):
        work = os.path.dirname(os.path.abspath(__file__))
    protocol = manifest["stage2_protocol"]
    args = core_args(work, entry, protocol, cli.device)
    reproducibility.set_seed(int(entry["seed"]), cuda_deterministic=False)
    hparams = core.load_hparams(args)
    hparams.cv_split_dir = os.path.dirname(entry["split_manifest"])
    hparams.cv_seed = int(entry["seed"])
    hparams.cv_num_folds = 10
    hparams.cv_val_policy = "adjacent"
    hparams.cv_use_all_samples = True
    args.device = hparams.device

    loader = GraphDataLoaderWrapper(hparams, data_name=hparams.data_name)
    core.sync_hparams_from_loader(hparams, loader)
    split_manifest = loader.load_cv_split_manifest(entry["split_manifest"])
    train_loader, val_loader, test_loader, split_meta = loader.get_cv_loaders_from_manifest(
        split_manifest, int(entry["fold"])
    )
    if entry["stage1"].get("legacy_result"):
        validate_split(entry["stage1"]["legacy_result"], split_meta)
    coarse_adj, coarse_path = load_or_build_coarse(manifest, entry, args, hparams, loader)
    checkpoint, model, payload, stage1 = ensure_stage1(
        entry, args, hparams, loader, train_loader, val_loader
    )
    masks = core.build_masks(payload["orig_indices"], split_meta)
    stage1_metrics = {
        name: core.metric_from_logits(payload["logits"], payload["labels"], mask)
        for name, mask in masks.items()
    }
    attention = ensure_attention(entry, checkpoint, model, hparams, loader, test_loader)
    stage2 = ensure_stage2(entry, args, protocol, payload, masks, coarse_adj)
    result = {
        "protocol": "single_grouped_stratified_10fold_8_1_1",
        "seed": int(entry["seed"]),
        "dataset_key": entry["dataset_key"],
        "data_name": entry["data_name"],
        "branch": entry["branch"],
        "fold_idx": int(entry["fold"]),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": split_meta,
        "coarse_adj": coarse_path,
        "stage1": {
            **stage1,
            "metrics": core.split_summary_metrics(stage1_metrics),
        },
        "stage2": {
            **{key: value for key, value in stage2.items() if key != "metrics"},
            "protocol": protocol,
            "metrics": core.split_summary_metrics(stage2["metrics"]),
        },
        "attention": attention,
    }
    result_path = os.path.join(entry["canonical_dir"], "fold_result.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    logging.info("Completed manifest entry: %s", result_path)


if __name__ == "__main__":
    main()

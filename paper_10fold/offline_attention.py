#!/usr/bin/env python3
"""Test-only MIL attention export and offline statistics; never trains a model."""

import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from MISGL.utils.global_variables import g_key
from paper_attention_metrics import _resolve_node_binary_labels


def _move(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _mean(values):
    return float(np.mean(values)) if values else None


def export_test_positive_attention(
    model,
    test_loader,
    hparams,
    dataset_raw,
    output_dir,
    data_name,
    cv_seed,
    fold_idx,
):
    """Run inference on the held-out test fold and persist only positive bags."""
    device = torch.device(hparams.device)
    model.eval()
    raw_rows = []
    metric_rows = []
    with torch.inference_mode():
        for raw_batch in test_loader:
            batch = _move(raw_batch, device)
            output = model(batch)
            if not isinstance(output, dict) or output.get("branch_b") is None:
                raise RuntimeError("MIL checkpoint did not expose node attention.")
            attention_flat = output["branch_b"]["a"].detach().cpu().reshape(-1)
            logits = output["ypred_A"].detach().cpu().reshape(-1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).to(torch.int64)
            labels = batch[g_key.y].detach().cpu().reshape(-1).to(torch.int64)
            node_counts = batch[g_key.node_num].detach().cpu().reshape(-1).to(torch.int64)
            orig_indices = batch[g_key.orig_graph_idx].detach().cpu().reshape(-1).to(torch.int64)
            subgraph_tensor = batch.get(g_key.subgraph_id)
            if isinstance(subgraph_tensor, torch.Tensor):
                subgraph_ids = subgraph_tensor.detach().cpu().reshape(-1).to(torch.int64)
            else:
                subgraph_ids = torch.full_like(labels, -1)

            cursor = 0
            for local_idx, count_tensor in enumerate(node_counts):
                num_nodes = int(count_tensor.item())
                weights = attention_flat[cursor : cursor + num_nodes].to(torch.float32).clone()
                cursor += num_nodes
                if int(labels[local_idx].item()) != 1 or num_nodes <= 0:
                    continue
                orig_idx = int(orig_indices[local_idx].item())
                instance_np = _resolve_node_binary_labels(dataset_raw, orig_idx, num_nodes)
                instance_labels = torch.as_tensor(instance_np, dtype=torch.int64).clone()
                positive_mask = instance_labels == 1
                positive_count = int(positive_mask.sum().item())
                positive_mass = float(weights[positive_mask].sum().item()) if positive_count else 0.0
                prevalence = float(positive_count) / float(num_nodes)
                enrichment = positive_mass / prevalence if prevalence > 0 else None
                ranking_auc = None
                if 0 < positive_count < num_nodes:
                    ranking_auc = float(roc_auc_score(instance_labels.numpy(), weights.numpy()))
                prediction = int(preds[local_idx].item())
                row = {
                    "orig_graph_idx": orig_idx,
                    "subgraph_id": int(subgraph_ids[local_idx].item()),
                    "fold_idx": int(fold_idx),
                    "bag_label": 1,
                    "logit": float(logits[local_idx].item()),
                    "probability": float(probs[local_idx].item()),
                    "prediction": prediction,
                    "correct": bool(prediction == 1),
                    "num_nodes": num_nodes,
                    "positive_instance_count": positive_count,
                    "positive_instance_prevalence": prevalence,
                    "positive_attention_mass": positive_mass,
                    "positive_attention_enrichment": enrichment,
                    "attention_ranking_auc": ranking_auc,
                }
                metric_rows.append(row)
                raw_rows.append(
                    {
                        **row,
                        "node_attention": weights,
                        "instance_labels": instance_labels,
                    }
                )

    enrichments = [r["positive_attention_enrichment"] for r in metric_rows if r["positive_attention_enrichment"] is not None]
    aucs = [r["attention_ranking_auc"] for r in metric_rows if r["attention_ranking_auc"] is not None]
    correct = [r["positive_attention_enrichment"] for r in metric_rows if r["correct"] and r["positive_attention_enrichment"] is not None]
    wrong = [r["positive_attention_enrichment"] for r in metric_rows if not r["correct"] and r["positive_attention_enrichment"] is not None]
    payload = {
        "data_name": data_name,
        "cv_seed": int(cv_seed),
        "fold_idx": int(fold_idx),
        "scope": "positive_bags_in_held_out_test_fold_only",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "positive_bag_count": len(metric_rows),
            "positive_attention_enrichment_mean": _mean(enrichments),
            "attention_ranking_auc_mean": _mean(aucs),
            "correct_positive_bag_count": sum(int(r["correct"]) for r in metric_rows),
            "incorrect_positive_bag_count": sum(int(not r["correct"]) for r in metric_rows),
            "correct_enrichment_mean": _mean(correct),
            "incorrect_enrichment_mean": _mean(wrong),
        },
        "positive_bags": metric_rows,
    }
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "test_positive_attention.pt")
    metrics_path = os.path.join(output_dir, "attention_metrics.json")
    torch.save(
        {
            "data_name": data_name,
            "cv_seed": int(cv_seed),
            "fold_idx": int(fold_idx),
            "scope": payload["scope"],
            "positive_bags": raw_rows,
        },
        raw_path,
    )
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return raw_path, metrics_path


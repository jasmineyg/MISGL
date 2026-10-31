"""Recompute predictions and held-out metrics from a public-release run."""

import argparse
import json
from pathlib import Path

import numpy as np

from MISGL.experiment import (
    context_graph_tensor,
    fold_is_complete,
    load_run,
    predict_checkpoint,
    write_json,
)
from MISGL.trainer import evaluate_splits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--data-dir", help="Processed-data directory if data have moved"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", help="Default: <run-dir>/evaluation.json")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    record, config, device, bundle = load_run(run_dir, args.data_dir, args.device)
    adjacency = context_graph_tensor(bundle, config, device)
    results = []
    for fold in range(10):
        folder = run_dir / ("fold_" + str(fold))
        if not fold_is_complete(folder, record["fingerprint"]):
            continue
        internal_logits, final, labels, indices = predict_checkpoint(
            folder, config, device, bundle, adjacency
        )
        with np.load(folder / "predictions.npz") as saved:
            if not np.array_equal(
                indices.numpy(), saved["sample_indices"]
            ) or not np.array_equal(labels.numpy(), saved["labels"]):
                raise ValueError("Inference sample order or labels changed")
            error = float(np.max(np.abs(final.numpy() - saved["final_logits"])))
            changes = int(np.sum((final.numpy() > 0) != (saved["final_logits"] > 0)))
        results.append(
            {
                "fold": fold,
                "max_logit_error": error,
                "changed_predictions": changes,
                "internal": evaluate_splits(
                    internal_logits, labels, bundle.split(fold)
                ),
                "final": evaluate_splits(final, labels, bundle.split(fold)),
            }
        )
    if not results:
        raise ValueError("No verified completed folds were found")
    write_json(
        args.output or run_dir / "evaluation.json",
        {"folds": results, "complete_10fold": len(results) == 10},
    )
    print(
        json.dumps(
            {
                "evaluated_folds": len(results),
                "max_logit_error": max(r["max_logit_error"] for r in results),
                "changed_predictions": sum(r["changed_predictions"] for r in results),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

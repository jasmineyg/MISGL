"""Numerical invariants and an isolated end-to-end checkpoint round trip."""

import json
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from evaluate import main as evaluate
from MISGL.config import default_config
from MISGL.data import DatasetBundle
from MISGL.experiment import benchmark_catalog, parse_args, run_experiment, sha256
from MISGL.context_head import build_context_adjacency, normalize_context_adjacency
from MISGL.internal_head import InternalHead, compute_structural_statistics
from data_construction.synthetic_data import SYNTHETIC_PRESETS, build_synthetic_dataset


class NumericalTests(unittest.TestCase):
    def test_structure_is_invariant_to_node_reindexing(self):
        adjacency = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
        order = np.array([2, 0, 3, 1])
        original = compute_structural_statistics(adjacency)
        permuted = compute_structural_statistics(adjacency[order][:, order])
        np.testing.assert_allclose(permuted, original[order], atol=2e-6)

    def test_gated_pooling_is_permutation_invariant(self):
        torch.manual_seed(3)
        head = InternalHead(8, 4, 32, 8, 0.1).eval()
        nodes, structure = torch.randn(5, 8), torch.rand(5, 9)
        subgraph_index = torch.tensor([0, 0, 0, 1, 1])
        order = torch.tensor([4, 2, 0, 3, 1])
        a = head(nodes, structure, subgraph_index)
        b = head(nodes[order], structure[order], subgraph_index[order])
        torch.testing.assert_close(a.z_int, b.z_int)
        torch.testing.assert_close(a.attention_weights[order], b.attention_weights)

    def test_coarse_weights_and_self_only_control(self):
        a = np.array(
            [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.float32
        )
        s = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
        coarse = build_context_adjacency(sp.csr_matrix(a), s, [0, 1], 1)
        np.testing.assert_array_equal(coarse.toarray(), [[0, 2], [2, 0]])
        np.testing.assert_allclose(
            normalize_context_adjacency(coarse).toarray(),
            [[1 / 3, 2 / 3], [2 / 3, 1 / 3]],
        )
        np.testing.assert_array_equal(
            normalize_context_adjacency(np.zeros((2, 2))).toarray(), np.eye(2)
        )

    def test_generated_folds_are_deterministic_and_keep_groups_together(self):
        labels = np.repeat(np.tile([0, 1], 20), 2)
        groups = np.repeat(np.arange(40), 2)
        first = DatasetBundle._build_grouped_folds(labels, groups, 10, 1024)
        second = DatasetBundle._build_grouped_folds(labels, groups, 10, 1024)
        seen = set()
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(set(labels[a]), {0, 1})
            current = set(groups[a])
            self.assertFalse(current & seen)
            seen.update(current)
        np.testing.assert_array_equal(
            np.sort(np.concatenate(first)), np.arange(len(labels))
        )


class RoundTripTests(unittest.TestCase):
    def test_train_resume_evaluate_and_reject_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            entry = benchmark_catalog()["syn3"]
            options = SimpleNamespace(
                preset=entry["dataset_name"],
                num_subgraphs=40,
                nodes_per_subgraph=16,
                feature_dim=4,
                cross_edges_per_pair=2,
            )
            data = build_synthetic_dataset(
                options, SYNTHETIC_PRESETS[entry["dataset_name"]]
            )
            data_path = directory / (entry["dataset_name"] + "_processed.pkl")
            with data_path.open("wb") as handle:
                pickle.dump(data, handle)
            config = default_config()
            config = replace(
                config,
                device="cpu",
                data_dir=str(directory),
                model=replace(config.model, node_dim=8, classifier_dim=4),
                internal_head=replace(
                    config.internal_head, attention_dim=4, structure_gate_dim=8
                ),
                context_head=replace(config.context_head, hidden_dim=8),
            )
            args = parse_args(
                [
                    "--benchmark",
                    "syn3",
                    "--device",
                    "cpu",
                    "--data-dir",
                    str(directory),
                    "--output-dir",
                    str(directory / "output"),
                    "--smoke",
                    "--folds",
                    "0",
                ]
            )
            configuration_patch = patch(
                "MISGL.experiment.default_config", return_value=config
            )
            configuration_patch.start()
            self.addCleanup(configuration_patch.stop)
            for variant in ["nasgl", "context", "internal", "gat"]:
                args.variant = variant
                output = run_experiment(args)
                summary = json.loads((output / "summary.json").read_text())
                self.assertFalse(summary["complete_10fold"])
                self.assertNotIn("metrics", summary)
                checkpoint_hash = sha256(output / "fold_0/internal_encoder.pt")
                run_experiment(args)
                self.assertEqual(
                    sha256(output / "fold_0/internal_encoder.pt"), checkpoint_hash
                )
                evaluate(["--run-dir", str(output), "--device", "cpu"])
                result = json.loads((output / "evaluation.json").read_text())
                self.assertEqual(result["folds"][0]["changed_predictions"], 0)
                self.assertLess(result["folds"][0]["max_logit_error"], 1e-6)
            with data_path.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(ValueError):
                run_experiment(args)


if __name__ == "__main__":
    # Only the tiny temporary integration fixture uses a single thread.
    torch.set_num_threads(1)
    unittest.main()

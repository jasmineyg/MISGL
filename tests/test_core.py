import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

import train
from MISGL.config import load_config
from MISGL.data import DatasetBundle
from MISGL.keys import BAG_INDEX, EDGE_INDEX, LABEL, SAMPLE_INDEX, STRUCTURE, X
from MISGL.losses import binary_loss, model_loss
from MISGL.metrics import binary_metrics
from MISGL.models.encoder import MISGLModel, ModelOutput
from MISGL.models.pos_head import POSHead
from MISGL.position_graph import build_position_adjacency, row_normalize, to_torch_sparse
from MISGL.trainer import run_dataset


ROOT = Path(__file__).resolve().parents[1]


def compact_config(**overrides):
    config = load_config(str(ROOT / "config" / "train.yml"))
    config = replace(
        config,
        device="cpu",
        cuda_device=None,
        folds=3,
        model=replace(
            config.model,
            encoder_dim=4,
            classifier_dim=4,
            dropout=0.0,
            gat_heads=2,
            gat_attention_dropout=0.0,
            gat_feature_dropout=0.0,
        ),
        training=replace(
            config.training,
            batch_size=4,
            epochs=1,
            patience=1,
            max_grad_norm=1.0,
        ),
        mil_head=replace(
            config.mil_head,
            attention_dim=4,
            attention_loss_weight=0.0,
        ),
        pos_head=replace(
            config.pos_head,
            top_k=2,
            hidden_dim=4,
            dropout=0.0,
            epochs=1,
            patience=1,
        ),
    )
    return replace(config, **overrides)


def graph_batch():
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4],
            [0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 4],
        ],
        dtype=torch.long,
    )
    return {
        X: features,
        EDGE_INDEX: edge_index,
        BAG_INDEX: torch.tensor([0, 0, 1, 1, 1]),
        STRUCTURE: torch.zeros(5, 7),
        LABEL: torch.tensor([0, 1]),
        SAMPLE_INDEX: torch.tensor([0, 1]),
    }


def synthetic_payload():
    subgraph_ids = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]
    subgraphs = []
    labels = []
    for index, subgraph_id in enumerate(subgraph_ids):
        node_count = 2 + index % 3
        graph = nx.path_graph(node_count)
        label = index % 2
        graph.graph.update(label=label, subgraph_id=subgraph_id)
        for node in graph.nodes:
            graph.nodes[node]["features"] = np.asarray(
                [float(label), float(node), float(index % 3), 1.0],
                dtype=np.float32,
            )
        subgraphs.append(graph)
        labels.append(label)

    original = nx.cycle_graph(36)
    assignment = sp.lil_matrix((36, 15), dtype=np.float32)
    for position, subgraph_id in enumerate(subgraph_ids):
        assignment[position * 3 : position * 3 + 3, subgraph_id] = 1.0
    return {
        "subgraph_structures": subgraphs,
        "subgraph_labels": labels,
        "train_test_split": {
            "train_indices": list(range(8)),
            "test_indices": list(range(8, 12)),
        },
        "original_graph": original,
        "assignment_matrix": assignment.tocsr(),
    }


class ConfigurationTest(unittest.TestCase):
    def test_head_switches_are_explicit(self):
        config = load_config(str(ROOT / "config" / "train.yml"))
        self.assertTrue(config.mil_head.enabled)
        self.assertTrue(config.pos_head.enabled)

        args = train.parse_args(["--no-mil-head", "--no-pos-head"])
        self.assertFalse(args.mil_head)
        self.assertFalse(args.pos_head)
        with self.assertRaisesRegex(ValueError, "requires mil_head"):
            replace(config, mil_head=replace(config.mil_head, enabled=False))


class ModelTest(unittest.TestCase):
    def test_mil_and_mean_modes_have_one_output_interface(self):
        batch = graph_batch()
        mil_output = MISGLModel(compact_config(), input_dim=4)(batch)
        self.assertEqual(mil_output.logits.shape, (2,))
        self.assertEqual(mil_output.embedding.shape, (2, 4))
        self.assertEqual(mil_output.attention.shape, (5,))
        for bag in range(2):
            torch.testing.assert_close(
                mil_output.attention[mil_output.bag_index == bag].sum(),
                torch.tensor(1.0),
            )

        config = compact_config()
        config = replace(
            config,
            mil_head=replace(config.mil_head, enabled=False),
            pos_head=replace(config.pos_head, enabled=False),
        )
        mean_output = MISGLModel(config, input_dim=4)(batch)
        self.assertEqual(mean_output.embedding.shape, (2, 4))
        self.assertIsNone(mean_output.attention)
        self.assertIsNone(mean_output.bag_index)

    def test_pos_head_shapes(self):
        embedding = torch.randn(4, 4)
        adjacency = to_torch_sparse(row_normalize(sp.eye(4, format="csr")), "cpu")
        logits = POSHead(embedding_dim=4, hidden_dim=4, dropout=0.0)(
            embedding, adjacency
        )
        self.assertEqual(logits.shape, (4,))


class LossAndMetricTest(unittest.TestCase):
    def test_focal_gamma_zero_matches_bce(self):
        logits = torch.tensor([-2.0, -0.2, 0.4, 2.5])
        targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
        bce_config = compact_config()
        focal_config = replace(
            bce_config,
            training=replace(bce_config.training, loss="focal", focal_gamma=0.0),
        )
        torch.testing.assert_close(
            binary_loss(logits, targets, focal_config),
            binary_loss(logits, targets, bce_config),
        )

    def test_enabled_attention_loss_requires_attention(self):
        config = compact_config()
        config = replace(
            config,
            mil_head=replace(config.mil_head, attention_loss_weight=0.2),
        )
        output = ModelOutput(
            logits=torch.zeros(2),
            embedding=torch.zeros(2, 4),
            attention=None,
            bag_index=None,
        )
        with self.assertRaisesRegex(ValueError, "returned no attention"):
            model_loss(output, torch.tensor([0, 1]), config)

    def test_binary_metrics(self):
        result = binary_metrics(
            torch.tensor([-2.0, 2.0, -0.5, 0.5]),
            torch.tensor([0, 1, 1, 0]),
        )
        self.assertEqual(set(result), {
            "acc", "precision", "recall", "f1", "balanced_acc",
            "roc_auc", "pr_auc", "tn", "fp", "fn", "tp",
        })


class DataAndTrainingTest(unittest.TestCase):
    def test_non_contiguous_subgraph_ids_and_end_to_end_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (root / "synthetic_processed.pkl").open("wb") as handle:
                pickle.dump(synthetic_payload(), handle)

            config = compact_config(
                datasets=("synthetic",),
                data_dir=str(root),
                output_dir=str(root / "results"),
            )
            bundle = DatasetBundle.load(
                config, "synthetic", require_position_graph=True
            )
            self.assertEqual(bundle.assignment_matrix.shape[1], 15)
            self.assertEqual(bundle.subgraph_ids[-1], 14)

            first_batch = next(iter(bundle.loaders(0, batch_size=4)["train"]))
            self.assertEqual(first_batch[STRUCTURE].shape, (first_batch[X].shape[0], 7))
            self.assertEqual(first_batch[EDGE_INDEX].shape[0], 2)

            adjacency = build_position_adjacency(
                bundle.original_graph,
                bundle.assignment_matrix,
                bundle.subgraph_ids,
                top_k=2,
            )
            self.assertEqual(adjacency.shape, (12, 12))
            self.assertLessEqual(max(np.diff(adjacency.indptr)), 2)

            result = run_dataset(config, "synthetic", torch.device("cpu"))
            self.assertEqual(len(result["folds"]), 3)
            self.assertTrue((root / "results" / "synthetic" / "misgl_mil_pos" / "metrics.pt").exists())


if __name__ == "__main__":
    unittest.main()

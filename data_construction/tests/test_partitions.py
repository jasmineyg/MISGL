"""Regression tests for loader contracts and validated partition algorithms."""

import tempfile
import unittest
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp

from data_construction.dataset_export import (
    calculate_label_profile,
    calculate_partition_statistics,
    export_partitioned_dataset,
)
from data_construction.graph_partition import (
    PartitionResult,
    balanced_multi_source_bfs,
    normalize_method,
    partition_graph,
    perturb_within_labels,
)
from data_construction.real_data import GraphDataset, load_real_graph


class BalancedMultiSourceBFSTests(unittest.TestCase):
    def setUp(self):
        graph = nx.Graph()
        graph.add_nodes_from(range(13))
        graph.add_edges_from((node, node + 1) for node in range(7))
        graph.add_edges_from([(8, 9), (9, 10)])
        # Nodes 11 and 12 are globally isolated.
        self.graph = graph
        self.nodes = list(range(13))

    def test_balanced_complete_and_reproducible_on_disconnected_graph(self):
        first, diagnostics = balanced_multi_source_bfs(
            self.graph, self.nodes, num_parts=4, random_seed=42
        )
        second, _ = balanced_multi_source_bfs(
            self.graph, self.nodes, num_parts=4, random_seed=42
        )
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.any(first < 0))
        sizes = np.bincount(first, minlength=4)
        self.assertEqual(int(sizes.max() - sizes.min()), 1)
        self.assertEqual(int(sizes.sum()), len(self.nodes))
        self.assertGreaterEqual(diagnostics["reseed_count"], 1)

    def test_partition_assignment_is_independent_of_features_and_labels(self):
        dataset_a = GraphDataset(
            name="fixture",
            graph=self.graph,
            nodes=self.nodes,
            node_features=np.zeros((13, 2), dtype=np.float32),
            node_categories={node: node % 2 for node in self.nodes},
        ).validate()
        dataset_b = GraphDataset(
            name="fixture",
            graph=self.graph,
            nodes=self.nodes,
            node_features=np.full((13, 7), 99, dtype=np.float32),
            node_categories={node: (node + 1) % 3 for node in self.nodes},
        ).validate()
        config = {"num_partitions": 4, "seed": 7}
        result_a = partition_graph(dataset_a, "bfs", config)
        result_b = partition_graph(dataset_b, "msbfs", config)
        np.testing.assert_array_equal(result_a.assignments, result_b.assignments)
        self.assertFalse(result_a.metadata["uses_node_features"])
        self.assertFalse(result_a.metadata["uses_node_labels"])

    def test_aliases_and_required_statistics(self):
        self.assertEqual(normalize_method("bfs"), "bfs")
        self.assertEqual(normalize_method("msbfs"), "bfs")
        assignment, _ = balanced_multi_source_bfs(
            self.graph, self.nodes, num_parts=4, random_seed=42
        )
        dataset = GraphDataset(
            name="fixture",
            graph=self.graph,
            nodes=self.nodes,
            node_features=np.zeros((13, 1), dtype=np.float32),
            node_categories={node: node % 2 for node in self.nodes},
        ).validate()
        stats = calculate_partition_statistics(dataset, assignment, 4)
        expected = {
            "requested_num_partitions",
            "num_subgraphs",
            "size_min",
            "size_max",
            "size_mean",
            "size_median",
            "size_std",
            "edgeless_subgraphs",
            "internal_isolated_nodes",
            "original_isolated_nodes",
            "cross_subgraph_edge_ratio",
            "coarse_graph_nodes",
            "coarse_graph_edges",
            "isolated_coarse_subgraphs",
        }
        self.assertTrue(expected.issubset(stats))
        self.assertEqual(stats["original_isolated_nodes"], 2)


class AlgorithmRegressionTests(unittest.TestCase):
    def setUp(self):
        graph = nx.cycle_graph(8)
        self.dataset = GraphDataset(
            name="fixture",
            graph=graph,
            nodes=list(range(8)),
            node_features=np.eye(8, dtype=np.float32),
            node_categories=dict(enumerate([1, 0, 1, 0, 1, 0, 1, 0])),
        ).validate()
        self.reference = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)

    def test_perturb_matches_original_reference(self):
        actual, diagnostics = perturb_within_labels(
            self.reference,
            self.dataset,
            target_category=1,
            perturb_ratio=0.5,
            random_seed=542,
        )
        expected = np.asarray([0, 0, 2, 3, 1, 2, 3, 1], dtype=np.int64)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(diagnostics["actual_swapped_nodes"], 4)
        self.assertEqual(diagnostics["rejected_same_partition_pairs"], 0)

    def test_perturb_preserves_metis_label_profile(self):
        reference_profile = calculate_label_profile(
            self.reference, self.dataset, target_category=1, num_parts=4
        )
        perturb_assignment, _ = perturb_within_labels(
            self.reference, self.dataset, 1, 0.5, 542
        )
        for assignment in (perturb_assignment,):
            profile = calculate_label_profile(
                assignment, self.dataset, target_category=1, num_parts=4
            )
            self.assertEqual(
                profile["positive_counts_by_part"],
                reference_profile["positive_counts_by_part"],
            )
            np.testing.assert_array_equal(
                np.bincount(assignment, minlength=4),
                np.bincount(self.reference, minlength=4),
            )


class DatasetAdapterTests(unittest.TestCase):
    def test_ogbn_arxiv_and_products_share_the_validated_loader(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fixture = Path(temporary.name)
        raw = fixture / "raw"
        raw.mkdir()
        import pandas as pd

        pd.DataFrame([[0, 1], [1, 2], [2, 3], [3, 0]]).to_csv(
            raw / "edge.csv.gz", header=False, index=False
        )
        pd.DataFrame(np.eye(4, 3, dtype=np.float32)).to_csv(
            raw / "node-feat.csv.gz", header=False, index=False
        )
        pd.DataFrame([0, 1, 1, 0]).to_csv(
            raw / "node-label.csv.gz", header=False, index=False
        )
        for name in ("ogbn-arxiv", "ogbn-products"):
            dataset = load_real_graph({"dataset": name, "dataset_path": str(fixture)})
            self.assertEqual(dataset.graph.number_of_nodes(), 4)
            self.assertEqual(dataset.graph.number_of_edges(), 4)
            self.assertEqual(dataset.node_features.shape, (4, 3))
            self.assertEqual(
                [dataset.node_categories[i] for i in range(4)], [0, 1, 1, 0]
            )

    def test_reddit_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw = Path(temp_dir) / "raw"
            raw.mkdir()
            features = np.eye(4, dtype=np.float32)
            labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
            np.savez(raw / "reddit_data.npz", feature=features, label=labels)
            adjacency = sp.csr_matrix(
                np.asarray(
                    [
                        [0, 1, 0, 1],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [1, 0, 1, 0],
                    ],
                    dtype=np.int8,
                )
            )
            sp.save_npz(raw / "reddit_graph.npz", adjacency)
            dataset = load_real_graph({"dataset": "reddit", "dataset_path": temp_dir})
            self.assertEqual(dataset.graph.number_of_edges(), 4)
            self.assertEqual(dataset.node_features.shape, (4, 4))

    def test_igb_tiny_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "processed"
            paper = root / "paper"
            edges_dir = root / "paper__cites__paper"
            paper.mkdir(parents=True)
            edges_dir.mkdir(parents=True)
            np.save(paper / "node_feat.npy", np.eye(4, dtype=np.float32))
            np.save(
                paper / "node_label_19.npy",
                np.asarray([0, 1, 1, 0], dtype=np.float32),
            )
            np.save(
                edges_dir / "edge_index.npy",
                np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64),
            )
            dataset = load_real_graph(
                {
                    "dataset": "igb-tiny",
                    "dataset_path": temp_dir,
                    "label_file": "node_label_19.npy",
                }
            )
            self.assertEqual(dataset.graph.number_of_nodes(), 4)
            self.assertEqual(dataset.graph.number_of_edges(), 4)
            self.assertEqual(dataset.node_features.shape, (4, 4))
            self.assertEqual(dataset.metadata["label_file"], "node_label_19.npy")


class OutputCompatibilityTests(unittest.TestCase):
    def test_bfs_summary_contains_requested_statistics(self):
        graph = nx.Graph()
        graph.add_nodes_from(range(9))
        graph.add_edges_from([(0, 1), (1, 2), (3, 4), (4, 5), (5, 6)])
        dataset = GraphDataset(
            name="fixture",
            graph=graph,
            nodes=list(range(9)),
            node_features=np.eye(9, dtype=np.float32),
            node_categories={node: node % 2 for node in range(9)},
        ).validate()
        config = {
            "num_partitions": 3,
            "seed": 42,
            "write_processed_pickle": False,
        }
        result = partition_graph(dataset, "bfs", config)
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = export_partitioned_dataset(
                dataset,
                result,
                "bfs",
                {**config, "output_dir": temp_dir},
            )
            import json

            with open(outputs["summary"], encoding="utf-8") as file:
                summary = json.load(file)

        stats = summary["partition_statistics"]
        self.assertEqual(stats["requested_num_partitions"], 3)
        self.assertEqual(stats["num_subgraphs"], 3)
        self.assertEqual(stats["original_isolated_nodes"], 2)
        self.assertEqual(stats["coarse_graph_nodes"], 3)
        self.assertIn("negative_subgraph_ratio", summary["label_profile"])
        self.assertEqual(summary["label_profile_reference"], "self")
        self.assertNotIn("label_profile_vs_metis", summary)

    def test_processed_pickle_preserves_downstream_schema(self):
        dataset = GraphDataset(
            name="fixture",
            graph=nx.complete_graph(4),
            nodes=list(range(4)),
            node_features=np.eye(4, dtype=np.float32),
            node_categories=dict(enumerate([0, 1, 1, 0])),
            metadata={"dataset_type": "fixture"},
        ).validate()
        assignment = np.asarray([0, 0, 1, 1], dtype=np.int64)
        result = PartitionResult(
            assignments=assignment,
            target_category=0,
            reference_assignments=assignment.copy(),
            metadata={"partition_method": "metis"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = export_partitioned_dataset(
                dataset,
                result,
                "metis",
                {
                    "num_partitions": 2,
                    "seed": 42,
                    "output_dir": temp_dir,
                    "write_processed_pickle": True,
                },
            )
            with open(outputs["processed_pickle"], "rb") as file:
                processed = pickle.load(file)
        expected_keys = {
            "subgraph_labels",
            "original_graph",
            "node_features",
            "subgraph_structures",
            "assignment_matrix",
            "subgraph_assignment",
            "dataset_metadata",
            "node_categories",
            "train_test_split",
            "node_binary_labels",
            "feature_dimension",
            "subgraph_node_counts",
        }
        self.assertEqual(set(processed), expected_keys)
        np.testing.assert_array_equal(processed["subgraph_assignment"], assignment)
        self.assertEqual(processed["train_test_split"]["train_ratio"], 0.8)


if __name__ == "__main__":
    unittest.main()

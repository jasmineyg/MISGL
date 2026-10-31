"""Check the paper presets and raw-file-to-processed-file interface."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from data_construction.dataset_settings import benchmark_catalog
from data_construction.real_data import paper_dataset_config, build_real_dataset


class ConstructionPipelineTests(unittest.TestCase):
    def test_all_real_settings_align_with_training_catalog(self):
        catalog = benchmark_catalog()
        real_keys = [key for key in catalog if not key.startswith("syn")]
        self.assertEqual(len(real_keys), 12)
        for key in real_keys:
            config = paper_dataset_config(key, "raw_data", "data")
            self.assertEqual(config["output_name"], catalog[key]["dataset_name"])
            directory = Path(catalog[key]["directory"])
            dataset, method = key.split("__")
            self.assertEqual(
                directory.parent.name, "igb-tiny" if dataset == "igb_tiny" else dataset
            )
            self.assertEqual(
                directory.name,
                "metis_perturb" if method == "metis_perturb_50" else method,
            )
            self.assertEqual(config["seed"], 42)
            self.assertEqual(
                config["num_partitions"],
                {
                    "igb_tiny": 1000,
                    "ogbn_arxiv": 2000,
                    "ogbn_products": 20000,
                    "reddit": 4000,
                }[key.split("__")[0]],
            )
            if key.endswith("perturb_50"):
                self.assertEqual(config["perturb_ratio"], 0.5)
                self.assertEqual(
                    Path(config["reference_partition"]).parent.name, "metis"
                )

    def test_raw_to_metis_perturb_and_bfs_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            count = 24
            features = np.eye(count, dtype=np.float32)
            labels = np.arange(count) % 3
            np.savez(raw / "reddit_data.npz", feature=features, label=labels)
            # Four dense communities with a few connecting edges.
            adjacency = np.zeros((count, count), dtype=np.int8)
            for left in range(count):
                for right in range(count):
                    if left != right and left // 6 == right // 6:
                        adjacency[left, right] = 1
            for left in (5, 11, 17):
                adjacency[left, left + 1] = adjacency[left + 1, left] = 1
            sp.save_npz(raw / "reddit_graph.npz", sp.csr_matrix(adjacency))
            reference = None
            for method in ("metis", "metis_perturb_50", "bfs"):
                config = paper_dataset_config(
                    "reddit__" + method,
                    root,
                    root / "data",
                    reference if method == "metis_perturb_50" else None,
                )
                config["num_partitions"] = (
                    4  # A tiny fixture, never a benchmark build_real_dataset.
                )
                outputs = build_real_dataset(config)
                with Path(outputs["processed_pickle"]).open("rb") as handle:
                    processed = pickle.load(handle)
                self.assertEqual(len(processed["subgraph_labels"]), 4)
                np.testing.assert_array_equal(processed["node_features"], features)
                np.testing.assert_array_equal(
                    np.asarray(processed["assignment_matrix"].sum(axis=1)).ravel(),
                    np.ones(count),
                )
                for bag, label in zip(
                    processed["subgraph_structures"], processed["subgraph_labels"]
                ):
                    target = processed["dataset_metadata"]["target_category"]
                    self.assertEqual(
                        label, int(any(labels[node] == target for node in bag))
                    )
                if method == "metis":
                    reference = outputs["partition"]
                    reference_assignments = processed["subgraph_assignment"]
                if method == "metis_perturb_50":
                    self.assertTrue(
                        processed["dataset_metadata"]["label_profile_vs_metis"][
                            "positive_count_profile_exact_match"
                        ]
                    )
                    np.testing.assert_array_equal(
                        np.bincount(processed["subgraph_assignment"], minlength=4),
                        np.bincount(reference_assignments, minlength=4),
                    )
                with self.assertRaises(FileExistsError):
                    build_real_dataset(config)


if __name__ == "__main__":
    unittest.main()

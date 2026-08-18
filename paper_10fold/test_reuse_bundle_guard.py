#!/usr/bin/env python3
"""Regression tests for cross-version Stage-1/coarse-graph reuse."""

import os
import sys
import unittest
from types import ModuleType
from types import SimpleNamespace
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
LEGACY_SCRIPT_DIR = os.path.join(REPO_ROOT, "paper_5x10")
for path in (HERE, LEGACY_SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# The local lightweight checkout does not contain the server-only legacy training
# stack. Install narrow import stubs so this unit test can still exercise the
# bundle guard; the server run imports and tests the real modules.
if not os.path.isfile(os.path.join(REPO_ROOT, "MISGL", "bin", "train_eval.py")):
    core_stub = ModuleType("run_coarse_gcn_paper")
    core_stub.active_subgraph_ids = lambda *_args, **_kwargs: []
    core_stub.cache_metadata_matches = lambda *_args, **_kwargs: False
    core_stub.load_or_build_coarse_adjacency = lambda *_args, **_kwargs: (None, {}, None)
    sys.modules["run_coarse_gcn_paper"] = core_stub

    encoder_stub = ModuleType("MISGL.models.encoder")
    encoder_stub.MISGLEncoder = object
    sys.modules["MISGL.models.encoder"] = encoder_stub

    coarse_stub = ModuleType("MISGL.utils.coarse_graph")
    coarse_stub.load_coarse_adjacency = lambda *_args, **_kwargs: (None, {})
    reproducibility_stub = ModuleType("MISGL.utils.reproducibility")
    utils_stub = ModuleType("MISGL.utils")
    utils_stub.coarse_graph = coarse_stub
    utils_stub.reproducibility = reproducibility_stub
    sys.modules["MISGL.utils"] = utils_stub
    sys.modules["MISGL.utils.coarse_graph"] = coarse_stub
    sys.modules["MISGL.utils.reproducibility"] = reproducibility_stub

    load_data_stub = ModuleType("MISGL.utils.load_data")
    load_data_stub.GraphDataLoaderWrapper = object
    sys.modules["MISGL.utils.load_data"] = load_data_stub

    attention_stub = ModuleType("offline_attention")
    attention_stub.export_test_positive_attention = lambda *_args, **_kwargs: None
    sys.modules["offline_attention"] = attention_stub

import build_reuse_manifest as manifest_builder
import run_manifest_entry as runner


class ReuseBundleRegressionTest(unittest.TestCase):
    def test_syn2_uses_the_mixed_august_10_bundle(self):
        path = manifest_builder.legacy_run_dir("syn2", "mil")
        self.assertIn("mil_consistent_synthetic2_mixed_20260810", path)
        self.assertNotIn("mil_consistent_synthetic_v2_20260810/mil_head", path)

    def test_reused_stage1_cannot_fall_back_to_a_different_coarse_graph(self):
        entry = {
            "dataset_key": "syn2",
            "canonical_dir": "/tmp/paper/mil/syn2/fold_0",
            "stage1": {"status": "reuse", "legacy_result": "/tmp/legacy.json"},
        }
        args = SimpleNamespace(top_k=16)
        loader = object()
        with (
            mock.patch.object(runner, "find_coarse_source", return_value="/tmp/old_coarse.npz"),
            mock.patch.object(runner.coarse_graph, "load_coarse_adjacency", return_value=(object(), {})),
            mock.patch.object(runner.core, "active_subgraph_ids", return_value=[1, 2, 3]),
            mock.patch.object(runner.core, "cache_metadata_matches", return_value=False),
            mock.patch.object(
                runner.core,
                "load_or_build_coarse_adjacency",
                return_value=(object(), {}, "/tmp/new_coarse.npz"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Refusing cross-version reuse"):
                runner.load_or_build_coarse({}, entry, args, object(), loader)


if __name__ == "__main__":
    unittest.main()

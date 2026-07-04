import unittest
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch

from MISGL.models.encoder import MISGLEncoder
from MISGL.models.subgnn_border import SubGNNBorderRouter
from MISGL.utils.hparam import HParams
from MISGL.utils import hparams_lib
from MISGL.utils.subgnn_border import (
    build_subgnn_border_features,
    dtw_similarity_vector,
    external_degree_sequence,
    sample_anchor_sequences,
    triangular_random_walk,
)


def _with_features(graph, feature_dim=2):
    out = graph.copy()
    for node in out.nodes():
        out.nodes[node]['original_id'] = node
        out.nodes[node]['features'] = np.asarray([float(node), 1.0], dtype=np.float32)[:feature_dim]
    return out


class SubGNNBorderUtilityTest(unittest.TestCase):
    def test_external_degree_sequence_uses_original_minus_internal_degree(self):
        original = nx.path_graph(4)
        subgraph = _with_features(original.subgraph([1, 2]))

        seq = external_degree_sequence(original, subgraph)

        np.testing.assert_allclose(seq, np.asarray([1.0, 1.0], dtype=np.float32))

    def test_triangular_random_walk_prefers_triangle_when_beta_one(self):
        original = nx.Graph()
        original.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3)])
        rng = __import__('random').Random(3)

        walk = triangular_random_walk(
            original,
            anchor_patch_nodes=[0, 1, 2, 3],
            walk_len=4,
            rng=rng,
            rw_beta=1.0,
            inside=True,
            start_node=0,
        )

        self.assertGreaterEqual(len(walk), 2)
        self.assertEqual(walk[0], 0)
        self.assertTrue(all(node in original for node in walk))

    def test_anchor_sampling_is_deterministic(self):
        original = nx.path_graph(8)

        seq1, patches1 = sample_anchor_sequences(original, 4, 3, seed=7)
        seq2, patches2 = sample_anchor_sequences(original, 4, 3, seed=7)

        self.assertEqual(patches1, patches2)
        for left, right in zip(seq1, seq2):
            np.testing.assert_allclose(left, right)

    def test_anchor_sampling_exports_walk_features_for_bilstm(self):
        original = nx.cycle_graph(6)
        lookup = {node: np.asarray([float(node), 1.0], dtype=np.float32) for node in original.nodes()}

        seqs, patches, walk_features = sample_anchor_sequences(
            original,
            3,
            4,
            seed=9,
            n_triangular_walks=2,
            random_walk_len=5,
            feature_lookup=lookup,
            feature_dim=2,
            return_walk_features=True,
        )

        self.assertEqual(len(seqs), 3)
        self.assertEqual(len(patches), 3)
        self.assertEqual(walk_features.shape, (3, 2, 5, 2))

    def test_dtw_similarity_shape_and_range(self):
        sims = dtw_similarity_vector(
            np.asarray([3.0, 1.0], dtype=np.float32),
            [np.asarray([3.0, 1.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32)],
            temperature=1.0,
        )

        self.assertEqual(sims.shape, (2,))
        self.assertAlmostEqual(float(sims[0]), 1.0, places=6)
        self.assertGreater(float(sims[1]), 0.0)
        self.assertLessEqual(float(sims[1]), 1.0)

    def test_shuffle_changes_similarity_only_not_external_count(self):
        original = nx.Graph()
        original.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 7)])
        subgraphs = [
            _with_features(original.subgraph([0, 1])),
            _with_features(original.subgraph([2, 3, 4])),
            _with_features(original.subgraph([5, 6, 7])),
        ]
        raw = {
            'original_graph': original,
            'subgraph_structures': subgraphs,
            'feature_dimension': 2,
            'dataset_metadata': {'feature_dim': 2},
        }
        base = SimpleNamespace(
            cv_seed=11,
            subgnn_border={
                'use': True,
                'num_anchors': 4,
                'anchor_size': 3,
                'anchor_seed': 11,
                'n_triangular_walks': 2,
                'random_walk_len': 4,
                'shuffle': False,
                'shuffle_seed': 19,
            },
        )
        shuffled = SimpleNamespace(
            cv_seed=11,
            subgnn_border={
                'use': True,
                'num_anchors': 4,
                'anchor_size': 3,
                'anchor_seed': 11,
                'n_triangular_walks': 2,
                'random_walk_len': 4,
                'shuffle': True,
                'shuffle_seed': 19,
            },
        )

        real_info = build_subgnn_border_features(raw, base, data_name='toy')
        shuffled_info = build_subgnn_border_features(raw, shuffled, data_name='toy')

        np.testing.assert_allclose(real_info['external_counts'], shuffled_info['external_counts'])
        self.assertEqual(real_info['features'].shape, shuffled_info['features'].shape)
        self.assertEqual(real_info['anchor_walk_features'].shape, (4, 2, 4, 2))


class SubGNNBorderModelTest(unittest.TestCase):
    def test_router_preserves_z_mil_shape_and_exports_diagnostics(self):
        router = SubGNNBorderRouter(
            input_dim=5,
            num_anchors=3,
            node_feature_dim=2,
            anchor_embed_dim=4,
            anchor_encoder_hidden_dim=4,
            gate_hidden_dim=6,
        )
        z_mil = torch.randn(2, 5)
        sims = torch.tensor([[1.0, 0.5, 0.1], [0.2, 0.4, 0.8]])
        walk_features = torch.randn(3, 2, 4, 2)

        out = router(z_mil, sims, walk_features)

        self.assertEqual(out['z_fused'].shape, z_mil.shape)
        self.assertEqual(out['z_border'].shape, z_mil.shape)
        self.assertEqual(out['border_anchor_embeds'].shape, (3, 4))
        self.assertEqual(out['border_anchor_entropy'].shape, (2,))
        self.assertEqual(out['border_residual_ratio'].shape, (2,))

    def test_disabled_border_keeps_branch_b_classifier_input_dim(self):
        hp = HParams(
            device='cpu',
            channel_list=[3, 4, 3, 2, 1],
            dropout=0.1,
            leaky_relu_alpha=0.2,
            gat_heads=1,
            gat_attn_dropout=0.0,
            gat_feat_dropout=0.0,
            gat_alpha=0.2,
            gat_concat=True,
            gat_residual=True,
            branch_b={'use': True, 'use_structural_features': False},
            subgnn_border={'use': False},
            use_coarse_graph=False,
        )
        hparams_lib.apply_defaults(hp)

        model = MISGLEncoder(hp, data_name='toy')

        self.assertIsNone(model.subgnn_border_router)
        self.assertEqual(model.classifier[0].in_features, model.branch_b_head.output_dim)


if __name__ == '__main__':
    unittest.main()

"""Regression tests for the full nine-feature, single-layer structural encoder."""
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch import nn
from MISGL.config import load_config
from MISGL.models.mil_head import MILHead
from MISGL.structural_features import FEATURE_NAMES, compute_structural_features

class StructureSelectionTest(unittest.TestCase):
    def test_path_topology_and_input_preservation(self):
        graph = np.array([[7, 2, 0], [0, 3, 5], [0, 0, 9]])
        original = graph.copy()
        values = compute_structural_features(graph)
        np.testing.assert_array_equal(graph, original)
        np.testing.assert_allclose(values[:, 0], [.5, 1, .5])
        np.testing.assert_allclose(values[:, 1], [1, .5, 1])
        np.testing.assert_allclose(values[:, 2], [.5, 0, .5])
        np.testing.assert_array_equal(values[:, 4:6], 0)
        np.testing.assert_allclose(values[:, 6], [.5, .5, .5])
        np.testing.assert_allclose(values, compute_structural_features(np.array([[0,1,0],[1,0,1],[0,1,0]])))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(((values >= 0) & (values <= 1)).all())

    def test_triangle_and_singleton_statistics(self):
        values = compute_structural_features(np.ones((3, 3)))
        np.testing.assert_allclose(values[:, [0,1,4,5,6,7]], 1)
        np.testing.assert_allclose(values[:, 2], 0)
        np.testing.assert_allclose(values[:, 3], 1/3 + (2/3)*.15/(1+.85/2), rtol=1e-6)
        single = np.zeros((1, 9), dtype=np.float32)
        single[:, [3,7]] = 1
        np.testing.assert_allclose(compute_structural_features(np.ones((1,1))), single)
        self.assertEqual(compute_structural_features(np.zeros((0,0))).shape, (0,9))
        with self.assertRaisesRegex(ValueError, 'square'):
            compute_structural_features(np.zeros((2,3)))
        with self.assertRaisesRegex(ValueError, 'finite'):
            compute_structural_features(np.array([[np.nan]]))

    def test_full_features_dimensions_and_gradients(self):
        torch.manual_seed(1024)
        config = load_config(str(Path(__file__).resolve().parents[1] / 'config/train.yml'))
        self.assertEqual(config.mil_head.structure_dim, 32)
        self.assertEqual(len(FEATURE_NAMES), 9)
        head = MILHead(256, 128, 32, 64, .1)
        self.assertEqual([type(m) for m in head.structure_encoder], [nn.Linear, nn.LayerNorm, nn.ReLU])
        self.assertEqual(head.structure_encoder[0].weight.shape, (32,9))
        self.assertEqual(head.tanh_projection.in_features, 288)
        self.assertEqual(head.structure_projection.weight.shape, (256,32))
        self.assertFalse(hasattr(head, 'structure_mask'))
        s = torch.rand(7,9,requires_grad=True)
        output = head(torch.randn(7,256), s, torch.tensor([0,0,0,1,1,1,1]))
        self.assertEqual(output.embedding.shape, (2,256))
        output.embedding.square().mean().backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertTrue((s.grad.abs().sum(dim=0) > 0).all())
        with self.assertRaisesRegex(ValueError, 'must be 32'):
            replace(config.mil_head, structure_dim=8)
        with self.assertRaisesRegex(ValueError, 'nodes, 9'):
            head(torch.randn(7,256), torch.rand(7,4), torch.tensor([0,0,0,1,1,1,1]))

if __name__ == '__main__':
    unittest.main()

"""Regression checks for the final paper C+M structural encoder."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from MISGL.config import load_config
from MISGL.models.mil_head import MILHead
from MISGL.structural_features import CM_MASK, compute_cm_features


class StructureSelectionTest(unittest.TestCase):
    def test_triangle_isolate_and_path_statistics(self):
        graph = np.zeros((4, 4))
        graph[:3, :3] = 1
        np.fill_diagonal(graph, 0)
        expected = np.zeros((4, 9), dtype=np.float32)
        expected[:3, 0:2] = 2 / 3
        expected[:3, 4] = 0.5  # log(1+1) / log(1+3)
        expected[:3, 5] = 1
        np.testing.assert_allclose(compute_cm_features(graph), expected)

        directed_weighted = np.array([[7, 2, 0], [0, 3, 5], [0, 0, 9]])
        original = directed_weighted.copy()
        expected = np.zeros((3, 9), dtype=np.float32)
        expected[:, 0] = [0.5, 1, 0.5]
        expected[:, 1] = [1, 0.5, 1]
        np.testing.assert_array_equal(compute_cm_features(directed_weighted), expected)
        np.testing.assert_array_equal(directed_weighted, original)

    def test_empty_singleton_and_invalid_topology(self):
        self.assertEqual(compute_cm_features(np.zeros((0, 0))).shape, (0, 9))
        np.testing.assert_array_equal(compute_cm_features(np.ones((1, 1))), np.zeros((1, 9)))
        with self.assertRaisesRegex(ValueError, "square"):
            compute_cm_features(np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_cm_features(np.array([[np.nan]]))

    def test_selected_dimensions_mask_and_gradient(self):
        torch.manual_seed(1024)
        config = load_config(str(Path(__file__).resolve().parents[1] / "config/train.yml"))
        self.assertEqual((config.mil_head.structure_features, config.mil_head.structure_layers,
                          config.mil_head.structure_dim), ("c_m", 1, 8))
        head = MILHead(256, 128, 8, 64, 0.1, 0.1, structure_layers=1)
        self.assertEqual([type(m) for m in head.structure_encoder], [nn.Linear, nn.LayerNorm, nn.ReLU])
        self.assertEqual(head.structure_encoder[0].weight.shape, (8, 9))
        self.assertEqual(head.tanh_projection.in_features, 264)
        self.assertEqual(head.structure_projection.weight.shape, (256, 8))
        self.assertEqual(head.structure_gate[0].weight.shape, (64, 512))
        self.assertEqual(head.structure_gate[2].weight.shape, (256, 64))
        h = torch.randn(7, 256, requires_grad=True)
        s = torch.rand(7, 9, requires_grad=True)
        bags = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        output = head(h, s, bags)
        self.assertEqual(output.embedding.shape, (2, 256))
        masked_slots = torch.tensor(CM_MASK) == 0
        changed = s.detach().clone()
        changed[:, masked_slots] = torch.randn(7, 5) * 100
        other = head(h, changed, bags)
        torch.testing.assert_close(output.embedding, other.embedding, rtol=0, atol=0)
        torch.testing.assert_close(output.attention, other.attention, rtol=0, atol=0)
        output.embedding.square().mean().backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertGreater(s.grad[:, ~masked_slots].abs().sum().item(), 0)
        self.assertEqual(s.grad[:, masked_slots].abs().sum().item(), 0)
        self.assertGreater(head.structure_encoder[0].weight.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "nodes, 9"):
            head(h, s[:, :7], bags)
        with self.assertRaisesRegex(ValueError, "must be 1 or 2"):
            replace(config.mil_head, structure_layers=3)
        with self.assertRaisesRegex(ValueError, "must be c_m"):
            replace(config.mil_head, structure_features="legacy7")

    def test_optional_two_layer_encoder(self):
        head = MILHead(16, 8, 32, 4, 0.1, 0.1, structure_layers=2)
        self.assertEqual([type(m) for m in head.structure_encoder],
                         [nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout,
                          nn.Linear, nn.LayerNorm, nn.ReLU])
        self.assertEqual(head.structure_encoder[4].weight.shape, (32, 32))
        output = head(torch.randn(3, 16), torch.rand(3, 9), torch.tensor([0, 0, 1]))
        output.embedding.sum().backward()
        self.assertTrue(torch.isfinite(head.structure_encoder[4].weight.grad).all())


if __name__ == "__main__":
    unittest.main()

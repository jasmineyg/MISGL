import unittest

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from MISGL.models.encoder import MISGLEncoder
from MISGL.utils import hparam
from MISGL.utils import hparams_lib
from MISGL.utils.global_variables import g_key
from MISGL.utils.lappe import align_assignment_to_subgraphs


class LapPEAlignmentTest(unittest.TestCase):
    def test_reindexes_assignment_clusters_to_filtered_subgraphs(self):
        original_graph = nx.path_graph(4)
        assignment = sp.csr_matrix(
            (
                np.ones(4, dtype=np.float32),
                (
                    np.arange(4),
                    np.asarray([0, 0, 2, 2]),
                ),
            ),
            shape=(4, 3),
        )
        subgraph_zero = original_graph.subgraph([0, 1]).copy()
        subgraph_one = original_graph.subgraph([2, 3]).copy()

        node_map, node_counts, source_ids, diagnostics = align_assignment_to_subgraphs(
            original_graph,
            assignment,
            [subgraph_zero, subgraph_one],
        )

        np.testing.assert_array_equal(node_map, np.asarray([0, 0, 1, 1]))
        np.testing.assert_array_equal(node_counts, np.asarray([2.0, 2.0]))
        np.testing.assert_array_equal(source_ids, np.asarray([0, 2]))
        self.assertEqual(diagnostics['unmapped_assignment_clusters'], 0)
        self.assertEqual(diagnostics['alignment_mismatch_nodes'], 0)


class PositionClassifierTest(unittest.TestCase):
    def _hparams(self):
        params = hparam.HParams(
            device='cpu',
            channel_list=[3, 4, 4, 4, 1],
            branch_b={'use': False},
            use_lappe=True,
            lap_pe_dim=2,
            pos_dim=3,
            classifier_input_mode='position',
            dropout=0.0,
        )
        return hparams_lib.apply_defaults(params)

    def test_lappe_is_the_classifier_input_when_branch_b_is_disabled(self):
        torch.manual_seed(7)
        model = MISGLEncoder(self._hparams()).eval()
        for module in model.position_mlp.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.constant_(module.weight, 0.5)
                torch.nn.init.constant_(module.bias, 0.0)
        for module in model.classifier.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.constant_(module.weight, 0.5)
                torch.nn.init.constant_(module.bias, 0.0)

        base_input = {
            g_key.x: torch.randn(2, 3, 3),
            g_key.adj_mat: torch.eye(3).repeat(2, 1, 1),
            g_key.node_num: torch.tensor([3, 3]),
        }
        zero_input = dict(base_input)
        zero_input[g_key.lap_pe] = torch.zeros(2, 2)
        one_input = dict(base_input)
        one_input[g_key.lap_pe] = torch.ones(2, 2)

        with torch.no_grad():
            zero_output, zero_embeddings = model.forward_with_embeddings(zero_input)
            one_output, one_embeddings = model.forward_with_embeddings(one_input)

        self.assertFalse(model.use_branch_b)
        self.assertEqual(model.classifier_input_mode, 'position')
        self.assertEqual(model.classifier[0].in_features, 3)
        self.assertTrue(
            torch.equal(
                zero_embeddings['graph_emb_classifier'],
                zero_embeddings['pos_emb'],
            )
        )
        self.assertGreater(
            float(
                (
                    one_embeddings['graph_emb_classifier']
                    - zero_embeddings['graph_emb_classifier']
                ).abs().max()
            ),
            0.0,
        )
        self.assertGreater(float((one_output - zero_output).abs().max()), 0.0)


if __name__ == '__main__':
    unittest.main()

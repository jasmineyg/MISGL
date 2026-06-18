# coding=utf-8

import unittest

import networkx as nx
import numpy as np

from MISGL.utils.boundary_context import compute_boundary_context_rows


def _subgraph(nodes, label=0):
    graph = nx.Graph()
    for node_id in nodes:
        graph.add_node(node_id, features=np.asarray([float(node_id), 1.0], dtype=np.float32))
    graph.graph['label'] = label
    return graph


class BoundaryContextTest(unittest.TestCase):

    def test_boundary_context_allows_overlapping_assignment(self):
        original_graph = nx.Graph()
        for node_id in range(4):
            original_graph.add_node(node_id, features=np.asarray([float(node_id), 1.0], dtype=np.float32))
        original_graph.add_edges_from([(0, 1), (1, 2), (1, 3)])

        subgraphs = [
            _subgraph([0, 1], label=0),
            _subgraph([2], label=1),
            _subgraph([1, 3], label=0),
        ]
        assignment = np.asarray(
            [
                [1, 0, 0],
                [1, 0, 1],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        dataset_raw = {
            'original_graph': original_graph,
            'subgraph_structures': subgraphs,
            'assignment_matrix': assignment,
            'subgraph_labels': np.asarray([0, 1, 0]),
        }

        rows = compute_boundary_context_rows(dataset_raw)

        self.assertEqual(rows[0]['boundary_node_count'], 1)
        self.assertEqual(rows[0]['boundary_ratio'], 0.5)
        self.assertEqual(rows[0]['cross_edge_count'], 2)
        self.assertEqual(rows[0]['context_node_count'], 2)
        self.assertEqual(rows[0]['context_to_boundary_count_ratio'], 2.0)

        self.assertEqual(rows[2]['boundary_node_count'], 1)
        self.assertEqual(rows[2]['cross_edge_count'], 2)
        self.assertEqual(rows[2]['context_node_count'], 2)

    def test_boundary_context_handles_no_external_context(self):
        original_graph = nx.Graph()
        original_graph.add_node(0, features=np.asarray([1.0], dtype=np.float32))
        original_graph.add_node(1, features=np.asarray([2.0], dtype=np.float32))
        original_graph.add_edge(0, 1)

        dataset_raw = {
            'original_graph': original_graph,
            'subgraph_structures': [_subgraph([0, 1], label=0)],
            'assignment_matrix': np.asarray([[1], [1]], dtype=np.float32),
        }

        rows = compute_boundary_context_rows(dataset_raw)

        self.assertEqual(rows[0]['boundary_node_count'], 0)
        self.assertEqual(rows[0]['context_node_count'], 0)
        self.assertEqual(rows[0]['cross_edge_count'], 0)
        self.assertEqual(rows[0]['context_feature_norm'], 0.0)

    def test_boundary_context_requires_original_graph(self):
        with self.assertRaisesRegex(ValueError, 'original_graph'):
            compute_boundary_context_rows({
                'subgraph_structures': [_subgraph([0])],
                'assignment_matrix': np.asarray([[1]], dtype=np.float32),
            })


if __name__ == '__main__':
    unittest.main()

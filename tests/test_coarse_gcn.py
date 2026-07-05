# coding=utf-8

from types import SimpleNamespace

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from MISGL.utils.coarse_graph import build_coarse_adjacency
from run_coarse_gcn import OneLayerCoarseGCN
from run_coarse_gcn import normalize_for_gcn
from run_coarse_gcn import train_stage2


def test_build_coarse_adjacency_active_order_and_topk():
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    assignment = sp.csr_matrix(
        np.asarray(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
    )

    coarse, metadata = build_coarse_adjacency(
        graph,
        assignment,
        active_subgraph_ids=[2, 1, 0],
        top_k=1,
        include_self=False,
        symmetrize=True,
    )

    assert coarse.shape == (3, 3)
    assert metadata['active_subgraph_ids'].tolist() == [2, 1, 0]
    assert np.allclose(coarse.diagonal(), 0.0)
    assert max(np.diff(coarse.indptr)) <= 1
    assert coarse[0, 1] > 0
    assert coarse[0, 2] == 0


def test_one_layer_coarse_gcn_shapes():
    z_mil = torch.randn(4, 5)
    adj = sp.csr_matrix(
        np.asarray(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.float32,
        )
    )
    coo = normalize_for_gcn(adj).tocoo()
    adj_norm = torch.sparse_coo_tensor(
        torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long),
        torch.tensor(coo.data, dtype=torch.float32),
        torch.Size(coo.shape),
    ).coalesce()

    model = OneLayerCoarseGCN(input_dim=5, hidden_dim=7, dropout=0.0)
    logits, z_pos = model(z_mil, adj_norm)

    assert logits.shape == (4,)
    assert z_pos.shape == (4, 5)


def test_train_stage2_smoke_cpu():
    z_mil = torch.randn(12, 4)
    labels = (z_mil[:, 0] > 0).long()
    coarse_adj = sp.eye(12, dtype=np.float32, format='csr')
    masks = {
        'train': torch.tensor([True] * 6 + [False] * 6),
        'val': torch.tensor([False] * 6 + [True] * 3 + [False] * 3),
        'test': torch.tensor([False] * 9 + [True] * 3),
    }
    args = SimpleNamespace(
        device='cpu',
        stage2_hidden_dim=8,
        stage2_dropout=0.0,
        stage2_lr=0.01,
        stage2_weight_decay=0.0,
        stage2_epochs=2,
        stage2_patience=2,
    )

    output = train_stage2(z_mil, labels, masks, coarse_adj, args)

    assert output['z_pos'].shape == z_mil.shape
    assert output['logits'].shape == (12,)
    assert 'test' in output['final_metrics']

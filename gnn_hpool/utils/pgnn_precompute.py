# pgnn_precompute.py
import random
import numpy as np
import networkx as nx
import torch

def precompute_dist_data(edge_index, num_nodes, approximate=0):
    graph = nx.Graph()
    edge_list = edge_index.transpose(1, 0).tolist()
    graph.add_edges_from(edge_list)

    dists_array = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    cutoff = approximate if approximate and approximate > 0 else None

    for src in graph.nodes():
        sp = nx.single_source_shortest_path_length(graph, src, cutoff=cutoff)
        for dst, dist in sp.items():
            dists_array[src, dst] = 1.0 / (dist + 1.0)
    return dists_array

def get_random_anchorset(n, c=1.0):
    m = int(np.log2(n))
    copy = int(c * m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n / (2 ** (i + 1)))
        for _ in range(copy):
            anchorset_id.append(np.random.choice(n, size=anchor_size, replace=False))
    return anchorset_id

def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0], len(anchorset_id)), device=device)
    dist_argmax = torch.zeros((dist.shape[0], len(anchorset_id)), dtype=torch.long, device=device)
    for i, ids in enumerate(anchorset_id):
        temp_id = torch.as_tensor(ids, dtype=torch.long, device=device)
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:, i] = dist_max_temp
        dist_argmax[:, i] = temp_id[dist_argmax_temp]
    return dist_max, dist_argmax

def build_pgnn_inputs(edge_index, num_nodes, device, approximate=0, c=1.0):
    dists = precompute_dist_data(edge_index, num_nodes, approximate=approximate)
    dists = torch.from_numpy(dists).to(device)
    anchorset_id = get_random_anchorset(num_nodes, c=c)
    dists_max, dists_argmax = get_dist_max(anchorset_id, dists, device=device)
    return dists_max, dists_argmax
import os
import torch
import numpy as np
import networkx as nx

def get_random_anchors(num_nodes, k, seed=42):
    """
    Sample k random anchor nodes.
    """
    rng = np.random.RandomState(seed)
    if k >= num_nodes:
        return np.arange(num_nodes)
    return rng.choice(num_nodes, size=k, replace=False)

def compute_anchor_distances(edge_index, num_nodes, anchors, max_dist_val=None):
    """
    Compute shortest path distances from anchors to all nodes using Dijkstra (BFS for unweighted).
    """
    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    if edge_index is not None and edge_index.numel() > 0:
        # edge_index: [2, E]
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.cpu().numpy()
        
        edge_list = edge_index.T.tolist()
        G.add_edges_from(edge_list)
    
    if max_dist_val is None:
        max_dist_val = num_nodes 

    # dists_matrix: [num_nodes, k]
    dists_matrix = np.full((num_nodes, len(anchors)), max_dist_val, dtype=np.int64)
    
    # For unweighted graph, single_source_shortest_path_length is BFS which is O(V+E)
    # User requested Dijkstra, but for unweighted graph BFS is equivalent and faster.
    # If weights are needed, we'd use dijkstra_path_length. 
    # Assuming unweighted based on "d_i in N^k" (integer distances).
    
    for i, anchor in enumerate(anchors):
        try:
            # Returns dictionary {target: length}
            length_dict = nx.single_source_shortest_path_length(G, source=anchor)
            
            for target, length in length_dict.items():
                dists_matrix[target, i] = length
        except Exception:
            # Should not happen for valid nodes, but safety net
            pass
            
    return dists_matrix

def precompute_and_save_pgnn(data_name, edge_index, num_nodes, save_dir, k=32, seed=42, force_recompute=False):
    """
    Main function to precompute and save/load anchor distances.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    file_path = os.path.join(save_dir, f'{data_name}_dist.pt')
    
    if os.path.exists(file_path) and not force_recompute:
        print(f"[PGNN] Loading precomputed distances from {file_path}")
        try:
            return torch.load(file_path, weights_only=False)
        except TypeError:
             # For older pytorch versions
            return torch.load(file_path)
    
    print(f"[PGNN] Computing distances for {data_name} (Nodes: {num_nodes}, Anchors: {k})...")
    
    anchors = get_random_anchors(num_nodes, k, seed)
    
    dists = compute_anchor_distances(edge_index, num_nodes, anchors, max_dist_val=num_nodes)
    
    # Convert to tensor
    anchor_distance_index = torch.from_numpy(dists).long()
    
    # Mask: 1 if reachable (dist < max_dist), 0 otherwise
    # Unreachable nodes have dist = num_nodes
    anchor_mask = (anchor_distance_index < num_nodes).float()
    
    result = {
        'anchor_distance_index': anchor_distance_index, # [N, k]
        'anchor_mask': anchor_mask,                     # [N, k]
        'anchors': anchors
    }
    
    torch.save(result, file_path)
    print(f"[PGNN] Saved to {file_path}")
    return result

def unit_test():
    """
    Randomly generate 20 nodes coarse graph, verify anchor_distance_index max <= |V_coarse| and no NaN.
    """
    print("Running PGNN Precompute Unit Test...")
    num_nodes = 20
    k = 5
    
    # Generate random edges
    sources = np.random.randint(0, num_nodes, 40)
    targets = np.random.randint(0, num_nodes, 40)
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    
    save_dir = 'pgnn_precompute_test'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    result = precompute_and_save_pgnn('test_graph', edge_index, num_nodes, save_dir, k=k)
    
    dists = result['anchor_distance_index']
    mask = result['anchor_mask']
    
    print(f"Dist shape: {dists.shape}, Mask shape: {mask.shape}")
    
    # Validation
    assert not torch.isnan(dists).any(), "NaN found in distances"
    assert dists.max() <= num_nodes, f"Max distance {dists.max()} exceeds num_nodes {num_nodes}"
    
    print("Unit Test Passed!")

if __name__ == '__main__':
    unit_test()

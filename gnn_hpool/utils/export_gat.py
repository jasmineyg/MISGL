import os
import pickle
import torch
import numpy as np
from gnn_hpool.utils.global_variables import g_key

def export_gat1_features(model, dataloader, epoch, dataset_raw, split="val", output_dir="/data/yg/Subgraph-MIL/DataAnalyze"):
    """
    Exports the first layer GAT features for the given dataloader (usually validation set).
    Reconstructs the node features to match the original graph's node order.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Starting GAT1 feature export for epoch {epoch} ({split} set)...")
    
    # Check if dataset_raw has required information
    if dataset_raw is None:
        print("[Warning] dataset_raw is None, cannot export GAT features with original node mapping.")
        return
        
    original_graph = dataset_raw.get('original_graph', None)
    if original_graph is None:
        print("[Warning] original_graph not found in dataset_raw. Using assignment_matrix shape or heuristic for N.")
        assignment_matrix = dataset_raw.get('assignment_matrix', None)
        if assignment_matrix is not None:
            N_total = assignment_matrix.shape[0]
        else:
            N_total = len(dataset_raw.get('node_binary_labels', []))
    else:
        N_total = len(original_graph.nodes)
        
    if N_total == 0:
        print("[Warning] N_total is 0, cannot determine original graph size.")
        return

    model.eval()
    
    # We don't know the hidden dimension until we run a batch
    d = None
    node_features = None
    
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch_idx, graph_data in enumerate(dataloader):
            # Move data to device
            for key, value in graph_data.items():
                if isinstance(value, torch.Tensor):
                    graph_data[key] = value.to(device)
                    
            out = model(graph_data)
            
            if not hasattr(model, 'current_x2'):
                print("[Warning] 'current_x2' not found in model. Cannot export GAT1 features.")
                return
                
            x2 = model.current_x2 # [B, max_num_nodes, d]
            
            if d is None:
                d = x2.shape[2]
                node_features = np.zeros((N_total, d), dtype=np.float32)
                
            # x2 shape: [B, max_num_nodes, d]
            batch_num_nodes = graph_data[g_key.node_num]
            if isinstance(batch_num_nodes, torch.Tensor):
                num_list = [int(n) for n in batch_num_nodes.detach().cpu().tolist()]
            else:
                num_list = [int(n) for n in batch_num_nodes]
                
            orig_idx_tensor = graph_data.get(g_key.orig_graph_idx, None)
            if orig_idx_tensor is not None and isinstance(orig_idx_tensor, torch.Tensor):
                orig_graph_indices = [int(i) for i in orig_idx_tensor.detach().cpu().tolist()]
            else:
                print("[Warning] orig_graph_idx not found, mapping might be incorrect.")
                continue
                
            # Get mapping for each subgraph in the batch
            subgraphs = dataset_raw.get('subgraph_structures', [])
            for i, (orig_idx, n_i) in enumerate(zip(orig_graph_indices, num_list)):
                if n_i <= 0:
                    continue
                if 0 <= orig_idx < len(subgraphs):
                    subgraph = subgraphs[orig_idx]
                    nodes = list(subgraph.nodes())
                    
                    # Extract features for valid nodes
                    # shape: [n_i, d]
                    feat = x2[i, :n_i, :].detach().cpu().numpy()
                    
                    for j, node_id in enumerate(nodes[:n_i]):
                        attr = subgraph.nodes[node_id]
                        orig_node_id = None
                        for key in ('original_id', 'original_index', 'orig_id', 'node_index'):
                            if key in attr and attr[key] is not None:
                                try:
                                    orig_node_id = int(attr[key])
                                    break
                                except Exception:
                                    pass
                        if orig_node_id is None and isinstance(node_id, (int, np.integer)):
                            orig_node_id = int(node_id)
                            
                        if orig_node_id is not None and 0 <= orig_node_id < N_total:
                            node_features[orig_node_id] = feat[j]
    
    if node_features is None:
        print("[Warning] No features were extracted.")
        return
        
    # Calculate subgraph node counts if not explicitly available
    subgraph_node_counts = []
    subgraphs = dataset_raw.get('subgraph_structures', [])
    for g in subgraphs:
        subgraph_node_counts.append(len(g.nodes))
        
    # Build output dictionary
    output_data = {
        'node_features': node_features,
        'node_binary_labels': dataset_raw.get('node_binary_labels', None),
        'node_categories': dataset_raw.get('node_categories', None),
        'original_graph': dataset_raw.get('original_graph', None),
        'dataset_metadata': dataset_raw.get('dataset_metadata', None),
        'subgraph_assignment': dataset_raw.get('assignment_matrix', None),
        'subgraph_labels': dataset_raw.get('subgraph_labels', None),
        'subgraph_node_counts': np.array(subgraph_node_counts) if subgraph_node_counts else None
    }
    
    filename = f"gat1_epoch{epoch}_{split}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(output_data, f)
        
    print(f"\n[INFO] ========================================")
    print(f"[INFO] Successfully exported GAT1 features to {filepath}")
    print(f"[INFO] ========================================\n")

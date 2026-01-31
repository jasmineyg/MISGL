import torch
import torch.nn as nn
import torch.nn.functional as F

class GINLayer(nn.Module):
    """
    Graph Isomorphism Network (GIN) Layer.
    Formula: h_v' = MLP((1 + eps) * h_v + sum_{u in N(v)} h_u)
    """
    def __init__(self, in_dim, out_dim, hidden_dim=None, eps=0.0, train_eps=True):
        super(GINLayer, self).__init__()
        if hidden_dim is None:
            hidden_dim = out_dim
            
        # MLP: Linear -> ReLU -> Linear
        # Note: We can add Batch Norm if needed, but for now keeping it simple as per request context
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, adj):
        """
        Args:
            x: Node features [B, N, D_in]
            adj: Adjacency matrix [B, N, N]
        """
        # Aggregate neighbors: A * X
        # GIN assumes sum aggregation. If adj is binary, this works as sum.
        neighbor_sum = torch.bmm(adj, x)
        
        # (1 + eps) * X + neighbor_sum
        out = (1 + self.eps) * x + neighbor_sum
        
        # Apply MLP
        out = self.mlp(out)
        
        return out

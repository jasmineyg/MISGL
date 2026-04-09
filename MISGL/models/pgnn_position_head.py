import torch
import torch.nn as nn
import torch.nn.functional as F

class SubgraphPositionEncoder(nn.Module):
    """
    Module 2: Subgraph Position Encoder.
    Encodes position using anchor distances and fuses with subgraph features.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, max_dist, k_anchors, d_pos=64, dropout=0.5):
        super(SubgraphPositionEncoder, self).__init__()
        self.d_pos = d_pos
        self.k_anchors = k_anchors
        
        # 1. Embedding Layer
        # Input indices are in [0, max_dist]. Size should be max_dist + 1.
        self.pos_embedding = nn.Embedding(max_dist + 1, d_pos)
        
        # 2. Attention Pooling
        # "Weighted average along anchor dimension"
        # We learn a weight for each anchor position context
        self.attn_fc = nn.Linear(d_pos, 1)
        
        # 3. Fusion & Classifier
        # Concatenate h (input_dim) + p (d_pos)
        fusion_dim = input_dim + d_pos
        
        # Two layers MLP (with BN, ReLU, Dropout=0.5)
        # Layer 1
        self.mlp_1 = nn.Linear(fusion_dim, hidden_dim)
        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        
        # Layer 2
        self.mlp_2 = nn.Linear(hidden_dim, num_classes)
        # Usually final layer doesn't have BN/ReLU/Dropout if it's logits, but user said:
        # "Two layers MLP (with BN, ReLU, Dropout=0.5) output bag-level logits"
        # This implies the structure is applied. But typically the last linear outputs logits directly.
        # I will assume: Linear -> BN -> ReLU -> Dropout -> Linear -> Logits.
        # This fits "Two layers MLP".
        
    def forward(self, h, anchor_dist_index, anchor_mask=None):
        """
        Args:
            h: Subgraph features [B, input_dim]
            anchor_dist_index: Distance indices [B, k]
            anchor_mask: Mask for valid anchors [B, k] (1=valid, 0=invalid)
        Returns:
            logits: [B, num_classes]
        """
        # 1. Embedding
        # p_raw: [B, k, d_pos]
        p_raw = self.pos_embedding(anchor_dist_index)
        
        # 2. Attention Pooling
        # scores: [B, k, 1]
        scores = self.attn_fc(p_raw)
        
        if anchor_mask is not None:
            # Mask out unreachable anchors (set scores to very small number)
            # anchor_mask is 1 for valid, 0 for invalid
            # We want to mask where anchor_mask is 0
            scores = scores.masked_fill(anchor_mask.unsqueeze(-1) == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=1) # [B, k, 1]
        
        # p: [B, d_pos] - Weighted sum
        p = torch.sum(p_raw * attn_weights, dim=1)
        
        # 3. Fusion
        # p_cat: [B, input_dim + d_pos]
        p_cat = torch.cat([h, p], dim=1)
        
        # 4. Classifier
        # Layer 1
        out = self.mlp_1(p_cat)
        out = self.bn_1(out)
        out = F.relu(out)
        out = self.dropout_1(out)
        
        # Layer 2 (Output)
        logits = self.mlp_2(out)
        
        return logits

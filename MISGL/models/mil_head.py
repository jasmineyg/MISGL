"""Structure-enhanced gated-attention MIL pooling."""

from dataclasses import dataclass

import torch
from torch import nn

from MISGL.structural_features import CM_MASK, INPUT_DIM


@dataclass(frozen=True)
class MILOutput:
    """Node attention and the resulting bag embeddings."""

    embedding: torch.Tensor
    attention: torch.Tensor
    bag_index: torch.Tensor


class MILHead(nn.Module):
    """Fuse node structure and pool variable-sized bags with gated attention."""

    def __init__(
        self,
        node_dim: int,
        attention_dim: int,
        structure_dim: int,
        structure_gate_dim: int,
        dropout: float,
        residual_init: float,
        structure_layers: int = 1,
    ) -> None:
        super().__init__()
        if min(node_dim, attention_dim, structure_dim, structure_gate_dim) <= 0:
            raise ValueError("MIL dimensions must be positive")

        if type(structure_layers) is not int or structure_layers not in (1, 2):
            raise ValueError("structure_layers must be 1 or 2")
        self.register_buffer(
            "structure_mask", torch.tensor(CM_MASK, dtype=torch.float32),
            persistent=False,
        )
        layers = [nn.Linear(INPUT_DIM, structure_dim), nn.LayerNorm(structure_dim), nn.ReLU()]
        if structure_layers == 2:
            layers.extend([
                nn.Dropout(dropout), nn.Linear(structure_dim, structure_dim),
                nn.LayerNorm(structure_dim), nn.ReLU(),
            ])
        self.structure_encoder = nn.Sequential(*layers)
        self.structure_projection = nn.Linear(structure_dim, node_dim)
        self.structure_gate = nn.Sequential(
            nn.Linear(node_dim * 2, structure_gate_dim),
            nn.ReLU(),
            nn.Linear(structure_gate_dim, node_dim),
            nn.Sigmoid(),
        )
        self.structure_scale = nn.Parameter(
            torch.tensor(float(residual_init), dtype=torch.float32)
        )

        score_dim = node_dim + structure_dim
        self.tanh_projection = nn.Linear(score_dim, attention_dim)
        self.sigmoid_projection = nn.Linear(score_dim, attention_dim)
        self.score_projection = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        structure: torch.Tensor,
        bag_index: torch.Tensor,
    ) -> MILOutput:
        if node_embeddings.ndim != 2:
            raise ValueError("node_embeddings must have shape [nodes, features]")
        if structure.ndim != 2 or structure.shape != (node_embeddings.size(0), INPUT_DIM):
            raise ValueError("structure must have shape [nodes, 9]")
        if bag_index.ndim != 1 or bag_index.size(0) != node_embeddings.size(0):
            raise ValueError("bag_index must contain one id per node")
        if node_embeddings.size(0) == 0:
            raise ValueError("MILHead requires at least one node")
        if bag_index.dtype != torch.long:
            raise TypeError("bag_index must have dtype torch.long")
        if structure.device != node_embeddings.device:
            raise ValueError("structure and node_embeddings must share a device")
        if bag_index.device != node_embeddings.device:
            raise ValueError("bag_index and node_embeddings must share a device")
        if torch.any(bag_index < 0):
            raise ValueError("bag_index must be non-negative")

        bag_count = int(bag_index.max().item()) + 1
        if torch.any(torch.bincount(bag_index, minlength=bag_count) == 0):
            raise ValueError("bag ids must be contiguous and start at zero")

        node_structure = self.structure_encoder(structure * self.structure_mask)
        score_input = torch.cat((node_embeddings, node_structure), dim=-1)
        gated_score = torch.tanh(self.tanh_projection(score_input))
        gated_score = gated_score * torch.sigmoid(
            self.sigmoid_projection(score_input)
        )
        scores = self.score_projection(gated_score).squeeze(-1).clamp(-12.0, 12.0)

        maximum = scores.new_full((bag_count,), -torch.inf)
        maximum.scatter_reduce_(
            0,
            bag_index,
            scores,
            reduce="amax",
            include_self=True,
        )
        exponentials = torch.exp(scores - maximum[bag_index])
        normalizer = scores.new_zeros(bag_count)
        normalizer.scatter_add_(0, bag_index, exponentials)
        attention = exponentials / normalizer[bag_index]
        attention = attention.clamp(1.0e-6, 1.0 - 1.0e-6)

        node_embedding = self._weighted_sum(
            node_embeddings, attention, bag_index, bag_count
        )
        bag_structure = self._weighted_sum(
            node_structure, attention, bag_index, bag_count
        )
        projected_structure = self.structure_projection(bag_structure)
        structure_gate = self.structure_gate(
            torch.cat((node_embedding, projected_structure), dim=-1)
        )
        embedding = (
            node_embedding
            + self.structure_scale * structure_gate * projected_structure
        )
        return MILOutput(
            embedding=embedding,
            attention=attention,
            bag_index=bag_index,
        )

    @staticmethod
    def _weighted_sum(
        values: torch.Tensor,
        attention: torch.Tensor,
        bag_index: torch.Tensor,
        bag_count: int,
    ) -> torch.Tensor:
        weighted = attention.unsqueeze(-1) * values
        pooled = values.new_zeros((bag_count, values.size(1)))
        pooled.index_add_(0, bag_index, weighted)
        return pooled

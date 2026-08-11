"""Sparse position-aware classification head."""

import torch
from torch import nn


class POSHead(nn.Module):
    """Classify samples after one sparse relation-propagation step."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError('embedding_dim must be positive')
        if hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive')

        self.embedding_dim = embedding_dim
        self.position_projection = nn.Linear(embedding_dim, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        embedding: torch.Tensor,
        normalized_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        if embedding.ndim != 2:
            raise ValueError('embedding must have shape [samples, features]')
        if embedding.size(1) != self.embedding_dim:
            raise ValueError(
                'expected embedding dimension {}, got {}'.format(
                    self.embedding_dim, embedding.size(1)
                )
            )
        if normalized_adjacency.layout != torch.sparse_coo:
            raise TypeError('normalized_adjacency must be a sparse COO tensor')
        sample_count = embedding.size(0)
        if normalized_adjacency.shape != (sample_count, sample_count):
            raise ValueError('normalized_adjacency shape must match sample count')
        if normalized_adjacency.device != embedding.device:
            raise ValueError(
                'normalized_adjacency and embedding must be on the same device'
            )

        propagated = torch.sparse.mm(normalized_adjacency, embedding)
        position_embedding = torch.relu(self.position_projection(propagated))
        combined = torch.cat((embedding, position_embedding), dim=-1)
        return self.classifier(combined).squeeze(-1)

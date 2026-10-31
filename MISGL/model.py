"""Residual GAT backbone and the first-stage subgraph encoder of NASGL."""

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import nn

from MISGL.config import Config
from MISGL.internal_head import InternalHead


class ResidualGAT(nn.Module):
    """Apply residual graph attention over an explicit edge list."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        heads: int,
        attention_dropout: float,
        feature_dropout: float,
        negative_slope: float,
        residual: bool,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if heads <= 0 or output_dim % heads != 0:
            raise ValueError("output_dim must be divisible by a positive head count")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.head_dim = output_dim // heads
        self.residual = residual

        self.weight = nn.Parameter(torch.empty(heads, input_dim, self.head_dim))
        self.source_attention = nn.Parameter(torch.empty(heads, self.head_dim))
        self.target_attention = nn.Parameter(torch.empty(heads, self.head_dim))
        self.activation = nn.LeakyReLU(negative_slope)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.residual_projection = (
            nn.Linear(input_dim, output_dim)
            if residual and input_dim != output_dim
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("leaky_relu", self.activation.negative_slope)
        for head_weight in self.weight:
            nn.init.xavier_uniform_(head_weight, gain=gain)
        nn.init.xavier_uniform_(self.source_attention, gain=gain)
        nn.init.xavier_uniform_(self.target_attention, gain=gain)
        if self.residual_projection is not None:
            nn.init.xavier_uniform_(self.residual_projection.weight, gain=gain)
            nn.init.zeros_(self.residual_projection.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if node_features.ndim != 2:
            raise ValueError("node_features must have shape [nodes, features]")
        if node_features.size(1) != self.input_dim:
            raise ValueError(
                "expected node feature dimension {}, got {}".format(
                    self.input_dim, node_features.size(1)
                )
            )
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long")
        if edge_index.device != node_features.device:
            raise ValueError("edge_index and node_features must be on the same device")
        if node_features.size(0) == 0 or edge_index.size(1) == 0:
            raise ValueError("graph-attention input must contain nodes and edges")

        node_count = node_features.size(0)
        if torch.any(edge_index < 0) or torch.any(edge_index >= node_count):
            raise IndexError("edge_index contains an invalid node index")

        centers, neighbors = edge_index
        if torch.any(torch.bincount(centers, minlength=node_count) == 0):
            raise ValueError("every node must have an outgoing edge or self-loop")

        features = self.feature_dropout(node_features)
        projected = torch.einsum("nf,hfd->hnd", features, self.weight)
        source_scores = torch.einsum("hnd,hd->hn", projected, self.source_attention)
        target_scores = torch.einsum("hnd,hd->hn", projected, self.target_attention)
        edge_scores = self.activation(
            source_scores[:, centers] + target_scores[:, neighbors]
        )

        head_offsets = (
            torch.arange(
                self.heads, device=edge_index.device, dtype=torch.long
            ).unsqueeze(1)
            * node_count
        )
        attention_groups = (centers.unsqueeze(0) + head_offsets).reshape(-1)
        flat_scores = edge_scores.reshape(-1)
        maxima = flat_scores.new_full((self.heads * node_count,), -torch.inf)
        maxima.scatter_reduce_(
            0,
            attention_groups,
            flat_scores,
            reduce="amax",
            include_self=True,
        )
        exponentials = torch.exp(flat_scores - maxima[attention_groups])
        normalizers = flat_scores.new_zeros(self.heads * node_count)
        normalizers.scatter_add_(0, attention_groups, exponentials)
        attention = (exponentials / normalizers[attention_groups]).reshape_as(
            edge_scores
        )
        attention = self.attention_dropout(attention)

        messages = attention.unsqueeze(-1) * projected[:, neighbors, :]
        attended = projected.new_zeros((self.heads, node_count, self.head_dim))
        attended.index_add_(1, centers, messages)
        attended = attended.transpose(0, 1).reshape(node_count, self.output_dim)

        if self.residual:
            residual = features
            if self.residual_projection is not None:
                residual = self.residual_projection(residual)
            attended = attended + residual
        return attended


@dataclass(frozen=True)
class SubgraphEncoding:
    """First-stage logits, internal representation z_int and node attention weights."""

    logits: torch.Tensor
    z_int: torch.Tensor
    attention_weights: Optional[torch.Tensor]
    subgraph_index: Optional[torch.Tensor]


class SubgraphEncoder(nn.Module):
    """Stage 1: residual GAT, Internal-Head (or mean pooling), and a classifier.

    Stage 2 fits ContextHead separately on the frozen z_int representations."""

    def __init__(self, config: Config, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        model_config = config.model
        self.node_encoder = ResidualGAT(
            input_dim=input_dim,
            output_dim=model_config.node_dim,
            heads=model_config.gat_heads,
            attention_dropout=model_config.gat_attention_dropout,
            feature_dropout=model_config.gat_feature_dropout,
            negative_slope=model_config.gat_negative_slope,
            residual=model_config.gat_residual,
        )
        self.internal_head = (
            InternalHead(
                node_dim=model_config.node_dim,
                attention_dim=config.internal_head.attention_dim,
                structure_dim=config.internal_head.structure_dim,
                structure_gate_dim=config.internal_head.structure_gate_dim,
                residual_init=config.internal_head.residual_scale_init,
            )
            if config.internal_head.enabled
            else None
        )
        self.internal_classifier = nn.Sequential(
            nn.Linear(model_config.node_dim, model_config.classifier_dim),
            nn.LeakyReLU(model_config.gat_negative_slope),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.classifier_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match the encoder and classifier initialization used by NASGL."""
        self.node_encoder.reset_parameters()
        negative_slope = self.internal_classifier[1].negative_slope
        gain = nn.init.calculate_gain("leaky_relu", negative_slope)
        for layer in self.internal_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def forward(self, graph_batch: Mapping[str, torch.Tensor]) -> SubgraphEncoding:
        h = self.node_encoder(
            graph_batch["x"],
            graph_batch["edge_index"],
        )
        subgraph_index = graph_batch["subgraph_index"]
        if subgraph_index.ndim != 1 or subgraph_index.size(0) != h.size(0):
            raise ValueError("subgraph_index must contain one id per node")
        if subgraph_index.dtype != torch.long:
            raise TypeError("subgraph_index must have dtype torch.long")
        if subgraph_index.device != h.device:
            raise ValueError("subgraph_index and node embeddings must share a device")

        if self.internal_head is not None:
            internal_output = self.internal_head(
                h,
                graph_batch["structure"],
                subgraph_index,
            )
            z_int = internal_output.z_int
            attention_weights = internal_output.attention_weights
            attention_subgraphs = internal_output.subgraph_index
        else:
            subgraph_count = int(subgraph_index.max().item()) + 1
            subgraph_sizes = torch.bincount(subgraph_index, minlength=subgraph_count)
            if torch.any(subgraph_sizes == 0):
                raise ValueError("subgraph ids must be contiguous and start at zero")
            z_int = h.new_zeros((subgraph_count, h.size(1)))
            z_int.index_add_(0, subgraph_index, h)
            z_int = z_int / subgraph_sizes.unsqueeze(-1)
            attention_weights = None
            attention_subgraphs = None

        return SubgraphEncoding(
            logits=self.internal_classifier(z_int).squeeze(-1),
            z_int=z_int,
            attention_weights=attention_weights,
            subgraph_index=attention_subgraphs,
        )

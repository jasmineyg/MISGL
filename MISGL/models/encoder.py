"""MISGL subgraph encoder and binary classifier."""

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import nn

from MISGL.config import Config
from MISGL.keys import BAG_INDEX, EDGE_INDEX, STRUCTURE, X
from MISGL.layers.graph_attention import GraphAttentionLayer
from MISGL.models.mil_head import MILHead


@dataclass(frozen=True)
class ModelOutput:
    """Predictions and the subgraph representation used by the classifier."""

    logits: torch.Tensor
    embedding: torch.Tensor
    attention: Optional[torch.Tensor]
    bag_index: Optional[torch.Tensor]


class MISGLModel(nn.Module):
    """Encode subgraphs, apply optional MIL pooling, and classify them."""

    def __init__(self, config: Config, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        model_config = config.model
        self.encoder = GraphAttentionLayer(
            input_dim=input_dim,
            output_dim=model_config.encoder_dim,
            heads=model_config.gat_heads,
            attention_dropout=model_config.gat_attention_dropout,
            feature_dropout=model_config.gat_feature_dropout,
            negative_slope=model_config.gat_negative_slope,
            residual=model_config.gat_residual,
        )
        self.mil_head = (
            MILHead(
                node_dim=model_config.encoder_dim,
                attention_dim=config.mil_head.attention_dim,
                structure_dim=config.mil_head.structure_dim,
                structure_gate_dim=config.mil_head.structure_gate_dim,
                dropout=config.mil_head.structure_dropout,
                residual_init=config.mil_head.structure_residual_init,
            )
            if config.mil_head.enabled
            else None
        )
        self.classifier = nn.Sequential(
            nn.Linear(model_config.encoder_dim, model_config.classifier_dim),
            nn.LeakyReLU(model_config.gat_negative_slope),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.classifier_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match the encoder and classifier initialization used by MISGL."""
        self.encoder.reset_parameters()
        negative_slope = self.classifier[1].negative_slope
        gain = nn.init.calculate_gain("leaky_relu", negative_slope)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def forward(self, graph_batch: Mapping[str, torch.Tensor]) -> ModelOutput:
        node_embeddings = self.encoder(
            graph_batch[X],
            graph_batch[EDGE_INDEX],
        )
        bag_index = graph_batch[BAG_INDEX]
        if bag_index.ndim != 1 or bag_index.size(0) != node_embeddings.size(0):
            raise ValueError("bag_index must contain one id per node")
        if bag_index.dtype != torch.long:
            raise TypeError("bag_index must have dtype torch.long")
        if bag_index.device != node_embeddings.device:
            raise ValueError("bag_index and node embeddings must share a device")

        if self.mil_head is not None:
            mil_output = self.mil_head(
                node_embeddings,
                graph_batch[STRUCTURE],
                bag_index,
            )
            embedding = mil_output.embedding
            attention = mil_output.attention
            attention_bags = mil_output.bag_index
        else:
            bag_count = int(bag_index.max().item()) + 1
            bag_sizes = torch.bincount(bag_index, minlength=bag_count)
            if torch.any(bag_sizes == 0):
                raise ValueError("bag ids must be contiguous and start at zero")
            embedding = node_embeddings.new_zeros(
                (bag_count, node_embeddings.size(1))
            )
            embedding.index_add_(0, bag_index, node_embeddings)
            embedding = embedding / bag_sizes.unsqueeze(-1)
            attention = None
            attention_bags = None

        return ModelOutput(
            logits=self.classifier(embedding).squeeze(-1),
            embedding=embedding,
            attention=attention,
            bag_index=attention_bags,
        )

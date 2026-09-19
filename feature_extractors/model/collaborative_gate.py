"""Modality-agnostic adaptation of the reference collaborative gate classifier."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


ORDER = ("T", "V", "A")  # Preserve the reference model's fused T,V,A order.


class CollaborativeGateClassifier(nn.Module):
    def __init__(
        self,
        input_dims: dict[str, int],
        shared_dim: int = 256,
        projection_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not input_dims or set(input_dims) - set(ORDER):
            raise ValueError(f"Invalid modalities: {list(input_dims)}")
        if min(*input_dims.values(), shared_dim, projection_dim) <= 0 or not 0 <= dropout < 1:
            raise ValueError("Dimensions must be positive and dropout must be in [0, 1)")
        self.modalities = tuple(modality for modality in ORDER if modality in input_dims)
        self.shared_dim = shared_dim
        self.projections = nn.ModuleDict({m: nn.Linear(input_dims[m], shared_dim) for m in self.modalities})
        self.norms = nn.ModuleDict({m: nn.LayerNorm(shared_dim) for m in self.modalities})
        if len(self.modalities) > 1:
            self.pair_projection = nn.Linear(2 * shared_dim, projection_dim)
            self.gate_projection = nn.Linear(projection_dim, shared_dim)
        self.classifier = nn.Sequential(
            nn.Linear(len(self.modalities) * shared_dim, 2 * shared_dim),
            nn.LayerNorm(2 * shared_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * shared_dim, shared_dim),
            nn.LayerNorm(shared_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(shared_dim, 128),
            nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if set(features) != set(self.modalities):
            raise ValueError(f"Expected {self.modalities}, received {tuple(features)}")
        shared = {}
        for modality in self.modalities:
            value = features[modality]
            if value.ndim != 2:
                raise ValueError(f"{modality} must have shape [batch, dimension]")
            shared[modality] = self.norms[modality](F.relu(self.projections[modality](value)))

        fused = []
        for anchor in self.modalities:
            value = shared[anchor]
            if len(self.modalities) > 1:
                pair_weights = [
                    torch.softmax(self.pair_projection(torch.cat((value, shared[peer]), dim=-1)), dim=-1)
                    for peer in self.modalities if peer != anchor
                ]
                gate = torch.softmax(self.gate_projection(torch.stack(pair_weights).sum(0)), dim=-1)
                value = value * (gate * self.shared_dim)
            fused.append(value)
        return self.classifier(torch.cat(fused, dim=-1))

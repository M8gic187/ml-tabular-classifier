"""Configurable MLP for tabular classification with batch norm and dropout."""

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with optional batch normalization and dropout.

    Each hidden block follows: Linear -> BatchNorm (optional) -> ReLU -> Dropout.
    The final layer is a raw linear projection (logits); no softmax applied here
    so the model is compatible with nn.CrossEntropyLoss directly.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = (256, 128, 64),
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

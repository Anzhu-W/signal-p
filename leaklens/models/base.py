from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class SCAModel(nn.Module, ABC):
    n_classes: int
    leakage_model: str   # "identity" | "HW" | "MSB"
    input_shape: tuple   # e.g. (700,) for 1-D

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, n_classes)."""

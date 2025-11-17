"""DeepSets encoder for population states."""

from __future__ import annotations

import torch
from torch import nn


class DeepSetsEncoder(nn.Module):
    """Permutation invariant encoder."""

    def __init__(self, d_in: int, h: int = 128, s: int = 256) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(h, s),
            nn.ReLU(),
            nn.Linear(s, s),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError("Expected tensor with shape (B, N, D).")
        phi_out = self.phi(x)
        pooled = phi_out.mean(dim=1)
        return self.rho(pooled)

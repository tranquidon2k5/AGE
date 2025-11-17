"""Problem definitions for constrained optimization benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ConstrainedProblem:
    """Base class for constrained minimization problems."""

    dim: int
    lower: np.ndarray
    upper: np.ndarray
    n_g: int = 0
    n_h: int = 0

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Sample n points uniformly inside the box constraints."""
        return rng.uniform(self.lower, self.upper, size=(n, self.dim))

    def evaluate(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return dict with keys f, g, h."""
        raise NotImplementedError


class ToyCEC(ConstrainedProblem):
    """Simple toy CEC-style constrained functions."""

    def __init__(self, func_id: int, dim: int):
        bounds = np.array([-5.0, 5.0], dtype=np.float32)
        lower = np.full(dim, bounds[0], dtype=np.float32)
        upper = np.full(dim, bounds[1], dtype=np.float32)
        super().__init__(dim=dim, lower=lower, upper=upper, n_g=2, n_h=1)
        self.func_id = func_id

    def _objective(self, X: np.ndarray) -> np.ndarray:
        if self.func_id == 0:
            return np.sum(X**2, axis=1)
        if self.func_id == 1:
            shift = X - 1.0
            return np.sum(shift**2, axis=1) + 10.0 * np.sum(np.cos(2 * np.pi * X), axis=1)
        shift = X + 0.5
        return np.sum(np.abs(shift), axis=1) + np.sum(shift**4, axis=1)

    def evaluate(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        X = np.atleast_2d(X)
        f = self._objective(X)
        sum_x = np.sum(X, axis=1)
        sum_sq = np.sum(X**2, axis=1)
        g1 = sum_x - 0.25 * self.dim
        g2 = 0.15 * self.dim - sum_x
        h1 = sum_sq - 0.5 * self.dim
        return {
            "f": f,
            "g": np.stack([g1, g2], axis=1),
            "h": h1[:, None],
        }


def make_problem(func_id: int, dim: int) -> ToyCEC:
    """Factory returning a ToyCEC problem."""
    return ToyCEC(func_id=func_id, dim=dim)

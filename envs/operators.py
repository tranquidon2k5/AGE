"""Mutation operators for AOS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class Operator:
    """Base class for population operators."""

    name: str

    def __call__(self, X: np.ndarray, rng: np.random.Generator, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class GaussianMutation(Operator):
    sigma: float

    def __call__(self, X: np.ndarray, rng: np.random.Generator, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        noise = rng.normal(scale=self.sigma, size=X.shape)
        mutated = X + noise
        return np.clip(mutated, lower, upper)


def default_operator_set() -> List[Operator]:
    """Return the default operator suite."""
    return [
        GaussianMutation(name="gauss_small", sigma=0.1),
        GaussianMutation(name="gauss_large", sigma=0.5),
    ]

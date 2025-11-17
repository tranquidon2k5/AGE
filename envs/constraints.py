"""Utility helpers for constraint handling."""

from __future__ import annotations

import numpy as np


def total_violation(g: np.ndarray | None, h: np.ndarray | None, eps: float = 1e-4) -> np.ndarray:
    """Return total violation per individual."""
    total = 0.0
    if g is not None and g.size > 0:
        total = total + np.sum(np.clip(g, 0.0, None), axis=1)
    if h is not None and h.size > 0:
        total = total + np.sum(np.clip(np.abs(h) - eps, 0.0, None), axis=1)
    return total


def rank_by_feasibility(f: np.ndarray, violation: np.ndarray) -> np.ndarray:
    """Rank individuals according to feasibility rules."""
    feasible = violation <= 0.0 + 1e-8
    feasible_idx = np.where(feasible)[0]
    infeasible_idx = np.where(~feasible)[0]
    order_feasible = feasible_idx[np.argsort(f[feasible_idx])] if feasible_idx.size else np.empty(0, dtype=int)
    order_infeasible = (
        infeasible_idx[np.argsort(violation[infeasible_idx])]
        if infeasible_idx.size
        else np.empty(0, dtype=int)
    )
    return np.concatenate([order_feasible, order_infeasible])

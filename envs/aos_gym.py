"""Adaptive Operator Selection gym environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constraints import rank_by_feasibility, total_violation
from .operators import Operator, default_operator_set
from .problems import ConstrainedProblem


@dataclass
class AOSConfig:
    problem: ConstrainedProblem
    pop_size: int = 50
    offspring_size: int = 50
    fe_budget: int = 10000
    seed: int = 0
    operators: Optional[List[Operator]] = None


@dataclass
class AOSGym(AOSConfig):
    """Simple evolutionary RL environment."""

    rng: np.random.Generator = field(init=False)
    population: np.ndarray = field(init=False)
    f: np.ndarray = field(init=False)
    g: np.ndarray = field(init=False)
    h: np.ndarray = field(init=False)
    violation: np.ndarray = field(init=False)
    fe_evals: int = field(init=False, default=0)
    best_feasible: Optional[float] = field(init=False, default=None)
    min_violation: float = field(init=False, default=np.inf)
    done: bool = field(init=False, default=False)
    last_action: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.operators is None:
            self.operators = default_operator_set()
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def evaluate(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        res = self.problem.evaluate(X)
        n = X.shape[0]
        g = res.get("g")
        h = res.get("h")
        if g is None or g.size == 0:
            g = np.zeros((n, 0), dtype=np.float32)
        if h is None or h.size == 0:
            h = np.zeros((n, 0), dtype=np.float32)
        return {"f": res["f"].astype(np.float32), "g": g.astype(np.float32), "h": h.astype(np.float32)}

    def reset(self) -> Dict[str, np.ndarray]:
        self.rng = np.random.default_rng(self.seed)
        self.population = self.problem.sample(self.rng, self.pop_size).astype(np.float32)
        res = self.evaluate(self.population)
        self.f = res["f"]
        self.g = res["g"]
        self.h = res["h"]
        self.violation = total_violation(self.g, self.h)
        self.fe_evals = self.pop_size
        self.min_violation = float(np.min(self.violation))
        feasible_mask = self.violation <= 0.0 + 1e-8
        self.best_feasible = float(np.min(self.f[feasible_mask])) if np.any(feasible_mask) else None
        self.done = self.fe_evals >= self.fe_budget
        self.last_action = np.ones(len(self.operators), dtype=np.float32) / len(self.operators)
        return self._get_obs()

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        mean = np.mean(arr)
        std = np.std(arr)
        return (arr - mean) / (std + 1e-8)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        f_norm = self._normalize(self.f)
        v_norm = self._normalize(self.violation)
        pop_features = np.concatenate(
            [self.population, f_norm[:, None].astype(np.float32), v_norm[:, None].astype(np.float32)],
            axis=1,
        )
        return {
            "pop": pop_features.astype(np.float32),
            "meta": {
                "fe_used": self.fe_evals,
                "best_feasible": self.best_feasible,
                "min_violation": self.min_violation,
                "last_action": None if self.last_action is None else self.last_action.copy(),
            },
        }

    def _select(self, X: np.ndarray, res: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        violation = total_violation(res["g"], res["h"])
        order = rank_by_feasibility(res["f"], violation)
        keep = order[: self.pop_size]
        return X[keep], {"f": res["f"][keep], "g": res["g"][keep], "h": res["h"][keep], "v": violation[keep]}

    def _offspring_counts(self, probs: np.ndarray, total: int) -> np.ndarray:
        if total <= 0:
            return np.zeros(len(self.operators), dtype=int)
        probs = probs / np.clip(np.sum(probs), 1e-8, None)
        return self.rng.multinomial(total, probs)

    def _apply_offspring(self, offspring: np.ndarray) -> float:
        if len(offspring):
            res_off = self.evaluate(offspring)
        else:
            res_off = {
                "f": np.empty(0, dtype=np.float32),
                "g": np.zeros((0, self.g.shape[1]), dtype=np.float32),
                "h": np.zeros((0, self.h.shape[1]), dtype=np.float32),
            }
        self.fe_evals += len(offspring)
        combined_X = np.concatenate([self.population, offspring], axis=0)
        combined_f = np.concatenate([self.f, res_off["f"]])
        combined_g = np.concatenate([self.g, res_off["g"]], axis=0)
        combined_h = np.concatenate([self.h, res_off["h"]], axis=0)
        selected_X, selected_metrics = self._select(
            combined_X, {"f": combined_f, "g": combined_g, "h": combined_h}
        )
        prev_min_violation = self.min_violation
        prev_best_feasible = self.best_feasible
        prev_has_feasible = prev_best_feasible is not None
        self.population = selected_X
        self.f = selected_metrics["f"]
        self.g = selected_metrics["g"]
        self.h = selected_metrics["h"]
        self.violation = selected_metrics["v"]
        self.min_violation = float(np.min(self.violation)) if len(self.violation) else prev_min_violation
        feasible_mask = self.violation <= 0.0 + 1e-8
        if np.any(feasible_mask):
            best_now = float(np.min(self.f[feasible_mask]))
            self.best_feasible = best_now if self.best_feasible is None else min(self.best_feasible, best_now)
        self.done = self.fe_evals >= self.fe_budget
        reward = 0.0
        has_feasible = self.best_feasible is not None
        if not prev_has_feasible:
            reward += 0.1 * max(0.0, (prev_min_violation - self.min_violation))
        if not prev_has_feasible and has_feasible:
            reward += 1.0
        elif prev_has_feasible and has_feasible and self.best_feasible is not None and prev_best_feasible is not None:
            if self.best_feasible < prev_best_feasible - 1e-8:
                reward += 1.0
        self.done = self.fe_evals >= self.fe_budget
        return float(reward)

    def step(self, probs: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        if self.done:
            return self._get_obs(), 0.0, True, {"reason": "budget_exhausted"}
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim != 1 or probs.size != len(self.operators):
            raise ValueError("Action must be 1-D with size equal to number of operators.")
        probs = np.clip(probs, 1e-6, None)
        probs = probs / probs.sum()
        remaining = max(0, self.fe_budget - self.fe_evals)
        if remaining == 0:
            self.done = True
            return self._get_obs(), 0.0, True, {"reason": "budget_exhausted"}
        n_offspring = min(self.offspring_size, remaining)
        counts = self._offspring_counts(probs, n_offspring)
        offspring_list = []
        for op, count in zip(self.operators, counts):
            if count <= 0:
                continue
            parents = self.population[self.rng.integers(0, self.pop_size, size=count)]
            mutated = op(parents, self.rng, self.problem.lower, self.problem.upper)
            offspring_list.append(mutated)
        if offspring_list:
            offspring = np.concatenate(offspring_list, axis=0)
        else:
            offspring = np.empty((0, self.problem.dim), dtype=np.float32)
        reward = self._apply_offspring(offspring)
        self.last_action = probs
        info = {
            "fe_used": self.fe_evals,
            "best_feasible": self.best_feasible,
            "min_violation": self.min_violation,
            "n_offspring": len(offspring),
        }
        return self._get_obs(), float(reward), self.done, info

"""Classical baselines for comparison."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .aos_gym import AOSGym


class _BaselineLogger:
    def __init__(self, env: AOSGym):
        self.env = env
        self.history: Dict[str, List[float]] = {
            "fe": [],
            "best_feasible": [],
            "min_violation": [],
            "reward": [],
        }
        self.actions: List[np.ndarray] = []

    def _log(self, reward: float, action_vec: Optional[np.ndarray]) -> None:
        best = self.env.best_feasible if self.env.best_feasible is not None else np.nan
        self.history["fe"].append(float(self.env.fe_evals))
        self.history["best_feasible"].append(float(best))
        self.history["min_violation"].append(float(self.env.min_violation))
        self.history["reward"].append(float(reward))
        if action_vec is None:
            action_vec = np.full(len(self.env.operators), np.nan, dtype=np.float32)
        self.actions.append(action_vec.astype(np.float32))

    def results(self) -> Dict[str, np.ndarray]:
        return {
            "fe": np.array(self.history["fe"], dtype=np.float32),
            "best_feasible": np.array(self.history["best_feasible"], dtype=np.float32),
            "min_violation": np.array(self.history["min_violation"], dtype=np.float32),
            "reward": np.array(self.history["reward"], dtype=np.float32),
            "p": np.stack(self.actions, axis=0) if self.actions else np.zeros((0, len(self.env.operators))),
        }


class RandomSearch(_BaselineLogger):
    """Generate offspring uniformly at random within the bounds."""

    def __init__(self, env: AOSGym, steps_per_gen: int):
        super().__init__(env)
        self.steps_per_gen = steps_per_gen
        self.rng = np.random.default_rng(env.seed)

    def run(self) -> Dict[str, np.ndarray]:
        self.env.reset()
        while not self.env.done:
            remaining = self.env.fe_budget - self.env.fe_evals
            if remaining <= 0:
                break
            n = min(self.steps_per_gen, remaining)
            offspring = self.rng.uniform(self.env.problem.lower, self.env.problem.upper, size=(n, self.env.problem.dim))
            reward = self.env._apply_offspring(offspring.astype(np.float32))
            self.env.last_action = np.full(len(self.env.operators), np.nan, dtype=np.float32)
            self._log(reward, None)
        return self.results()


class DE_rand1_bin(_BaselineLogger):
    """Minimal Differential Evolution with rand/1/bin strategy."""

    def __init__(self, env: AOSGym, F: float = 0.5, Cr: float = 0.9):
        super().__init__(env)
        self.F = F
        self.Cr = Cr
        self.rng = np.random.default_rng(env.seed + 1337)

    def _trial_vector(self, idx: int) -> np.ndarray:
        pop = self.env.population
        dim = pop.shape[1]
        if pop.shape[0] < 4:
            return pop[idx].copy()
        available = [j for j in range(pop.shape[0]) if j != idx]
        r1, r2, r3 = self.rng.choice(available, size=3, replace=False)
        mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
        cross = self.rng.random(dim) < self.Cr
        if not np.any(cross):
            cross[self.rng.integers(0, dim)] = True
        trial = np.where(cross, mutant, pop[idx])
        return np.clip(trial, self.env.problem.lower, self.env.problem.upper)

    def run(self) -> Dict[str, np.ndarray]:
        self.env.reset()
        while not self.env.done:
            remaining = self.env.fe_budget - self.env.fe_evals
            if remaining <= 0:
                break
            n = min(self.env.pop_size, remaining)
            trials = [self._trial_vector(i) for i in range(n)]
            offspring = np.stack(trials, axis=0).astype(np.float32)
            reward = self.env._apply_offspring(offspring)
            self.env.last_action = np.full(len(self.env.operators), np.nan, dtype=np.float32)
            self._log(reward, None)
        return self.results()

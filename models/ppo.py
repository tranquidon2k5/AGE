"""PPO with Dirichlet policy for Adaptive Operator Selection."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Dirichlet

from .deepsets import DeepSetsEncoder


class DirichletActor(nn.Module):
    """Outputs concentration parameters for a Dirichlet distribution."""

    def __init__(self, s_dim: int, action_dim: int, eps: float = 1e-3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, s_dim),
            nn.ReLU(),
            nn.Linear(s_dim, action_dim),
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.net(x)
        return torch.nn.functional.softplus(logits) + self.eps


class Critic(nn.Module):
    def __init__(self, s_dim: int) -> None:
        super().__init__()
        self.v = nn.Sequential(nn.Linear(s_dim, s_dim), nn.ReLU(), nn.Linear(s_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.v(x).squeeze(-1)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    epochs: int = 5
    minibatch_size: int = 32
    max_grad_norm: float = 0.5
    kl_target: float = 0.01
    rollout_horizon: int = 64


class RolloutBuffer:
    def __init__(self) -> None:
        self.pop_states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def add(
        self,
        obs_pop: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.pop_states.append(obs_pop.astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def compute_returns(self, last_value: float, gamma: float, lam: float) -> None:
        size = len(self.rewards)
        advantages = np.zeros(size, dtype=np.float32)
        last_adv = 0.0
        next_value = last_value
        for t in reversed(range(size)):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            last_adv = delta + gamma * lam * mask * last_adv
            advantages[t] = last_adv
            next_value = self.values[t]
        returns = advantages + np.array(self.values, dtype=np.float32)
        self.advantages = advantages
        self.returns = returns

    def __len__(self) -> int:
        return len(self.rewards)

    def iter_batches(self, batch_size: int) -> Iterator[np.ndarray]:
        size = len(self)
        if size == 0:
            return
            yield  # pragma: no cover
        indices = np.arange(size)
        np.random.shuffle(indices)
        actual_size = batch_size if batch_size > 0 else size
        if actual_size >= size:
            yield indices
            return
        for start in range(0, size, actual_size):
            yield indices[start : start + actual_size]


class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = DeepSetsEncoder(obs_dim).to(self.device)
        latent_dim = 256
        self.actor = DirichletActor(latent_dim, action_dim).to(self.device)
        self.critic = Critic(latent_dim).to(self.device)
        params = list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.config.lr)

    def _encode(self, pop_np: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(pop_np).float().unsqueeze(0).to(self.device)
        return self.encoder(tensor)

    def _distribution(self, features: torch.Tensor) -> Dirichlet:
        alpha = self.actor(features)
        return Dirichlet(alpha)

    def act(self, pop_np: np.ndarray) -> Tuple[np.ndarray, float, float]:
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            features = self._encode(pop_np)
            dist = self._distribution(features)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(features)
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def value(self, pop_np: np.ndarray) -> float:
        self.encoder.eval()
        self.critic.eval()
        with torch.no_grad():
            features = self._encode(pop_np)
            value = self.critic(features)
        return float(value.item())

    def policy_mean(self, pop_np: np.ndarray) -> np.ndarray:
        self.encoder.eval()
        self.actor.eval()
        with torch.no_grad():
            features = self._encode(pop_np)
            dist = self._distribution(features)
            alpha = dist.concentration
            mean = alpha / alpha.sum(dim=-1, keepdim=True)
        return mean.squeeze(0).cpu().numpy()

    def collect_rollout(
        self,
        env,
        obs: Dict[str, np.ndarray],
    ) -> Tuple[RolloutBuffer, Dict[str, np.ndarray], bool]:
        buffer = RolloutBuffer()
        for _ in range(self.config.rollout_horizon):
            if env.done:
                break
            action, logp, value = self.act(obs["pop"])
            next_obs, reward, done, _ = env.step(action)
            buffer.add(obs["pop"], action, logp, value, reward, done)
            obs = next_obs
            if done:
                break
        last_value = 0.0 if env.done else self.value(obs["pop"])
        if len(buffer) > 0:
            buffer.compute_returns(last_value, self.config.gamma, self.config.lam)
        return buffer, obs, env.done

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        if len(buffer) == 0 or buffer.advantages is None or buffer.returns is None:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "steps": 0}
        obs_tensor = torch.from_numpy(np.stack(buffer.pop_states)).float().to(self.device)
        actions_tensor = torch.from_numpy(np.stack(buffer.actions)).float().to(self.device)
        logp_tensor = torch.from_numpy(np.array(buffer.log_probs)).float().to(self.device)
        adv_tensor = torch.from_numpy(buffer.advantages).float().to(self.device)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        ret_tensor = torch.from_numpy(buffer.returns).float().to(self.device)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "steps": float(len(buffer))}
        n_updates = 0
        for _ in range(self.config.epochs):
            stop = False
            for indices in buffer.iter_batches(self.config.minibatch_size):
                batch_obs = obs_tensor[indices]
                batch_actions = actions_tensor[indices]
                batch_old_logp = logp_tensor[indices]
                batch_adv = adv_tensor[indices]
                batch_returns = ret_tensor[indices]
                features = self.encoder(batch_obs)
                dist = self._distribution(features)
                new_logp = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logp - batch_old_logp)
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
                policy_loss = -(torch.min(ratio * batch_adv, clipped * batch_adv)).mean()
                values = self.critic(features)
                value_loss = nn.functional.mse_loss(values, batch_returns)
                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                approx_kl = torch.clamp((batch_old_logp - new_logp).mean(), min=0.0)
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["approx_kl"] = float(approx_kl.item())
                n_updates += 1
                if approx_kl.item() > self.config.kl_target:
                    stop = True
                    break
            if stop:
                break
        if n_updates > 0:
            stats["policy_loss"] /= n_updates
            stats["value_loss"] /= n_updates
            stats["entropy"] /= n_updates
        return stats

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())

    def save_checkpoint(self, path: str) -> None:
        state = {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "np_rng": np.random.get_state(),
        }
        tmp_path = f"{path}.tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(state["encoder"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["np_rng"])


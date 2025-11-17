"""Training script for PPO-Dirichlet on toy CEC problems."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from envs.aos_gym import AOSGym
from envs.problems import make_problem
from models.ppo import PPOAgent, PPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent for adaptive operator selection.")
    parser.add_argument("--func_id", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--fe_budget", type=int, default=5000)
    parser.add_argument("--logdir", type=Path, default=Path("logs"))
    parser.add_argument("--ckpt_dir", type=Path, default=Path("ckpt"))
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_logs(out_path: Path, records: Dict[str, List[float]], probs: List[np.ndarray], meta: Dict) -> None:
    if probs:
        p_arr = np.stack(probs, axis=0).astype(np.float32)
    else:
        p_arr = np.zeros((0, 0), dtype=np.float32)
    np.savez(
        out_path,
        fe=np.array(records["fe"], dtype=np.float32),
        best_feasible=np.array(records["best_feasible"], dtype=np.float32),
        min_violation=np.array(records["min_violation"], dtype=np.float32),
        reward=np.array(records["reward"], dtype=np.float32),
        p=p_arr,
        meta=json.dumps(meta),
    )


def main() -> None:
    args = parse_args()
    ensure_dir(args.logdir)
    ensure_dir(args.ckpt_dir)
    problem = make_problem(args.func_id, args.dim)
    env = AOSGym(
        problem=problem,
        pop_size=50,
        offspring_size=50,
        fe_budget=args.fe_budget,
        seed=args.seed,
    )
    obs = env.reset()
    obs_dim = obs["pop"].shape[-1]
    action_dim = len(env.operators)
    config = PPOConfig()
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config)
    ckpt_path = args.ckpt_dir / f"func{args.func_id}_D{args.dim}_seed{args.seed}.pt"
    if args.resume and ckpt_path.exists():
        agent.load_checkpoint(str(ckpt_path))
        print(f"Resumed agent from {ckpt_path}")
    history = {"fe": [], "best_feasible": [], "min_violation": [], "reward": []}
    probs: List[np.ndarray] = []
    for update_idx in range(1, args.updates + 1):
        buffer, obs, done_env = agent.collect_rollout(env, obs)
        stats = agent.update(buffer)
        reward_mean = float(np.mean(buffer.rewards)) if buffer.rewards else 0.0
        best = env.best_feasible if env.best_feasible is not None else np.nan
        probs_now = agent.policy_mean(obs["pop"])
        history["fe"].append(float(env.fe_evals))
        history["best_feasible"].append(float(best))
        history["min_violation"].append(float(env.min_violation))
        history["reward"].append(reward_mean)
        probs.append(probs_now.astype(np.float32))
        print(
            f"[Update {update_idx}] FE={env.fe_evals} best={best} "
            f"vmin={env.min_violation:.4f} reward={reward_mean:.3f} "
            f"KL={stats['approx_kl']:.4f} p={np.array2string(probs_now, precision=3)}"
        )
        if (update_idx % args.save_every == 0) or done_env:
            agent.save_checkpoint(str(ckpt_path))
        if done_env:
            break
    log_path = args.logdir / f"func{args.func_id}_D{args.dim}_seed{args.seed}.npz"
    meta = {
        "func_id": args.func_id,
        "dim": args.dim,
        "seed": args.seed,
        "fe_budget": args.fe_budget,
        "updates": args.updates,
    }
    save_logs(log_path, history, probs, meta)
    print(f"Logs stored in {log_path}")


if __name__ == "__main__":
    main()

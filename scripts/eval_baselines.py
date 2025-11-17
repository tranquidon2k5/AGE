"""Evaluate classical baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from envs.aos_gym import AOSGym
from envs.baselines import DE_rand1_bin, RandomSearch
from envs.problems import make_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baselines.")
    parser.add_argument("--algo", choices=["random", "de"], required=True)
    parser.add_argument("--func_id", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fe_budget", type=int, default=5000)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def save_npz(path: Path, payload: dict, meta: dict) -> None:
    np.savez(
        path,
        fe=payload["fe"],
        best_feasible=payload["best_feasible"],
        min_violation=payload["min_violation"],
        reward=payload["reward"],
        p=payload["p"],
        meta=json.dumps(meta),
    )


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    problem = make_problem(args.func_id, args.dim)
    env = AOSGym(
        problem=problem,
        pop_size=50,
        offspring_size=50,
        fe_budget=args.fe_budget,
        seed=args.seed,
    )
    if args.algo == "random":
        runner = RandomSearch(env, steps_per_gen=env.offspring_size)
    else:
        runner = DE_rand1_bin(env)
    payload = runner.run()
    meta = {
        "algo": args.algo,
        "func_id": args.func_id,
        "dim": args.dim,
        "seed": args.seed,
        "fe_budget": args.fe_budget,
    }
    save_npz(args.out, payload, meta)
    print(f"Baseline results stored in {args.out}")


if __name__ == "__main__":
    main()

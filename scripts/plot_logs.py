"""Plot training or baseline logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot log summaries.")
    parser.add_argument("--inp", type=Path, required=True, help="Path to .npz log file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.inp, allow_pickle=True)
    fe = data["fe"]
    best = data["best_feasible"]
    vmin = data["min_violation"]
    p = data["p"]
    meta_raw = data["meta"].item() if data["meta"].shape == () else data["meta"]
    try:
        meta = json.loads(meta_raw)
    except Exception:
        meta = {}
    title = f"func {meta.get('func_id','?')} dim {meta.get('dim','?')} seed {meta.get('seed','?')}"

    plt.figure()
    plt.plot(fe, vmin)
    plt.xlabel("Function evaluations")
    plt.ylabel("Min violation")
    plt.title(f"Violation trend ({title})")

    plt.figure()
    plt.plot(fe, best)
    plt.xlabel("Function evaluations")
    plt.ylabel("Best feasible f")
    plt.title(f"Best feasible trend ({title})")

    if p.size > 0:
        plt.figure()
        updates = np.arange(p.shape[0]) + 1
        for k in range(p.shape[1]):
            plt.plot(updates, p[:, k], label=f"op{k}")
        plt.xlabel("Update")
        plt.ylabel("Probability")
        plt.title(f"Operator ratios ({title})")
        plt.legend()

    plt.show()


if __name__ == "__main__":
    main()

## AOS-RL

Adaptive Operator Selection (AOS) playground for constrained CEC-style optimization. A PPO-Dirichlet agent learns to mix mutation operators using DeepSets state encoding. Baseline methods (Random Search, Differential Evolution) are provided for comparison.

### Requirements

- Python 3.10+
- `pip install numpy torch matplotlib pytest`

### Project Layout

- `envs/`: benchmark problems, constraint utilities, operators, evolutionary gym, baselines.
- `models/`: DeepSets encoder and PPO agent with Dirichlet policy.
- `scripts/`: training / evaluation CLI entry-points plus plotting helper.
- `tests/`: unit tests covering constraints, encoder invariance, and PPO shapes.

### Quick Start

All commands below work on Linux/macOS shells and PowerShell on Windows.

```bash
pip install numpy torch matplotlib pytest

# Train PPO-Dirichlet (toy setup)
python -m scripts.train_cec --func_id 0 --dim 10 --seed 1 --updates 10 --fe_budget 5000 --logdir logs --ckpt_dir ckpt --save_every 2 --resume

# Plot logged curves
python scripts/plot_logs.py --inp logs/func0_D10_seed1.npz

# Baseline (random search)
python -m scripts.eval_baselines --algo random --func_id 0 --dim 10 --seed 1 --fe_budget 5000 --out logs/random_func0_D10_seed1.npz

# Unit tests
pytest -q
```

When running scripts directly (e.g., `python scripts/train_cec.py ...`), an automatic `sys.path` fallback adds the repository root, so the same commands also work outside `-m` invocations. Check the saved `.npz` logs and use `scripts/plot_logs.py` to compare PPO with the baselines.

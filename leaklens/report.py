"""Aggregate per-seed GE curves into a verdict JSON + GE-curve PNG."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _n_star(ge: np.ndarray) -> int | None:
    """Smallest N (1-based) where GE(N) <= 1; None if never reached."""
    hits = np.where(ge <= 1.0 + 1e-9)[0]
    return None if hits.size == 0 else int(hits[0] + 1)


def build_verdict(ge_curves: np.ndarray, *, dataset: str, target_byte: int,
                  model: str, seeds: list[int]) -> dict:
    """ge_curves: (n_seeds, n_traces) — mean-over-folds GE per seed."""
    n_stars = [_n_star(ge_curves[i]) for i in range(ge_curves.shape[0])]
    valid = [n for n in n_stars if n is not None]
    n_star_mean = float(np.mean(valid)) if valid else None
    pass_main = n_star_mean is not None and n_star_mean <= 1000
    return {
        "dataset": dataset,
        "target_byte_index": target_byte,
        "leakage_model": "identity",
        "model": model,
        "seeds": seeds,
        "metrics": {
            "N_star_GE1": {
                "mean": n_star_mean,
                "std":  float(np.std(valid)) if valid else None,
                "per_seed": n_stars,
            },
            "GE_at_1000": {
                "mean": float(ge_curves[:, -1].mean()),
                "std":  float(ge_curves[:, -1].std()),
            },
        },
        "pass_main":    pass_main,
        "pass_stretch": n_star_mean is not None and n_star_mean < 300,
    }


def plot_ge_curve(ge_curves: np.ndarray, out: Path) -> None:
    mean = ge_curves.mean(axis=0)
    std  = ge_curves.std(axis=0)
    x = np.arange(1, ge_curves.shape[1] + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(x, mean, label="mean GE")
    plt.fill_between(x, mean - std, mean + std, alpha=0.3, label="±1σ")
    plt.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="GE = 1")
    plt.xlabel("Attack traces"); plt.ylabel("Guessing entropy")
    plt.yscale("log"); plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()

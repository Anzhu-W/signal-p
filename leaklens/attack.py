"""Key-recovery attack: rank + guessing-entropy curve.

Critical implementation notes:
- Likelihoods accumulated in log space (sum of log-softmax), NEVER as product of softmax.
- Rank is 1-indexed: rank 1 = best guess is the true key.
- Trace order is preserved (no shuffling within a fold).
"""
import numpy as np


def compute_rank(true_key: int, log_probs: np.ndarray) -> int:
    """log_probs: shape (n_classes,). Returns rank (1..n_classes)."""
    order = np.argsort(-log_probs, kind="stable")
    return int(np.where(order == true_key)[0][0]) + 1


def guessing_entropy_curve(fold_log_probs: np.ndarray, true_keys: np.ndarray) -> np.ndarray:
    """
    fold_log_probs: (n_folds, n_traces, n_classes) — per-trace log-likelihood per candidate.
    true_keys:      (n_folds,) — the correct key byte per fold.
    Returns:        (n_traces,) — mean GE across folds at each cumulative trace count.
    """
    n_folds, n_traces, _ = fold_log_probs.shape
    cum = np.cumsum(fold_log_probs, axis=1)
    ranks = np.zeros((n_folds, n_traces))
    for f in range(n_folds):
        for t in range(n_traces):
            ranks[f, t] = compute_rank(int(true_keys[f]), cum[f, t])
    return ranks.mean(axis=0)

import numpy as np
from leaklens.attack import compute_rank, guessing_entropy_curve


def test_rank_is_one_indexed():
    log_probs = np.array([-1.0, -0.1, -2.0])  # candidate 1 most likely
    assert compute_rank(true_key=1, log_probs=log_probs) == 1
    assert compute_rank(true_key=0, log_probs=log_probs) == 2
    assert compute_rank(true_key=2, log_probs=log_probs) == 3


def test_uniform_logits_random_ge():
    """Small Gaussian noise per trace per candidate → ranks scatter uniformly,
    so cumulative GE drifts to ~128 (mean rank over 256 classes)."""
    rng = np.random.default_rng(0)
    fold_log_probs = rng.normal(size=(100, 1000, 256))
    true_keys = rng.integers(0, 256, size=100)
    ge = guessing_entropy_curve(fold_log_probs, true_keys)
    assert 100 < ge[-1] < 160  # ~128 ± noise


def test_oracle_gives_ge_one():
    n_folds, n_traces, k = 5, 100, 256
    true_keys = np.array([42, 7, 200, 0, 255])
    log_probs = np.full((n_folds, n_traces, k), -np.inf)
    for f, true_k in enumerate(true_keys):
        log_probs[f, :, true_k] = 0.0
    ge = guessing_entropy_curve(log_probs, true_keys)
    assert np.allclose(ge, 1.0)

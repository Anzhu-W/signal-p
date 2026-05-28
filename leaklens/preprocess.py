"""Per-feature standardization. Train-only fit."""
import numpy as np


class Standardizer:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_:  np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_  = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Call fit() first"
        return ((X - self.mean_) / self.std_).astype(np.float32)

"""HDF5 loader for ASCADv1 fixed-key."""
from pathlib import Path
import h5py
import numpy as np


class ASCADSource:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self, split: str):
        assert split in {"profiling", "attack"}
        group = "Profiling_traces" if split == "profiling" else "Attack_traces"
        with h5py.File(self.path, "r") as f:
            traces = f[f"{group}/traces"][:].astype(np.float32)
            labels = f[f"{group}/labels"][:].astype(np.int64)
            meta_raw = f[f"{group}/metadata"][:]
        meta = {
            "plaintext": np.stack([m["plaintext"] for m in meta_raw]),
            "key":       np.stack([m["key"]       for m in meta_raw]),
        }
        return traces, labels, meta

"""Run the key-recovery attack on trained CNN checkpoints.

For each checkpoint:
  1. Load model + Standardizer state from .pt
  2. Forward attack traces -> log_softmax (10000, 256)
  3. For each candidate key byte k:  score(t, k) = log_p[t, SBox[pt_attack[t,2] ^ k]]
  4. Reshape into 10 folds of 1000 traces -> mean GE curve
  5. Save reports/ge_curve_seed{N}.npy

Then `leaklens report --ge-curves ... --out reports/verdict.json` finalizes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from leaklens import get_device
from leaklens.sources.ascad import ASCADSource
from leaklens.labels import ascad_identity_label, TARGET_BYTE_INDEX
from leaklens.aes import SBOX
from leaklens.models import BenadjilaCNNBest
from leaklens.attack import guessing_entropy_curve


def attack_one(checkpoint_path: Path, attack_traces: np.ndarray, pt_attack: np.ndarray,
               key_byte_true: int, device: str, n_folds: int = 10) -> np.ndarray:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Restore Standardizer
    mean = ckpt["standardizer_mean"]
    std  = ckpt["standardizer_std"]
    at_z = ((attack_traces - mean) / std).astype(np.float32)

    # Load model
    model = BenadjilaCNNBest().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Forward pass in batches (66M params + 10k traces)
    batch_size = 200
    log_p_chunks = []
    with torch.no_grad():
        for i in range(0, len(at_z), batch_size):
            xb = torch.from_numpy(at_z[i:i + batch_size]).to(device)
            log_p_chunks.append(torch.log_softmax(model(xb), dim=1).cpu().numpy())
    log_p = np.concatenate(log_p_chunks, axis=0)  # (10000, 256)

    # score(t, k) = log_p[t, SBox[plaintext[t, 2] XOR k]]
    pt2 = pt_attack[:, TARGET_BYTE_INDEX]
    n_traces, n_classes = log_p.shape
    scores = np.empty((n_traces, n_classes), dtype=np.float64)
    for k in range(n_classes):
        scores[:, k] = log_p[np.arange(n_traces), SBOX[pt2 ^ k]]

    # 10 folds of 1000 traces, each fold uses the SAME true key (fixed-key dataset)
    fold_scores = scores.reshape(n_folds, n_traces // n_folds, n_classes)
    true_keys = np.full(n_folds, int(key_byte_true), dtype=int)
    return guessing_entropy_curve(fold_scores, true_keys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   type=Path, help="Paths to benadjila_seed*.pt")
    p.add_argument("--data", type=Path, default=Path("data/ASCAD.h5"))
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    p.add_argument("--n-folds", type=int, default=10)
    args = p.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    device = get_device()
    print(f"device: {device}")

    src = ASCADSource(args.data)
    attack_traces, _, at_meta = src.load("attack")
    pt_attack = at_meta["plaintext"]
    key_byte_true = int(at_meta["key"][0, TARGET_BYTE_INDEX])
    print(f"attack traces: {attack_traces.shape}   true key byte {TARGET_BYTE_INDEX}: 0x{key_byte_true:02X}")

    for ckpt_path in args.checkpoints:
        seed = int(ckpt_path.stem.split("seed")[-1])
        print(f"\n--- seed {seed} ({ckpt_path.name}) ---")
        ge = attack_one(ckpt_path, attack_traces, pt_attack,
                        key_byte_true, device, n_folds=args.n_folds)
        out = args.out_dir / f"ge_curve_seed{seed}.npy"
        np.save(out, ge)
        # Quick summary
        n_traces = len(ge)
        n_star = next((i + 1 for i, v in enumerate(ge) if v <= 1.0 + 1e-9), None)
        print(f"  GE[1]    = {ge[0]:.2f}")
        print(f"  GE[100]  = {ge[99]:.2f}" if n_traces >= 100 else "")
        print(f"  GE[500]  = {ge[499]:.2f}" if n_traces >= 500 else "")
        print(f"  GE[end={n_traces}] = {ge[-1]:.2f}")
        print(f"  N* (first N where GE<=1): {n_star}")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()

"""Train BenadjilaCNNBest on ASCADv1 fixed-key, save checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from leaklens import get_device
from leaklens.seed import set_global_seed
from leaklens.sources.ascad import ASCADSource
from leaklens.labels import ascad_identity_label
from leaklens.preprocess import Standardizer
from leaklens.models import BenadjilaCNNBest
from leaklens.train import train_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--data", type=Path, default=Path("data/ASCAD.h5"))
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    args = p.parse_args()
    args.out.mkdir(exist_ok=True)

    set_global_seed(args.seed)
    device = get_device()
    print(f"device: {device}  seed: {args.seed}  epochs: {args.epochs}")

    src = ASCADSource(args.data)
    tr_traces, _, tr_meta = src.load("profiling")
    y_tr = ascad_identity_label(tr_meta["plaintext"], tr_meta["key"])

    sc = Standardizer().fit(tr_traces[:45000])
    tr_z  = sc.transform(tr_traces[:45000])
    val_z = sc.transform(tr_traces[45000:])

    to_t = lambda a: torch.from_numpy(a).float()
    dl_tr = DataLoader(
        TensorDataset(to_t(tr_z), torch.from_numpy(y_tr[:45000])),
        batch_size=args.batch_size, shuffle=True,
    )
    dl_val = DataLoader(
        TensorDataset(to_t(val_z), torch.from_numpy(y_tr[45000:])),
        batch_size=args.batch_size,
    )

    model = BenadjilaCNNBest()
    history = train_model(
        model, dl_tr, dl_val,
        epochs=args.epochs, lr=args.lr,
        optimizer="rmsprop", device=device,
    )

    ckpt = args.out / f"benadjila_seed{args.seed}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "history": history,
            "standardizer_mean": sc.mean_,
            "standardizer_std":  sc.std_,
        },
        ckpt,
    )
    (args.out / f"history_seed{args.seed}.json").write_text(
        json.dumps(history, default=float)
    )
    print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()

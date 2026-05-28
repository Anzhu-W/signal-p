"""Wait for ASCAD download to finish, extract ASCAD.h5, run the MLP smoke test.

Outputs:
    reports/smoke_history.json     - training/val loss + acc per epoch
    reports/smoke_loss_curve.png   - loss + val-acc plot

Designed to be launched in the background; safe to run twice.
"""
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
ZIP_PATH = DATA / "ASCAD_data.zip"
H5_PATH = DATA / "ASCAD.h5"
EXPECTED_ZIP_SIZE = 4.0 * (1024 ** 3)  # ~4.2 GB; allow a wide tolerance

REPORTS.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_zip() -> None:
    log(f"Waiting for {ZIP_PATH} to reach ~4 GB ...")
    while True:
        if H5_PATH.exists():
            log("ASCAD.h5 already extracted.")
            return
        if not ZIP_PATH.exists():
            log("Zip not present yet.")
            time.sleep(20)
            continue
        size = ZIP_PATH.stat().st_size
        log(f"  zip size = {size / 1e9:.2f} GB")
        # Once the file stops growing for 30s AND is at least 3.5 GB, assume done.
        time.sleep(15)
        new_size = ZIP_PATH.stat().st_size
        if size == new_size and size >= 3.5 * 1e9:
            log("Download appears complete.")
            return


def extract_h5() -> None:
    if H5_PATH.exists():
        log("ASCAD.h5 already present, skipping extraction.")
        return
    log(f"Extracting ASCAD.h5 ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
        # canonical is ASCAD_data/ASCAD_databases/ASCAD.h5 (desync=0 fixed-key)
        candidate = next((n for n in names if n.endswith("/ASCAD.h5")), None)
        if candidate is None:
            log("ERROR: could not find ASCAD.h5 inside zip. Top-level entries:")
            for n in names[:20]:
                log(f"  {n}")
            sys.exit(1)
        log(f"Using inner path: {candidate}")
        with zf.open(candidate) as src, H5_PATH.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
    log(f"Extracted to {H5_PATH}  ({H5_PATH.stat().st_size / 1e6:.1f} MB)")


def run_smoke() -> None:
    log("Running MLP smoke test (5 epochs) ...")
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from leaklens import get_device
    from leaklens.seed import set_global_seed
    from leaklens.sources.ascad import ASCADSource
    from leaklens.labels import ascad_identity_label
    from leaklens.preprocess import Standardizer
    from leaklens.models import MLPBaseline
    from leaklens.train import train_model

    set_global_seed(0)
    device = get_device()
    log(f"  device = {device}")

    src = ASCADSource(H5_PATH)
    tr_traces, _, tr_meta = src.load("profiling")
    log(f"  train traces: {tr_traces.shape}")

    y_tr = ascad_identity_label(tr_meta["plaintext"], tr_meta["key"])
    sc = Standardizer().fit(tr_traces[:45000])
    tr_z = sc.transform(tr_traces[:45000])
    val_z = sc.transform(tr_traces[45000:])

    to_t = lambda a: torch.from_numpy(a).float()
    dl_tr = DataLoader(TensorDataset(to_t(tr_z), torch.from_numpy(y_tr[:45000])),
                       batch_size=200, shuffle=True)
    dl_val = DataLoader(TensorDataset(to_t(val_z), torch.from_numpy(y_tr[45000:])),
                        batch_size=200)

    history = train_model(MLPBaseline(), dl_tr, dl_val,
                          epochs=5, lr=1e-3, optimizer="adam", device=device)

    (REPORTS / "smoke_history.json").write_text(json.dumps(history, default=float))

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3))
    a.plot(history["train_loss"], label="train"); a.plot(history["val_loss"], label="val")
    a.set_xlabel("epoch"); a.set_ylabel("loss"); a.legend(); a.set_title("MLP loss")
    b.plot(history["val_acc"], marker="o")
    b.axhline(1 / 256, color="r", ls="--", label="random (1/256)")
    b.set_xlabel("epoch"); b.set_ylabel("val acc"); b.legend(); b.set_title("MLP val_acc (beats random?)")
    plt.tight_layout(); plt.savefig(REPORTS / "smoke_loss_curve.png", dpi=150); plt.close()
    log(f"Wrote {REPORTS / 'smoke_loss_curve.png'}")


def main() -> int:
    wait_for_zip()
    extract_h5()
    run_smoke()
    log("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

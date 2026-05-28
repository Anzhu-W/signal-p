"""leaklens CLI — train | report.

Note: the `attack` step is currently driven via notebooks (see notebooks/02-smoke-test.ipynb)
because it needs the model checkpoint + ASCAD attack split loaded together. A CLI `attack`
subcommand will be added once the notebook flow stabilises.
"""
import json
import subprocess
from pathlib import Path

import click


@click.group()
def cli():
    """LeakLens: side-channel-analysis pipeline."""


@cli.command()
@click.option("--seed", type=int, default=0)
@click.option("--epochs", type=int, default=75)
def train(seed: int, epochs: int):
    """Train BenadjilaCNNBest for one seed."""
    subprocess.check_call(
        ["python", "scripts/train_cnn.py", "--seed", str(seed), "--epochs", str(epochs)]
    )


@cli.command()
@click.option("--ge-curves", "ge_curves_paths", multiple=True, required=True,
              type=click.Path(exists=True, path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), default=Path("reports/verdict.json"))
def report(ge_curves_paths, out: Path):
    """Build verdict JSON + GE-curve PNG from per-seed numpy arrays."""
    import numpy as np
    from leaklens.report import build_verdict, plot_ge_curve

    curves = np.stack([np.load(p) for p in ge_curves_paths])
    verdict = build_verdict(
        curves,
        dataset="ASCADv1_fixed_desync0",
        target_byte=2,
        model="BenadjilaCNNBest",
        seeds=list(range(len(ge_curves_paths))),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2))
    plot_ge_curve(curves, out.with_suffix(".png"))
    click.echo(
        json.dumps(
            {"pass_main": verdict["pass_main"], "pass_stretch": verdict["pass_stretch"]}
        )
    )


if __name__ == "__main__":
    cli()

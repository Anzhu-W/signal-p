from pathlib import Path
import json
import numpy as np
from leaklens.report import build_verdict, plot_ge_curve


def test_verdict_passes_when_ge_reaches_one_within_1000(tmp_path: Path):
    ge_curves = np.stack([np.linspace(128, 1, 1000)] * 3)
    verdict = build_verdict(
        ge_curves, dataset="ASCADv1_fixed_desync0",
        target_byte=2, model="BenadjilaCNNBest", seeds=[0, 1, 2],
    )
    assert verdict["pass_main"] is True
    assert verdict["metrics"]["N_star_GE1"]["mean"] is not None
    assert verdict["metrics"]["N_star_GE1"]["mean"] > 0
    out = tmp_path / "verdict.json"
    out.write_text(json.dumps(verdict))
    assert json.loads(out.read_text())["model"] == "BenadjilaCNNBest"


def test_plot_ge_curve_makes_png(tmp_path: Path):
    ge_curves = np.stack([np.linspace(128, 1, 1000)] * 3)
    out = tmp_path / "ge.png"
    plot_ge_curve(ge_curves, out)
    assert out.exists() and out.stat().st_size > 5000


def test_pass_main_false_when_never_reaches_one():
    ge_curves = np.stack([np.linspace(128, 10, 1000)] * 3)
    verdict = build_verdict(
        ge_curves, dataset="x", target_byte=2, model="x", seeds=[0],
    )
    assert verdict["pass_main"] is False

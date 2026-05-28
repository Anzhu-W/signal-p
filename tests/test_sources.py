from pathlib import Path
import numpy as np
import pytest
from leaklens.sources.ascad import ASCADSource

DATA = Path(__file__).resolve().parents[1] / "data" / "ASCAD.h5"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="ASCAD.h5 not downloaded")


def test_profiling_shape():
    src = ASCADSource(DATA)
    traces, labels, meta = src.load("profiling")
    assert traces.shape == (50000, 700)
    assert traces.dtype == np.float32
    assert labels.shape == (50000,)
    assert meta["plaintext"].shape == (50000, 16)
    assert meta["key"].shape == (50000, 16)


def test_attack_shape():
    src = ASCADSource(DATA)
    traces, labels, meta = src.load("attack")
    assert traces.shape == (10000, 700)


def test_fixed_key():
    src = ASCADSource(DATA)
    _, _, meta = src.load("attack")
    assert np.all(meta["key"] == meta["key"][0])

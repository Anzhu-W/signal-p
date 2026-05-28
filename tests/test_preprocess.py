import numpy as np
from leaklens.preprocess import Standardizer


def test_fit_transform_train_is_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    train = rng.normal(5.0, 3.0, size=(1000, 700)).astype(np.float32)
    s = Standardizer().fit(train)
    z = s.transform(train)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(z.std(axis=0),  1.0, atol=1e-4)


def test_val_not_forced_to_zero_mean():
    """Critical: Standardizer fit on train must NOT zero-mean val data."""
    rng = np.random.default_rng(1)
    train = rng.normal(0.0, 1.0, size=(1000, 700)).astype(np.float32)
    val   = rng.normal(7.0, 1.0, size=(200,  700)).astype(np.float32)
    s = Standardizer().fit(train)
    val_z = s.transform(val)
    assert abs(val_z.mean()) > 1.0


def test_no_rewindowing():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 700)).astype(np.float32)
    z = Standardizer().fit(x).transform(x)
    assert z.shape == (100, 700)

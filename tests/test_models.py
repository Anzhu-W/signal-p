import torch
from leaklens.models import SCAModel, MLPBaseline, BenadjilaCNNBest


def test_mlp_implements_scamodel():
    m = MLPBaseline()
    assert isinstance(m, SCAModel)
    assert m.n_classes == 256
    assert m.leakage_model == "identity"
    assert m.input_shape == (700,)


def test_mlp_forward_shape():
    m = MLPBaseline()
    x = torch.zeros(8, 700)
    y = m(x)
    assert y.shape == (8, 256)


def test_cnn_implements_scamodel():
    m = BenadjilaCNNBest()
    assert isinstance(m, SCAModel)
    assert m.n_classes == 256
    assert m.leakage_model == "identity"
    assert m.input_shape == (700,)


def test_cnn_forward_shape():
    m = BenadjilaCNNBest()
    x = torch.zeros(4, 1, 700)
    y = m(x)
    assert y.shape == (4, 256)


def test_cnn_param_count_canonical():
    m = BenadjilaCNNBest()
    n_params = sum(p.numel() for p in m.parameters())
    # Reference is ~67M; allow wide tolerance for padding/bias choices
    assert 50_000_000 < n_params < 90_000_000, f"got {n_params:,}"

import warnings

import numpy as np
import pytest

# The neural estimators are an optional extra; skip the whole module if their
# dependencies are not installed.
torch = pytest.importorskip("torch")
pytest.importorskip("nflows")

from syntropy.neural import mutual_information, total_correlation
from syntropy.neural.multivariate_mi import higher_order_information

# Fast configuration: a tiny flow trained for a couple of epochs. The default
# tests below check structural properties (shapes, finiteness, the local/average
# identity, and regression guards) that hold regardless of training quality, so
# they do not need convergence and run in ~1 second.
FAST_FLOW = {"num_layers": 2, "hidden_features": 8}
FAST_TRAIN = {"num_epochs": 2}
FAST_N = 256


def _toy_data(n=FAST_N, d=3, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, d)


# ---------------------------------------------------------------------------
# Fast tests (default suite)
# ---------------------------------------------------------------------------
def test_mutual_information_runs():
    data = _toy_data()
    ptw, mi = mutual_information(
        idxs_x=(0,),
        idxs_y=(1,),
        data=data,
        flow_kwargs=FAST_FLOW,
        train_kwargs=FAST_TRAIN,
    )
    assert np.isfinite(float(mi))
    assert ptw.shape == (FAST_N,)
    # Structural identity: pointwise values average to the expected value
    # (holds regardless of how well the flow trained).
    assert float(ptw.mean()) == pytest.approx(float(mi), abs=1e-5)


def test_total_correlation_runs():
    """Regression guard: total_correlation once raised a TypeError from a stray
    ``context`` keyword argument."""
    data = _toy_data()
    ptw, tc = total_correlation(
        idxs=(0, 1, 2),
        data=data,
        flow_kwargs=FAST_FLOW,
        train_kwargs=FAST_TRAIN,
    )
    assert np.isfinite(float(tc))
    assert ptw.shape == (FAST_N,)


def test_higher_order_information_keys():
    """Regression guard: the result dict is keyed oinfo/sinfo/tc/dtc, each with
    ``avg`` and ``ptw`` entries (the keys the documentation example relies on)."""
    data = _toy_data()
    res = higher_order_information(
        idxs=(0, 1, 2),
        data=data,
        flow_kwargs=FAST_FLOW,
        train_kwargs=FAST_TRAIN,
    )
    assert set(res) == {"oinfo", "sinfo", "tc", "dtc"}
    for entry in res.values():
        assert "avg" in entry and "ptw" in entry
        assert np.isfinite(float(entry["avg"]))


def test_no_zero_element_warning():
    """Regression guard: an unconditional flow once emitted a torch
    'Initializing zero-element tensors is a no-op' UserWarning."""
    data = _toy_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mutual_information(
            idxs_x=(0,),
            idxs_y=(1,),
            data=data,
            flow_kwargs=FAST_FLOW,
            train_kwargs=FAST_TRAIN,
        )
    zero_element = [w for w in caught if "zero-element" in str(w.message).lower()]
    assert not zero_element


# ---------------------------------------------------------------------------
# Slow test (opt-in via `pytest --runslow`)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_converges_to_gaussian_mi():
    """
    With proper training, the neural estimate recovers the analytic Gaussian
    mutual information I = -1/2 * ln(1 - rho^2). Gated behind --runslow because
    training the flows takes ~1 minute.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    n, rho = 10_000, 0.6
    x = torch.randn(n)
    y = rho * x + np.sqrt(1 - rho**2) * torch.randn(n)
    data = torch.stack([x, y], dim=1)

    ptw, mi = mutual_information(idxs_x=(0,), idxs_y=(1,), data=data, verbose=True)

    analytic = -0.5 * np.log(1 - rho**2)
    assert float(mi) == pytest.approx(analytic, abs=3e-2)
    assert float(ptw.mean()) == pytest.approx(float(mi), abs=1e-5)

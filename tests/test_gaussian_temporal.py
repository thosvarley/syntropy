import pytest
import numpy as np

from syntropy.gaussian.temporal import (
    construct_csd_tensor,
    total_correlation_rate,
    dual_total_correlation_rate,
    s_information_rate,
    o_information_rate,
    k_wms_rate,
    fftfreqs_hz,
)
from syntropy.gaussian.multivariate_mi import (
    total_correlation,
    dual_total_correlation,
    s_information,
    o_information,
)

# Loose tolerance for comparisons against an analytic/static ground truth,
# where the rate estimate carries spectral-estimation noise from a finite
# sample. Tight tolerance for identities computed from the *same*
# deterministic dataset, which hold to floating-point precision.
ANALYTIC_TOL = 1e-2
EXACT_TOL = 1e-9


def equicorr_matrix(N: int, rho: float) -> np.ndarray:
    """N x N equicorrelation matrix with off-diagonal rho."""
    return (1 - rho) * np.eye(N) + rho * np.ones((N, N))


def _real(x):
    return np.real(x)


# ---------------------------------------------------------------------------
# White-noise fixtures.
#
# For i.i.d. multivariate Gaussian white noise, the cross-spectral density
# is flat across frequency and equal to the instantaneous covariance matrix
# -- exactly the property tests/test_gaussian.py's
# TestAR1EntropyRate.test_white_noise_entropy_rate relies on for the
# univariate case. Because (1/2*pi) * integral over the 2*pi-wide frequency
# range of a constant just returns that constant, every *_rate function
# here should reduce to the corresponding *static* multivariate measure
# evaluated at that covariance matrix.
# ---------------------------------------------------------------------------
_N = 3
_RHO = 0.45
_IDXS = tuple(range(_N))
_COV = equicorr_matrix(_N, _RHO)

_rng_large = np.random.default_rng(7)
_NPERSEG_LARGE = 2**12
_DATA_EQUICORR = _rng_large.multivariate_normal(
    mean=np.zeros(_N), cov=_COV, size=1_500_000
).T
_DATA_INDEPENDENT = _rng_large.standard_normal((_N, 1_500_000))

# Small/fast dataset for internal-identity and wrapper-correctness checks,
# which only need self-consistency (not convergence to a ground truth).
_rng_small = np.random.default_rng(11)
_NPERSEG_SMALL = 2**11
_DATA_SMALL = _rng_small.multivariate_normal(
    mean=np.zeros(_N), cov=_COV, size=300_000
).T


def test_rate_measures_match_static_for_white_noise():
    _, tc_rate = total_correlation_rate(_IDXS, _DATA_EQUICORR, nperseg=_NPERSEG_LARGE)
    _, dtc_rate = dual_total_correlation_rate(
        _IDXS, _DATA_EQUICORR, nperseg=_NPERSEG_LARGE
    )
    _, s_rate = s_information_rate(_IDXS, _DATA_EQUICORR, nperseg=_NPERSEG_LARGE)
    _, o_rate = o_information_rate(_IDXS, _DATA_EQUICORR, nperseg=_NPERSEG_LARGE)

    assert _real(tc_rate) == pytest.approx(total_correlation(_COV, _IDXS), abs=ANALYTIC_TOL)
    assert _real(dtc_rate) == pytest.approx(
        dual_total_correlation(_COV, _IDXS), abs=ANALYTIC_TOL
    )
    assert _real(s_rate) == pytest.approx(s_information(_COV, _IDXS), abs=ANALYTIC_TOL)
    assert _real(o_rate) == pytest.approx(o_information(_COV, _IDXS), abs=ANALYTIC_TOL)


def test_rate_measures_near_zero_for_independent_white_noise():
    _, tc_rate = total_correlation_rate(
        _IDXS, _DATA_INDEPENDENT, nperseg=_NPERSEG_LARGE
    )
    _, dtc_rate = dual_total_correlation_rate(
        _IDXS, _DATA_INDEPENDENT, nperseg=_NPERSEG_LARGE
    )
    _, s_rate = s_information_rate(_IDXS, _DATA_INDEPENDENT, nperseg=_NPERSEG_LARGE)
    _, o_rate = o_information_rate(_IDXS, _DATA_INDEPENDENT, nperseg=_NPERSEG_LARGE)

    for val in (tc_rate, dtc_rate, s_rate, o_rate):
        assert _real(val) == pytest.approx(0.0, abs=ANALYTIC_TOL)


def test_rate_measure_identities_and_k_wms_wrappers():
    # These are computed from the same deterministic dataset (scipy.signal.csd
    # has no randomness given fixed data), so the identities hold to
    # floating-point precision regardless of estimation noise -- a
    # small/fast dataset and a tight tolerance both work here.
    _, tc = total_correlation_rate(_IDXS, _DATA_SMALL, nperseg=_NPERSEG_SMALL)
    _, dtc = dual_total_correlation_rate(_IDXS, _DATA_SMALL, nperseg=_NPERSEG_SMALL)
    _, s = s_information_rate(_IDXS, _DATA_SMALL, nperseg=_NPERSEG_SMALL)
    _, o = o_information_rate(_IDXS, _DATA_SMALL, nperseg=_NPERSEG_SMALL)
    _, kwms0 = k_wms_rate(_IDXS, k=0, data=_DATA_SMALL, nperseg=_NPERSEG_SMALL)
    _, kwms1 = k_wms_rate(_IDXS, k=1, data=_DATA_SMALL, nperseg=_NPERSEG_SMALL)
    _, kwms2 = k_wms_rate(_IDXS, k=2, data=_DATA_SMALL, nperseg=_NPERSEG_SMALL)

    assert _real(s) == pytest.approx(_real(tc) + _real(dtc), abs=EXACT_TOL)
    assert _real(o) == pytest.approx(_real(tc) - _real(dtc), abs=EXACT_TOL)
    assert _real(dtc) == pytest.approx(_real(kwms1), abs=EXACT_TOL)
    assert _real(s) == pytest.approx(_real(kwms0), abs=EXACT_TOL)
    assert _real(o) == pytest.approx(-_real(kwms2), abs=EXACT_TOL)


def test_k_wms_rate_matches_definition_from_total_correlation_rate():
    # WMS^k(X) = (N - k) * TC(X) - sum_i TC(X^{-i}).
    # k=3=N is an edge case (N - k = 0, so only the leave-one-out sum
    # contributes), distinct from the k=0/1/2 wrappers tested above.
    k = 3
    _, tc_whole = total_correlation_rate(_IDXS, _DATA_SMALL, nperseg=_NPERSEG_SMALL)

    sum_residual = 0.0
    for i in range(_N):
        residual_idxs = tuple(_IDXS[j] for j in range(_N) if j != i)
        _, tc_res = total_correlation_rate(
            residual_idxs, _DATA_SMALL, nperseg=_NPERSEG_SMALL
        )
        sum_residual += _real(tc_res)

    expected = (_N - k) * _real(tc_whole) - sum_residual

    _, kwms = k_wms_rate(_IDXS, k=k, data=_DATA_SMALL, nperseg=_NPERSEG_SMALL)
    assert _real(kwms) == pytest.approx(expected, abs=EXACT_TOL)


def test_construct_csd_tensor_structural_properties():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 50_000))
    nperseg = 2**8

    S, omega = construct_csd_tensor((0, 1, 2), data, fs=1, nperseg=nperseg)

    assert S.shape == (nperseg, 3, 3)
    assert omega.shape == (nperseg,)
    assert np.allclose(S, np.conj(np.transpose(S, (0, 2, 1))))  # Hermitian
    assert np.allclose(np.imag(np.diagonal(S, axis1=1, axis2=2)), 0.0, atol=1e-8)

    assert omega.min() == pytest.approx(-np.pi, abs=1e-6)
    assert omega.max() < np.pi
    assert np.all(np.diff(omega) > 0)


def test_construct_csd_tensor_matches_variance_for_white_noise():
    # For independent white noise the CSD is flat across frequency and
    # equal to the instantaneous (co)variance matrix.
    rng = np.random.default_rng(3)
    variances = np.array([1.0, 4.0, 0.25])
    data = rng.standard_normal((3, 200_000)) * variances[:, None] ** 0.5

    S, _ = construct_csd_tensor((0, 1, 2), data, nperseg=2**10)

    diag_mean = np.real(np.diagonal(S, axis1=1, axis2=2)).mean(axis=0)
    assert diag_mean == pytest.approx(variances, abs=0.1)

    offdiag_mean_mag = np.abs(S[:, 0, 1]).mean()
    assert offdiag_mean_mag < 0.2


def test_fftfreqs_hz():
    nperseg, fs = 128, 10
    result = fftfreqs_hz(nperseg, fs)

    assert np.allclose(result, np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / fs)))
    assert np.all(np.diff(result) > 0)
    assert len(result) == nperseg


def test_fftfreqs_hz_matches_construct_csd_tensor_omega():
    # omega (returned by construct_csd_tensor) is 2*pi*f/fs where f is the
    # underlying FFT bin frequency, so it should invert cleanly back to the
    # same Hz values fftfreqs_hz produces independently.
    rng = np.random.default_rng(5)
    fs = 10
    nperseg = 2**7
    data = rng.standard_normal((2, 5000))

    _, omega = construct_csd_tensor((0, 1), data, fs=fs, nperseg=nperseg)
    derived_hz = omega * fs / (2 * np.pi)

    assert np.allclose(derived_hz, fftfreqs_hz(nperseg, fs))

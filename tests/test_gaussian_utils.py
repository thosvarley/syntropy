import pytest
import numpy as np

from syntropy.gaussian.utils import (
    check_cov,
    correlation_to_mutual_information,
    copula_transform,
    covariance_to_correlation,
)

pytest_abs = 1e-6


def equicorr_matrix(N: int, rho: float) -> np.ndarray:
    """N x N equicorrelation matrix with off-diagonal rho."""
    return (1 - rho) * np.eye(N) + rho * np.ones((N, N))


def test_check_cov():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3, 5000))

    computed = check_cov(None, data)
    assert computed.shape == (3, 3)
    assert np.allclose(computed, np.cov(data, ddof=0))

    given = equicorr_matrix(3, 0.5)
    returned = check_cov(given, data)
    assert np.allclose(returned, given)
    assert returned is not given  # copied, not aliased
    given[0, 0] = 999.0
    assert returned[0, 0] != 999.0


def test_check_cov_rejects_mismatched_dimensionality():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3, 100))
    with pytest.raises(AssertionError):
        check_cov(equicorr_matrix(4, 0.5), data)


def test_correlation_to_mutual_information():
    with np.errstate(divide="ignore"):
        mi_zero = correlation_to_mutual_information(np.eye(2))

        rho = 0.6
        mi_rho = correlation_to_mutual_information(equicorr_matrix(2, rho))

        mi_diag = correlation_to_mutual_information(equicorr_matrix(3, 0.3))

    assert mi_zero[0, 1] == pytest.approx(0.0, abs=pytest_abs)

    expected = -np.log(1 - rho**2) / 2.0
    assert mi_rho[0, 1] == pytest.approx(expected, abs=pytest_abs)
    assert mi_rho[1, 0] == pytest.approx(expected, abs=pytest_abs)

    assert np.all(np.isnan(np.diag(mi_diag)))


def test_copula_transform_shape_and_correlation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 1000))

    Z, R = copula_transform(X)

    assert Z.shape == X.shape
    assert R.shape == (3, 3)
    assert np.allclose(R, np.corrcoef(Z), atol=pytest_abs)


def test_copula_transform_gaussianizes_marginals():
    # The rank transform followed by the inverse normal CDF should produce
    # approximately standard-normal marginals regardless of the input
    # marginal distribution.
    rng = np.random.default_rng(0)
    X = rng.exponential(size=(2, 5000))

    Z, _ = copula_transform(X)

    assert Z.mean(axis=1) == pytest.approx(np.zeros(2), abs=0.1)
    assert Z.std(axis=1) == pytest.approx(np.ones(2), abs=0.1)


def test_copula_transform_invariant_to_monotonic_rescaling():
    # A monotonic transform of the input should not change the copula
    # correlation, since it is computed purely from ranks.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 500))
    X_rescaled = np.vstack([X[0] * 3.0 + 1.0, np.exp(X[1])])

    _, R1 = copula_transform(X)
    _, R2 = copula_transform(X_rescaled)

    assert np.allclose(R1, R2, atol=pytest_abs)


def test_covariance_to_correlation():
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    corr = covariance_to_correlation(cov)

    assert np.diag(corr) == pytest.approx(np.ones(2), abs=pytest_abs)
    expected_off_diag = 1.0 / (2.0 * 3.0)
    assert corr[0, 1] == pytest.approx(expected_off_diag, abs=pytest_abs)
    assert corr[1, 0] == pytest.approx(expected_off_diag, abs=pytest_abs)

    # A matrix that's already a correlation matrix is left unchanged.
    corr_in = equicorr_matrix(3, 0.4)
    assert np.allclose(covariance_to_correlation(corr_in), corr_in, atol=pytest_abs)

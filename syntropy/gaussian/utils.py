from __future__ import annotations

import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray

# %% LIBRARY


def check_cov(
    cov: NDArray[np.floating] | None, data: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Normalizes an optional covariance matrix against a data array.

    Parameters
    ----------
    cov : NDArray[np.floating] | None
        The covariance matrix that defines the distribution. If None, it
        is computed directly from data.
    data : NDArray[np.floating]
        Data array of shape (n_variables, n_samples), used to compute the
        covariance matrix when cov is None, and to validate its
        dimensionality otherwise.

    Returns
    -------
    NDArray[np.floating]
        cov itself (copied) if given, otherwise the covariance matrix
        computed from data.

    """
    if cov is None:
        cov_ = np.atleast_2d(np.cov(data, ddof=0))
    else:
        assert cov.shape[0] == data.shape[0], (
            "The data and given covariance matrix must have the same dimensionality"
        )
        cov_ = cov.copy()

    return cov_


def correlation_to_mutual_information(
    cov: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Converts a Pearson correlation matrix to a Guassian mutual
    information matrix based on the identity:

    :math:`I(X;Y) = \\frac{-\\log(1-(r_{XY})^2)}{2}`

    Where :math:`r_{XY}` is the Pearson correlation coefficient between :math:`X` and :math:`Y`.

    Also works for a covariance matrix if the processes have 0 mean
    and unit variance.

    Parameters
    ----------
    cov : NDArray[np.floating]
        A covariance matrix.

    Returns
    -------
    NDArray[np.floating]
        The equivalent mutual information matrix.

    """
    mi = -np.log(1 - (cov**2)) / 2.0
    np.fill_diagonal(mi, np.nan)

    return mi


def copula_transform(
    X: NDArray[np.floating],
) -> tuple[NDArray[np.floating], ...]:
    """
    Transform data to Gaussian copula space and compute the correlation matrix. Useful for copula-based information estimators.

    The resulting data can be plugged into any local or expected information estimator.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_samples)
        Input data (channels x samples)

    Returns
    -------
    Z : NDArray[np.floating]
        Gaussianized copula data (zero-mean, unit-variance)
    R : NDArray[np.floating]
        Copula correlation matrix

    References
    ----------
    Ince, R. A. A., Giordano, B. L., Kayser, C., Rousselet, G. A., Gross, J., & Schyns, P. G. (2016).
    A statistical framework for neuroimaging data analysis based on mutual information estimated via a gaussian copula.
    Human Brain Mapping, 38(3), 1541–1573.
    https://doi.org/10.1002/hbm.23471

    """

    N0: int
    N1: int
    N0, N1 = X.shape

    # Step 1: rank transform to uniforms
    U: NDArray[np.floating] = np.zeros_like(X)

    for i in range(N0):
        ranks = stats.rankdata(X[i, :], method="average")
        U[i, :] = (ranks - 0.5) / N1  # rescale to (0,1)

    # Step 2: map uniforms to standard normal
    Z: NDArray[np.floating] = stats.norm.ppf(U)

    # Step 3: correlation matrix of transformed data
    R: NDArray[np.floating] = np.corrcoef(Z)

    return Z, R


def covariance_to_correlation(cov: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Converts a non-standardized covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : NDArray[np.floating]
        The covariance matrix.

    Returns
    -------
    NDArray[np.floating]
        The correlation matrix.


    """
    diag: NDArray[np.floating] = np.sqrt(np.diag(cov))
    d_inv: NDArray[np.floating] = np.diag(1.0 / diag)

    return d_inv @ cov @ d_inv

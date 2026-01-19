import numpy as np
import scipy.stats as stats
import itertools as it
from numpy.typing import NDArray


COV_NULL: NDArray[np.floating] = np.array([[-1.0]])

# %% LIBRARY


def check_cov(
    cov: NDArray[np.floating], data: NDArray[np.floating]
) -> NDArray[np.floating]:
    if cov.shape == ():
        cov_ = cov
    else:
        if cov[0, 0] == -1:
            cov_ = np.cov(data, ddof=0.0)
        else:
            cov_ = cov.copy()

    return cov_


def make_powerset(iterable):
    """
    Computes the powerset of a collection of elements.

    .. math::
        \\mathcal{P}(\\{X_1,X_2,X_3\\}) \\to (\\{\\}, \\{X_1\\}, \\{X_2\\}, \\{X_3\\}, \\{X_1,X_2\\}, \\{X_1,X_3\\}, \\{X_1,X_2,X_3\\} )

    """
    xs: list = list(iterable)
    # note we return an iterator rather than a list
    return it.chain.from_iterable(it.combinations(xs, n) for n in range(len(xs) + 1))


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
    Transform data to Gaussian copula space and compute the correlation matrix.

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

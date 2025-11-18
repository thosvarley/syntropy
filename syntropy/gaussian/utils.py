import numpy as np
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




def correlation_to_mutual_information(cov: NDArray[np.floating]) -> NDArray[np.floating]:
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

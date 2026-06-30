from __future__ import annotations

import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree


def sample_entropy(
    idx: tuple[int],
    data: NDArray[np.floating],
    m: int = 2,
    r: float = 0.2,
    normalize_r: bool = True,
    norm: float = np.inf,
) -> float:
    """
    Computes the Sample Entropy of a univariate time series.
    The sample entropy is a non-parametric approximation of the entropy rate of a continuous process.
    For analogous functions, see the Lempel-Ziv complexity for discrete time series and the spectral entropy rate. 

    Note that while it is in the KNN module, it does not use the same logic as the Kraskov-based estimators, but since it leverages the same cKDTree machinery, we opted to include it here.

    Parameters
    ----------
    idx : tuple[int]
        The index of the variable to be analyzed, as a tuple.
        E.g. for index of 0, pass (0,)
        Currently Syntropy only support univariate sample entropy.
    data : NDArray[np.floating]
        The time series data to be analyzed in channels x time format.
    m : int
        The embedding dimension.
        The default value is 2.
    r : float
        The radius within which to count matching templates.
        The default value of 0.2.
    normalize_r : bool
        Whether to normalize r by the standard deviation of the data.
        The default value is True.
    norm : float
        The norm to use when computing the distance between templates.
        The default value is np.inf (the Chebyshev or max-norm)

    Returns
    -------
    float
        The sample entropy.

    References
    ----------
    Richman JS, Moorman JR. (2000)
    Physiological time-series analysis using approximate entropy and sample entropy.
    Am J Physiol Heart Circ Physiol.278(6):H2039-49
    https://pubmed.ncbi.nlm.nih.gov/10843903/

    """
    series: NDArray[np.floating] = data[idx[0], :]
    N: int = series.shape[0] - m

    embed_small: NDArray = sliding_window_view(series, m)[:N]
    embed_big: NDArray = sliding_window_view(series, m + 1)

    tree_small: cKDTree = cKDTree(embed_small)
    tree_big: cKDTree = cKDTree(embed_big)

    if normalize_r:
        r *= np.std(series)

    counts_small: int = tree_small.count_neighbors(tree_small, r, p=norm) - N
    counts_big: int = tree_big.count_neighbors(tree_big, r, p=norm) - N

    if counts_big == 0:
        return np.inf
    else:
        return -np.log(counts_big / counts_small)

def cross_sample_entropy(
    idx_x: tuple[int],
    idx_y: tuple[int],
    data: NDArray[np.floating],
    m: int = 2,
    r: float = 0.2,
    norm: float = np.inf,
) -> float:
    """
    Computes the cross-sample entropy between two time series. 
    The cross-sample entropy is not a measure of dependency, but rather, a measure of how similar the dynamics between two time series are. 
    Cross-sample entropy is low when two time series are similar, and high when they are asynchronous. 

    Parameters
    ----------
    idx_x : tuple[int]
        A tuple giving the index of the X variable.
        Syntropy currently only supports univariate X.
    idx_y : tuple[int]
        A tuple giving the index of the Y variable.
        Syntropy currently only supports univariate Y.
    data : NDArray[np.floating]
        The time series data to be analyzed in channels x time format.
    m : int
        The embedding dimension.
        The default value is 2.
    r : float
        The radius within which to count matching templates.
        The default value of 0.2.
    norm : float
        The norm to use when computing the distance between templates.
        The default value is np.inf (the Chebyshev or max-norm)
        
    Returns
    -------
    float
       The cross-sample entropy.

    References
    ----------
    Richman JS, Moorman JR. (2000)
    Physiological time-series analysis using approximate entropy and sample entropy.
    Am J Physiol Heart Circ Physiol.278(6):H2039-49
    https://pubmed.ncbi.nlm.nih.gov/10843903/


    """
    N: int = data.shape[1] - m
    
    series_x: NDArray[np.floating] = stats.zscore(data[idx_x[0], :])
    series_y: NDArray[np.floating] = stats.zscore(data[idx_y[0], :])

    embed_x_small: NDArray = sliding_window_view(series_x, m)[:N]
    embed_x_big: NDArray = sliding_window_view(series_x, m + 1)
    embed_y_small: NDArray = sliding_window_view(series_y, m)[:N]
    embed_y_big: NDArray = sliding_window_view(series_y, m + 1)

    tree_x_small: cKDTree = cKDTree(embed_x_small)
    tree_x_big: cKDTree = cKDTree(embed_x_big)

    tree_y_small: cKDTree = cKDTree(embed_y_small)
    tree_y_big: cKDTree = cKDTree(embed_y_big)

    B = tree_x_small.count_neighbors(tree_y_small, r, p=norm)
    A = tree_x_big.count_neighbors(tree_y_big, r, p=norm)

    if A == 0:
        return np.inf
    else:
        return -np.log(A / B)

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma
from scipy.spatial import cKDTree
from utils import check_idxs, build_tree_and_get_distances, get_counts_from_tree 


def total_correlation(
    data: NDArray[np.floating],
    k: int,
    idxs: tuple[int, ...] = (-1,),
    algorithm: int = 1,
) -> tuple[NDArray[np.floating], float]:
    """

    Parameters
    ----------
    algorithm :

    idxs_x : tuple[int, ...]

    idxs_y : tuple[int, ...]

    k : int

    data : NDArray[np.floating]


    Returns
    -------
    tuple[NDArray[np.floating], float]


    """
    assert algorithm in {1, 2}, "Algorithm must be 1 or 2."

    if algorithm == 1:
        return total_correlation_1(data=data, k=k, idxs=idxs)
    else:
        return total_correlation_2(data=data, k=k, idxs=idxs)


def total_correlation_1(
    data: NDArray[np.floating], k: int, idxs: tuple[int, ...] = (-1,)
) -> tuple[NDArray[np.floating], float]:
    """

    Parameters
    ----------
    data : NDArray[np.floating]

    k : int

    idxs : tuple[int, ...]


    Returns
    -------
    tuple[NDArray[np.floating], float]


    """
    idxs_: tuple[int,...] = check_idxs(idxs, data.shape[0])

    N: int = data.shape[1]
    m: int = len(idxs_)
    
    psi_k, psi_N = digamma([k, N])

    tree: cKDTree = cKDTree(data[idxs_, :].T)
    distances: NDArray[np.floating]
    distances, _ = tree.query(data[idxs_, :].T, k=k + 1, p=np.inf)
    eps: NDArray[np.floating] = distances[:, -1]

    ptw: NDArray[np.floating] = np.zeros(N)

    for i in idxs_:
        tree_i: cKDTree = cKDTree(data[(i,), :].T)
        
        counts: NDArray[np.integer] = get_counts_from_tree(tree_i, data[i,:].T, eps)
        ptw -= digamma(counts + 1)

    ptw += psi_k + (m - 1) * psi_N

    return ptw, ptw.mean() 


def total_correlation_2(
    data: NDArray[np.floating], k: int, idxs: tuple[int, ...] = (-1,)
) -> tuple[NDArray[np.floating], float]:
    """
    Computes the Kraskov, Stogbauer, Grassberger estimate of the total correlation using the second algorithm presented in:

    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information.
    Physical Review E, 69(6), 066138.
    https://doi.org/10.1103/PhysRevE.69.066138

    Parameters
    ----------
    data : NDArray[np.floating]

    k : int

    idxs : tuple[int, ...]


    Returns
    -------
    tuple[NDArray[np.floating], float]


    """
    idxs_: tuple[int,...] = check_idxs(idxs, data.shape[0])

    N: int = data.shape[1]
    m: int = len(idxs_)

    psi_k, psi_N = digamma([k, N])

    tree: cKDTree = cKDTree(data[idxs_, :].T)

    distances: NDArray[np.floating]
    indices: NDArray[np.integer]
    distances, indices = tree.query(data[idxs_, :].T, k=k + 1, p=np.inf)

    neighbors: NDArray[np.integer] = indices[:, 1:]
    ptw: NDArray[np.floating] = np.zeros(N)

    for i in idxs_:
        data_i: NDArray[np.floating] = data[(i,), :].T
        eps: NDArray[np.floating] = np.repeat(-np.inf, N)

        for j in range(k):
            norm: NDArray[np.floating] = np.linalg.norm(
                data_i - data_i[neighbors[:, j]], ord=np.inf, axis=1
            )
            eps = np.maximum(eps, norm)

        tree_i: cKDTree = cKDTree(data_i)
        counts: NDArray[np.integer] = get_counts_from_tree(tree_i, data_i.T, eps)
        ptw -= digamma(counts)

    ptw += psi_k - ((m - 1) / k) + ((m - 1) * psi_N)

    return ptw, ptw.mean()


def dual_total_correlation(
    data: NDArray[np.floating], k: int, idxs: tuple[int, ...] = (-1,)
) -> tuple[NDArray[np.floating], float]:
    """
    Compute dual total correlation using KSG estimation.

    Parameters
    ----------
    data : NDArray[np.floating]
        Data array of shape (n_variables, n_samples)
    k : int
        Number of nearest neighbors
    idxs : tuple[int, ...]
        Indices of variables to use (-1 means all)

    Returns
    -------
    tuple[NDArray[np.floating], float]
        Local values and average dual total correlation
    """
    idxs_: tuple[int,...] = check_idxs(idxs, data.shape[0])

    N: int = data.shape[1]
    m: int = len(idxs_)

    psi_k, psi_N = digamma([k, N])

    # Build tree for joint distribution (all m dimensions)
    tree, distances, _ = build_tree_and_get_distances(data[idxs_, :], k=k)
    eps: NDArray[np.floating] = distances[:, -1]

    # Initialize local values: start with (ψ(k) - ψ(N))
    ptw: NDArray[np.floating] = np.full(N, psi_k - psi_N)

    # Build marginal trees and compute counts
    # Each marginal excludes one dimension (so has m-1 dimensions)
    for i in range(m):
        # Get indices for this marginal (all dimensions except j)
        residual_idxs = [idxs_[j] for j in range(m) if j != i]
        marginal_data = data[residual_idxs, :].T
        tree_i = cKDTree(marginal_data)

        # Count neighbors strictly within eps for each point
        counts: NDArray[np.integer] = get_counts_from_tree(tree_i, marginal_data.T, eps)

        # Subtract the contribution from this marginal, divided by (m-1)
        ptw -= (digamma(counts + 1) - psi_N) / (m - 1)

    # Multiply everything by (m-1)
    ptw *= m - 1

    return ptw, ptw.mean() 


def s_information(
    data: NDArray[np.floating], k: int, idxs: tuple[int, ...] = (-1,)
) -> tuple[NDArray[np.floating], float]:
    """
    Compute S-information using KSG estimation.
    
    S-information quantifies the balance between redundancy and synergy
    in multivariate information.

    Parameters
    ----------
    data : NDArray[np.floating]
        Data array of shape (n_variables, n_samples)
    k : int
        Number of nearest neighbors
    idxs : tuple[int, ...]
        Indices of variables to use (-1 means all)

    Returns
    -------
    tuple[NDArray[np.floating], float]
        Local values and average S-information
    """
    idxs_: tuple[int,...] = check_idxs(idxs, data.shape[0])

    N: int = data.shape[1]
    m: int = len(idxs_)

    psi_k, psi_N = digamma([k, N])

    # Build tree for joint distribution (all m dimensions)
    tree, distances, _ = build_tree_and_get_distances(data[idxs_, :], k=k)
    eps: NDArray[np.floating] = distances[:, -1]

    # Initialize local values: start with (ψ(k) - ψ(N))
    ptw: NDArray[np.floating] = np.full(N, psi_k - psi_N)

    # For each dimension d
    for d in range(m):
        # Small marginal: just dimension d alone (1D)
        small_marginal_data = data[idxs_[d]:idxs_[d]+1, :].T  # Shape: (N, 1)
        tree_small = cKDTree(small_marginal_data)
        
        # Big marginal: all dimensions except d (m-1 dimensions)
        big_marginal_idxs = [idxs_[j] for j in range(m) if j != d]
        big_marginal_data = data[big_marginal_idxs, :].T  # Shape: (N, m-1)
        tree_big = cKDTree(big_marginal_data)
        
        counts_small = get_counts_from_tree(tree_small, small_marginal_data.T, eps)
        counts_big = get_counts_from_tree(tree_big, big_marginal_data.T, eps)
        
        # Subtract contributions from both marginals, divided by m
        ptw -= (digamma(counts_big + 1) - psi_N) / m
        ptw -= (digamma(counts_small + 1) - psi_N) / m

    # Multiply everything by m
    ptw *= m

    return ptw, ptw.mean()


def o_information(
    data: NDArray[np.floating], k: int, idxs: tuple[int, ...] = (-1,)
) -> tuple[NDArray[np.floating], float]:
    """
    Compute O-information using KSG estimation.
    
    O-information quantifies the balance between redundancy (positive values)
    and synergy (negative values) in multivariate information.

    Parameters
    ----------
    data : NDArray[np.floating]
        Data array of shape (n_variables, n_samples)
    k : int
        Number of nearest neighbors
    idxs : tuple[int, ...]
        Indices of variables to use (-1 means all)

    Returns
    -------
    tuple[NDArray[np.floating], float]
        Local values and average O-information
    """
    if idxs[0] == -1:
        idxs_ = tuple(range(data.shape[0]))
    else:
        idxs_ = idxs

    N: int = data.shape[1]
    m: int = len(idxs_)

    # Special case: O-information is 0 for 2D data
    if m == 2:
        return np.zeros(N), 0.0

    psi_k, psi_N = digamma([k, N])

    tree, distances, _ = build_tree_and_get_distances(data[idxs_, :], k=k)
    eps: NDArray[np.floating] = distances[:, -1]

    # Initialize local values: start with (ψ(k) - ψ(N))
    ptw: NDArray[np.floating] = np.full(N, psi_k - psi_N)

    for d in range(m):
        # Small marginal: just dimension d alone (1D)
        small_marginal_data = data[idxs_[d]:idxs_[d]+1, :]  # Shape: (N, 1)
        tree_small = cKDTree(small_marginal_data.T)
        
        # Big marginal: all dimensions except d (m-1 dimensions)
        big_marginal_idxs = [idxs_[j] for j in range(m) if j != d]
        big_marginal_data = data[big_marginal_idxs, :]  # Shape: (N, m-1)
        tree_big = cKDTree(big_marginal_data.T)
        
        counts_small = get_counts_from_tree(tree_small, small_marginal_data, eps)
        counts_big = get_counts_from_tree(tree_big, big_marginal_data, eps)
        
        ptw -= (digamma(counts_big + 1) - psi_N) / (m - 2)
        ptw += (digamma(counts_small + 1) - psi_N) / (m - 2)

    # Multiply everything by (2 - m)
    ptw *= (2 - m)

    return ptw, ptw.mean() 

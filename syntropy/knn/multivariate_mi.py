import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma
from scipy.spatial import cKDTree
from .utils import check_idxs, build_tree_and_get_distances, get_counts_from_tree

def total_correlation(
    data: NDArray[np.floating],
    k: int,
    idxs: tuple[int, ...] = (-1,),
    algorithm: int = 1,
) -> tuple[NDArray[np.floating], float]:
    """
    A wrapper function for the two TC functions.

    Parameters
    ----------
    data : NDArray[np.floating]
        Data array of shape (n_variables, n_samples)
    k : int
        Number of nearest neighbors
    idxs : tuple[int, ...]
        Indices of variables to use (-1 means all)
    algorithm : int
        Whether to use algorithm 1 or 2.
        Defaults to 1

    See Also
    --------
    total_correlation_1 : The TC computed using algorithm 1.
    total_correlation_2 : The TC computed using algorithm 2.

    Returns
    -------
    NDArray[np.floating
        The local total correlation for each sample.
    float
        The expected total correlation over all samples

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
    Computes the Kraskov, Stogbauer, Grassberger estimate of the total correlation using the first algorithm presented Kraskov et. al (2004)

    .. math::
        \hat{TC}(X) = \psi(k) - (m-1)\psi(N) -\\langle \psi(n_{x_{1}}+1) + \ldots + \psi(n_{x_{N}}+1)\\rangle

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
    NDArray[np.floating
        The local total correlation for each sample.
    float
        The expected total correlation over all samples

    References
    ----------
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information.
    Physical Review E, 69(6), 066138.
    https://doi.org/10.1103/PhysRevE.69.066138

    Watanabe, S. (1960).
    Information Theoretical Analysis of Multivariate Correlation.
    IBM Journal of Research and Development, 4(1), Article 1.
    https://doi.org/10.1147/rd.41.0066

    """
    idxs: tuple[int, ...] = check_idxs(idxs, data.shape[0])

    m: int = len(idxs)
    N: int = data.shape[1]

    psi_k: float = digamma(k)
    psi_N: float = digamma(N)

    ptw: NDArray[np.floating] = np.full((1, N), psi_k + (m - 1) * psi_N)

    distances: NDArray[np.floating]
    _, distances, _ = build_tree_and_get_distances(data=data[idxs, :], k=k)

    for idx in idxs:
        tree, _, _ = build_tree_and_get_distances(data[(idx,), :], k=k)
        counts: NDArray[np.integer] = get_counts_from_tree(
            tree, data[(idx,), :], distances[:, -1]
        )

        ptw[0, :] -= digamma(counts + 1)

    return ptw, ptw.mean()


def total_correlation_2(
    data: NDArray[np.floating], k: int, idxs: tuple[int, ...] = (-1,)
) -> tuple[NDArray[np.floating], float]:
    """
    Computes the Kraskov, Stogbauer, Grassberger estimate of the total correlation using the second algorithm presented in Kraskov et. al., (2004).

    .. math::
        \hat{TC}(X) = \psi(k) - ((m-1)/k) - (m-1)\psi(N) - \langle \psi(n_{x_{1}}) + \ldots + \psi(n_{x_{N}}) \\rangle`

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
    NDArray[np.floating
        The local total correlation for each sample.
    float
        The expected total correlation over all samples

    References
    ----------
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information.
    Physical Review E, 69(6), 066138.
    https://doi.org/10.1103/PhysRevE.69.066138

    Watanabe, S. (1960).
    Information Theoretical Analysis of Multivariate Correlation.
    IBM Journal of Research and Development, 4(1), Article 1.
    https://doi.org/10.1147/rd.41.0066

    """
    idxs = check_idxs(idxs, data.shape[0])

    m: int = len(idxs)
    N: int = data.shape[1]

    psi_k: float = digamma(k)
    psi_N: float = digamma(N)

    ptw: NDArray[np.floating] = np.full(
        (1, N), psi_k - ((m - 1) / k) + ((m - 1) * psi_N)
    )

    distances: NDArray[np.floating]
    indices: NDArray[np.integer]
    _, distances, indices = build_tree_and_get_distances(data[idxs, :], k=k)
    neighbors: NDArray[np.floating] = indices[:, 1:]

    for idx in idxs:
        data_idx: NDArray[np.floating] = data[
            (idx,),
            :,
        ].T

        eps: NDArray[np.floating] = np.full(N, -np.inf)

        for j in range(k):
            norm: NDArray[np.floating] = np.linalg.norm(
                data_idx - data_idx[neighbors[:, j]], ord=np.inf, axis=1
            )
            eps = np.maximum(norm, eps)

        tree, distances, _ = build_tree_and_get_distances(data_idx.T, k=k)
        counts: NDArray[np.integer] = get_counts_from_tree(
            tree, data_idx.T, eps, strict=False
        )
        ptw[0, :] -= digamma(counts)

    return ptw, ptw.mean()


# def dual_total_correlation(
#     data: NDArray[np.floating], k: int, idxs: tuple[int, ...] = (-1,)
# ) -> tuple[NDArray[np.floating], float]:
#     """
#     Compute dual total correlation using KSG estimation.
#     Code adapted from JIDT
#     https://github.com/jlizier/jidt/blob/master/java/source/infodynamics/measures/continuous/kraskov/DualTotalCorrelationCalculatorKraskov.java
#
#     Parameters
#     ----------
#     data : NDArray[np.floating]
#         Data array of shape (n_variables, n_samples)
#     k : int
#         Number of nearest neighbors
#     idxs : tuple[int, ...]
#         Indices of variables to use (-1 means all)
#
#     Returns
#     -------
#     NDArray[np.floating
#         The local dual total correlation for each sample.
#     float
#         The expected dual total correlation over all samples
#
#     References
#     ----------
#     Abdallah, S. A., & Plumbley, M. D. (2012).
#     A measure of statistical complexity based on predictive information with application to finite spin systems.
#     Physics Letters A, 376(4), 275–281.
#     https://doi.org/10.1016/j.physleta.2011.10.066
#
#     Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
#     Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information.
#     Physical Review E, 100(3), Article 3.
#     https://doi.org/10.1103/PhysRevE.100.032305
#
#     """
#     idxs_: tuple[int, ...] = check_idxs(idxs, data.shape[0])
#
#     N: int = data.shape[1]
#     m: int = len(idxs_)
#
#     psi_k, psi_N = digamma([k, N])
#
#     # Build tree for joint distribution (all m dimensions)
#     tree, distances, _ = build_tree_and_get_distances(data[idxs_, :], k=k)
#     eps: NDArray[np.floating] = distances[:, -1]
#
#     # Initialize local values: start with (ψ(k) - ψ(N))
#     ptw: NDArray[np.floating] = np.full((1,N), psi_k - psi_N)
#
#     # Build marginal trees and compute counts
#     # Each marginal excludes one dimension (so has m-1 dimensions)
#     for i in range(m):
#         # Get indices for this marginal (all dimensions except j)
#         residual_idxs = [idxs_[j] for j in range(m) if j != i]
#         marginal_data = data[residual_idxs, :].T
#         tree_i = cKDTree(marginal_data)
#
#         # Count neighbors strictly within eps for each point
#         counts: NDArray[np.integer] = get_counts_from_tree(tree_i, marginal_data.T, eps)
#
#         # Subtract the contribution from this marginal, divided by (m-1)
#         ptw[0,:] -= (digamma(counts + 1) - psi_N) / (m - 1)
#
#     # Multiply everything by (m-1)
#     ptw *= m - 1
#
#     return ptw, ptw.mean()
#
#
# def s_information(
#     data: NDArray[np.floating], k: int, idxs: tuple[int, ...] = (-1,)
# ) -> tuple[NDArray[np.floating], float]:
#     """
#     Compute S-information using KSG estimation.
#     Code adapted from JIDT
#     https://github.com/jlizier/jidt/blob/master/java/source/infodynamics/measures/continuous/kraskov/SInfoCalculatorKraskov.java
#
#     Parameters
#     ----------
#     data : NDArray[np.floating]
#         Data array of shape (n_variables, n_samples)
#     k : int
#         Number of nearest neighbors
#     idxs : tuple[int, ...]
#         Indices of variables to use (-1 means all)
#
#     Returns
#     -------
#     NDArray[np.floating
#         The local S-information for each sample.
#     float
#         The expected S-information over all samples
#
#     References
#     ----------
#     Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
#     Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information.
#     Physical Review E, 100(3), Article 3.
#     https://doi.org/10.1103/PhysRevE.100.032305
#
#     Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
#     Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
#     Communications Biology, 6(1), Article 1.
#     https://doi.org/10.1038/s42003-023-04843-w
#
#     """
#     idxs_: tuple[int, ...] = check_idxs(idxs, data.shape[0])
#
#     N: int = data.shape[1]
#     m: int = len(idxs_)
#
#     psi_k, psi_N = digamma([k, N])
#
#     # Build tree for joint distribution (all m dimensions)
#     tree, distances, _ = build_tree_and_get_distances(data[idxs_, :], k=k)
#     eps: NDArray[np.floating] = distances[:, -1]
#
#     # Initialize local values: start with (ψ(k) - ψ(N))
#     ptw: NDArray[np.floating] = np.full((1,N), psi_k - psi_N)
#
#     # For each dimension d
#     for d in range(m):
#         # Small marginal: just dimension d alone (1D)
#         small_marginal_data = data[idxs_[d] : idxs_[d] + 1, :].T  # Shape: (N, 1)
#         tree_small = cKDTree(small_marginal_data)
#
#         # Big marginal: all dimensions except d (m-1 dimensions)
#         big_marginal_idxs = [idxs_[j] for j in range(m) if j != d]
#         big_marginal_data = data[big_marginal_idxs, :].T  # Shape: (N, m-1)
#         tree_big = cKDTree(big_marginal_data)
#
#         counts_small = get_counts_from_tree(tree_small, small_marginal_data.T, eps)
#         counts_big = get_counts_from_tree(tree_big, big_marginal_data.T, eps)
#
#         # Subtract contributions from both marginals, divided by m
#         ptw[0,:] -= (digamma(counts_big + 1) - psi_N) / m
#         ptw[0,:] -= (digamma(counts_small + 1) - psi_N) / m
#
#     # Multiply everything by m
#     ptw *= m
#
#     return ptw, ptw.mean()
#
#
# def o_information(
#     data: NDArray[np.floating], k: int, idxs: tuple[int, ...] = (-1,)
# ) -> tuple[NDArray[np.floating], float]:
#     """
#     Compute O-information using KSG estimation.
#     Code adapted from JIDT
#     https://github.com/jlizier/jidt/blob/master/java/source/infodynamics/measures/continuous/kraskov/OInfoCalculatorKraskov.java
#
#     O-information quantifies the balance between redundancy (positive values)
#     and synergy (negative values) in multivariate information.
#
#     Parameters
#     ----------
#     data : NDArray[np.floating]
#         Data array of shape (n_variables, n_samples)
#     k : int
#         Number of nearest neighbors
#     idxs : tuple[int, ...]
#         Indices of variables to use (-1 means all)
#
#     Returns
#     -------
#     NDArray[np.floating
#         The local O-information for each sample.
#     float
#         The expected O-information over all samples
#
#     References
#     ----------
#     Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
#     Quantifying High-order Interdependencies via Multivariate
#     Extensions of the Mutual Information.
#     Physical Review E, 100(3), Article 3.
#     https://doi.org/10.1103/PhysRevE.100.032305
#
#     Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
#     Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
#     Communications Biology, 6(1), Article 1.
#     https://doi.org/10.1038/s42003-023-04843-w
#
#     """
#     if idxs[0] == -1:
#         idxs_ = tuple(range(data.shape[0]))
#     else:
#         idxs_ = idxs
#
#     N: int = data.shape[1]
#     m: int = len(idxs_)
#
#     # Special case: O-information is 0 for 2D data
#     if m == 2:
#         return np.zeros(N), 0.0
#
#     psi_k, psi_N = digamma([k, N])
#
#     tree, distances, _ = build_tree_and_get_distances(data[idxs_, :], k=k)
#     eps: NDArray[np.floating] = distances[:, -1]
#
#     # Initialize local values: start with (ψ(k) - ψ(N))
#     ptw: NDArray[np.floating] = np.full((1,N), psi_k - psi_N)
#
#     for d in range(m):
#         # Small marginal: just dimension d alone (1D)
#         small_marginal_data = data[idxs_[d] : idxs_[d] + 1, :]  # Shape: (N, 1)
#         tree_small = cKDTree(small_marginal_data.T)
#
#         # Big marginal: all dimensions except d (m-1 dimensions)
#         big_marginal_idxs = [idxs_[j] for j in range(m) if j != d]
#         big_marginal_data = data[big_marginal_idxs, :]  # Shape: (N, m-1)
#         tree_big = cKDTree(big_marginal_data.T)
#
#         counts_small = get_counts_from_tree(tree_small, small_marginal_data, eps)
#         counts_big = get_counts_from_tree(tree_big, big_marginal_data, eps)
#
#         ptw[0,:] -= (digamma(counts_big + 1) - psi_N) / (m - 2)
#         ptw[0,:] += (digamma(counts_small + 1) - psi_N) / (m - 2)
#
#     # Multiply everything by (2 - m)
#     ptw *= 2 - m
#
#     return ptw, ptw.mean()

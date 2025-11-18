import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma
from scipy.spatial import cKDTree
from .utils import check_idxs, build_tree_and_get_distances, get_counts_from_tree


def differential_entropy(
    data: NDArray[np.floating], k: int, idxs: tuple[int, ...] = (-1,)
) -> tuple[NDArray[np.floating], float]:
    """
    Computes the differential entropy using the Kozachenko-Leoneko estimator.

    .. math::
        \hat{H}(X) = -\psi(k)+\psi(N) + (1/N)\\sum_{i=1}^{N}\log d_i

    Parameters
    ----------
    data : NDArray[np.floating]
        Numpy array of shape (n_variables, n_samples)
    k : int
        Number of nearest neighbors
    idxs : tuple[int, ...]
        Indices of variables to use (-1 means all)

    Returns
    -------
    NDArray[np.floating]
        The local differential entropy for each sample.
    float
        The expected differential entropy over all samples

    References
    ----------
    Delattre, S., & Fournier, N. (2017).
    On the Kozachenko–Leonenko entropy estimator.
    Journal of Statistical Planning and Inference, 185, 69–93.
    https://doi.org/10.1016/j.jspi.2017.01.004

    Kozachenko, L. F., & Leonenko, N. N. (1987).
    Sample Estimate of the Entropy of a~Random Vector.
    Problems of Information Transmission, 23(2), 9.

    """

    idxs: tuple[int, ...] = check_idxs(idxs, data.shape[0])

    d: int = len(idxs)
    N: int = data.shape[1]

    psi_k: float = digamma(k)
    psi_N: float = digamma(N)

    _, distances, _ = build_tree_and_get_distances(data[idxs], k=k)

    ptw: NDArray[np.floating] = -psi_k + psi_N + d * np.log(2 * distances[:, -1])

    return ptw, ptw.mean()


def mutual_information(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    k: int,
    data: NDArray[np.floating],
    algorithm: int = 1,
) -> tuple[NDArray[np.floating], float]:
    """
    A wrapper function for the two KSG mutual information functions.

    See Also
    --------
    mutual_information_1 : Using the KSG-1 algorithm.
    mutual_information_2 : Using the KSG-2 algorithm.

    Parameters
    ----------
    idxs_x : tuple[int, ...]
        Indices of the x-variable.
    idxs_y : tuple[int, ...]
        Indices of the y-variable.
    k : int
        Number of nearest neighbors
    data : NDArray[np.floating]
        Numpy array of shape (n_variables, n_samples)
    algorithm : int
        Whether to use algorithm 1 or 2.
        Defaults to 1

    Returns
    -------
    NDArray[np.floating]
        The local mutual information for each sample.
    float
        The expected mutual information over all samples

    """

    assert algorithm in {1, 2}, "Algorithm must be 1 or 2."

    if algorithm == 1:
        return mutual_information_1(idxs_x=idxs_x, idxs_y=idxs_y, k=k, data=data)
    else:
        return mutual_information_2(idxs_x=idxs_x, idxs_y=idxs_y, k=k, data=data)


def mutual_information_1(
    idxs_x: tuple[int, ...], idxs_y: tuple[int, ...], k: int, data: NDArray[np.floating]
) -> tuple[NDArray[np.floating], float]:
    """
    Computes the Kraskov, Stogbauer, Grassberger estimate of the bivariate mutual information
    using the first algorithm presented in Kraskov et. al., (2004)

    .. math::
        \hat{I}(X;Y) = \psi(k) - \psi(N) -\\langle \psi(x+1) + \psi(y+1)\\rangle

    Parameters
    ----------
    idxs_x : tuple[int, ...]
        Indices of the x-variable.
    idxs_y : tuple[int, ...]
        Indices of the y-variable.
    k : int
        Number of nearest neighbors
    data : NDArray[np.floating]
        Numpy array of shape (n_variables, n_samples)

    Returns
    -------
    NDArray[np.floating
        The local mutual information for each sample.
    float
        The expected mutual information over all samples

    References
    ----------
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information.
    Physical Review E, 69(6), 066138.
    https://doi.org/10.1103/PhysRevE.69.066138

    """

    idxs_xy: tuple[int, ...] = idxs_x + idxs_y

    N: int = data.shape[1]

    psi_k: float = digamma(k)
    psi_N: float = digamma(N)

    distances: NDArray[np.floating]
    _, distances, _ = build_tree_and_get_distances(data[idxs_xy, :], k=k)

    ptw: NDArray[np.floating] = np.full(N, psi_k + psi_N)
    for idxs in (idxs_x, idxs_y):
        tree, _, _ = build_tree_and_get_distances(data[idxs, :], k=k)
        counts: NDArray[np.integer] = get_counts_from_tree(
            tree, data[idxs, :], distances[:, -1]
        )
        ptw -= digamma(counts + 1)

    return ptw, ptw.mean()


def mutual_information_2(
    idxs_x: tuple[int, ...], idxs_y: tuple[int, ...], k: int, data: NDArray[np.floating]
) -> tuple[NDArray[np.floating], float]:
    """
    Computes the Kraskov, Stogbauer, Grassberger estimate of the bivariate mutual information
    using the second algorithm presented in Kraskove et. al., (2004).

    .. math::
        \hat{I}(X;Y) = \psi(k) - \\frac{1}{k} - \psi(N) - \langle \psi(x) + \ldots + \psi(y) \\rangle

    Parameters
    ----------
    idxs_x : tuple[int, ...]
        Indices of the x-variable.
    idxs_y : tuple[int, ...]
        Indices of the y-variable.
    k : int
        Number of nearest neighbors
    data : NDArray[np.floating]
        Numpy array of shape (n_variables, n_samples)

    Returns
    -------
    NDArray[np.floating
        The local mutual information for each sample.
    float
        The expected mutual information over all samples

    References
    ----------
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information.
    Physical Review E, 69(6), 066138.
    https://doi.org/10.1103/PhysRevE.69.066138

    """

    idxs_xy: tuple[int, ...] = idxs_x + idxs_y

    N: int = data.shape[1]

    psi_k: float = digamma(k)
    psi_N: float = digamma(N)

    _, distances, indices = build_tree_and_get_distances(data[idxs_xy, :], k=k)
    neighbors: NDArray[np.integer] = indices[:, 1:]

    ptw: NDArray[np.floating] = np.full(N, psi_k - (1 / k) + psi_N)
    for idxs in (idxs_x, idxs_y):
        data_idx: NDArray[np.floating] = data[idxs, :].T
        eps: NDArray[np.floating] = np.repeat(-np.inf, N)

        for j in range(k):
            norm: NDArray[np.floating] = np.linalg.norm(
                data_idx - data_idx[neighbors[:, j]], ord=np.inf, axis=1
            )
            eps = np.maximum(norm, eps)

        tree, distances, _ = build_tree_and_get_distances(data_idx.T, k=k)
        counts = get_counts_from_tree(tree, data_idx.T, eps)
        ptw -= digamma(counts)

    return ptw, ptw.mean()


def conditional_mutual_information_1(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    idxs_z: tuple[int, ...],
    k: int,
    data: NDArray[np.floating],
) -> tuple[NDArray[np.floating], float]:
    """
    Computes the conditional mutual information I(X;Y|Z) using the KSG algorithm 1
    as described in Frenzel & Pompe (2007).

    The conditional mutual information is estimated using:

    .. math::
        \\hat{I}(X;Y|Z) = \\psi(k) - \\langle \\psi(n_{xz}+1) + \\psi(n_{yz}+1) - \\psi(n_z+1) \\rangle

    where n_{xz}, n_{yz}, and n_z are the counts of neighbors within the epsilon-ball
    in the respective subspaces.

    Parameters
    ----------
    idxs_x : tuple[int, ...]
        Indices of the x-variable.
    idxs_y : tuple[int, ...]
        Indices of the y-variable.
    idxs_z : tuple[int, ...]
        Indices of the conditioning variable z.
    k : int
        Number of nearest neighbors
    data : NDArray[np.floating]
        Numpy array of shape (n_variables, n_samples)

    Returns
    -------
    NDArray[np.floating]
        The local conditional mutual information for each sample.
    float
        The expected conditional mutual information over all samples

    References
    ----------
    Frenzel, S., & Pompe, B. (2007).
    Partial Mutual Information for Coupling Analysis of Multivariate Time Series.
    Physical Review Letters, 99(20), 204101.
    https://doi.org/10.1103/PhysRevLett.99.204101

    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information.
    Physical Review E, 69(6), 066138.
    https://doi.org/10.1103/PhysRevE.69.066138

    """

    idxs_x = check_idxs(idxs_x, data.shape[0])
    idxs_y = check_idxs(idxs_y, data.shape[0])
    idxs_z = check_idxs(idxs_z, data.shape[0])

    # Combine indices for joint space
    idxs_xyz: tuple[int, ...] = idxs_x + idxs_y + idxs_z
    idxs_xz: tuple[int, ...] = idxs_x + idxs_z
    idxs_yz: tuple[int, ...] = idxs_y + idxs_z

    N: int = data.shape[1]
    psi_k: float = digamma(k)

    # Find k-nearest neighbors in the joint space (X, Y, Z)
    _, distances, _ = build_tree_and_get_distances(data[idxs_xyz, :], k=k)
    eps: NDArray[np.floating] = distances[:, -1]

    # Build trees for each conditional subspace
    tree_xz, _, _ = build_tree_and_get_distances(data[idxs_xz, :], k=k)
    tree_yz, _, _ = build_tree_and_get_distances(data[idxs_yz, :], k=k)
    tree_z, _, _ = build_tree_and_get_distances(data[idxs_z, :], k=k)

    # Count neighbors in each subspace within epsilon
    n_xz: NDArray[np.integer] = get_counts_from_tree(tree_xz, data[idxs_xz, :], eps)
    n_yz: NDArray[np.integer] = get_counts_from_tree(tree_yz, data[idxs_yz, :], eps)
    n_z: NDArray[np.integer] = get_counts_from_tree(tree_z, data[idxs_z, :], eps)

    # Compute local conditional mutual information
    ptw: NDArray[np.floating] = (
        psi_k - digamma(n_xz + 1) - digamma(n_yz + 1) + digamma(n_z + 1)
    )

    return ptw, ptw.mean()


def conditional_mutual_information_2(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    idxs_z: tuple[int, ...],
    k: int,
    data: NDArray[np.floating],
) -> tuple[NDArray[np.floating], float]:
    """
    Computes the conditional mutual information I(X;Y|Z) using the KSG algorithm 2
    as adapted by Wibral et al. (2014) from Frenzel & Pompe (2007).

    The conditional mutual information is estimated using:

    .. math::
        \\hat{I}(X;Y|Z) = \\psi(k) - 2/k + \\langle \\psi(n_z) - \\psi(n_{xz}) - \\psi(n_{yz}) + 1/n_{xz} + 1/n_{yz} \\rangle

    where epsilon_x, epsilon_y, and epsilon_z are the maximum norms in each subspace
    among the k nearest neighbors in the joint space.

    Parameters
    ----------
    idxs_x : tuple[int, ...]
        Indices of the x-variable.
    idxs_y : tuple[int, ...]
        Indices of the y-variable.
    idxs_z : tuple[int, ...]
        Indices of the conditioning variable z.
    k : int
        Number of nearest neighbors
    data : NDArray[np.floating]
        Numpy array of shape (n_variables, n_samples)

    Returns
    -------
    NDArray[np.floating]
        The local conditional mutual information for each sample.
    float
        The expected conditional mutual information over all samples

    References
    ----------
    Wibral, M., Vicente, R., & Lindner, M. (2014).
    Transfer Entropy in Neuroscience.
    In Directed Information Measures in Neuroscience (pp. 3-36). Springer.
    https://doi.org/10.1007/978-3-642-54474-3_1

    Frenzel, S., & Pompe, B. (2007).
    Partial Mutual Information for Coupling Analysis of Multivariate Time Series.
    Physical Review Letters, 99(20), 204101.
    https://doi.org/10.1103/PhysRevLett.99.204101

    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information.
    Physical Review E, 69(6), 066138.
    https://doi.org/10.1103/PhysRevE.69.066138

    """

    idxs_x = check_idxs(idxs_x, data.shape[0])
    idxs_y = check_idxs(idxs_y, data.shape[0])
    idxs_z = check_idxs(idxs_z, data.shape[0])

    # Combine indices for joint space
    idxs_xyz: tuple[int, ...] = idxs_x + idxs_y + idxs_z
    idxs_xz: tuple[int, ...] = idxs_x + idxs_z
    idxs_yz: tuple[int, ...] = idxs_y + idxs_z

    N: int = data.shape[1]
    psi_k: float = digamma(k)
    
    # Determine inverse k term based on whether there's a conditional
    inverse_k_term: float = 2.0 / k if len(idxs_z) > 0 else 1.0 / k

    # Find k-nearest neighbors in the joint space (X, Y, Z)
    _, distances, indices = build_tree_and_get_distances(data[idxs_xyz, :], k=k)
    neighbors: NDArray[np.integer] = indices[:, 1:]

    # For each sample, find the maximum norm in each subspace among k neighbors
    data_x: NDArray[np.floating] = data[idxs_x, :].T
    data_y: NDArray[np.floating] = data[idxs_y, :].T
    data_z: NDArray[np.floating] = data[idxs_z, :].T

    eps_x: NDArray[np.floating] = np.repeat(-np.inf, N)
    eps_y: NDArray[np.floating] = np.repeat(-np.inf, N)
    eps_z: NDArray[np.floating] = np.repeat(-np.inf, N)

    for j in range(k):
        norm_x: NDArray[np.floating] = np.linalg.norm(
            data_x - data_x[neighbors[:, j]], ord=np.inf, axis=1
        )
        norm_y: NDArray[np.floating] = np.linalg.norm(
            data_y - data_y[neighbors[:, j]], ord=np.inf, axis=1
        )
        norm_z: NDArray[np.floating] = np.linalg.norm(
            data_z - data_z[neighbors[:, j]], ord=np.inf, axis=1
        )
        eps_x = np.maximum(norm_x, eps_x)
        eps_y = np.maximum(norm_y, eps_y)
        eps_z = np.maximum(norm_z, eps_z)

    # Build trees for each conditional subspace
    tree_xz, _, _ = build_tree_and_get_distances(data[idxs_xz, :], k=k)
    tree_yz, _, _ = build_tree_and_get_distances(data[idxs_yz, :], k=k)
    tree_z, _, _ = build_tree_and_get_distances(data[idxs_z, :], k=k)

    # Count neighbors within or on the boundary in each subspace
    # For algorithm 2, we use <= instead of < for the radius check
    n_xz: NDArray[np.integer] = np.array([
        tree_xz.query_ball_point(data[idxs_xz, i], r=max(eps_x[i], eps_z[i]), p=np.inf, return_length=True)
        for i in range(N)
    ])
    
    n_yz: NDArray[np.integer] = np.array([
        tree_yz.query_ball_point(data[idxs_yz, i], r=max(eps_y[i], eps_z[i]), p=np.inf, return_length=True)
        for i in range(N)
    ])
    
    n_z: NDArray[np.integer] = np.array([
        tree_z.query_ball_point(data[idxs_z, i], r=eps_z[i], p=np.inf, return_length=True)
        for i in range(N)
    ])

    # Compute inverse terms (set to 0 if no conditional variable)
    inv_n_xz: NDArray[np.floating] = 1.0 / n_xz if len(idxs_z) > 0 else np.zeros(N)
    inv_n_yz: NDArray[np.floating] = 1.0 / n_yz if len(idxs_z) > 0 else np.zeros(N)

    # Compute local conditional mutual information
    ptw: NDArray[np.floating] = (
        psi_k - inverse_k_term
        + digamma(n_z) - digamma(n_xz) - digamma(n_yz)
        + inv_n_xz + inv_n_yz
    )

    return ptw, ptw.mean()

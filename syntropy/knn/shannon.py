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

    _, distances, _ = build_tree_and_get_distances(data[idxs, :], k=k)

    ptw: NDArray[np.floating] = np.zeros((1, N))
    ptw[0, :] += -psi_k + psi_N + d * np.log(2 * distances[:, -1])

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

    ptw: NDArray[np.floating] = np.full((1, N), psi_k + psi_N)
    for idxs in (idxs_x, idxs_y):
        tree, _, _ = build_tree_and_get_distances(data[idxs, :], k=k)
        counts: NDArray[np.integer] = get_counts_from_tree(
            tree, data[idxs, :], distances[:, -1]
        )
        ptw[0, :] -= digamma(counts + 1)

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

    ptw: NDArray[np.floating] = np.full((1, N), psi_k - (1 / k) + psi_N)
    for idxs in (idxs_x, idxs_y):
        data_idx: NDArray[np.floating] = data[idxs, :].T
        eps: NDArray[np.floating] = np.repeat(-np.inf, N)

        for j in range(k):
            norm: NDArray[np.floating] = np.linalg.norm(
                data_idx - data_idx[neighbors[:, j]], ord=np.inf, axis=1
            )
            eps = np.maximum(norm, eps)

        tree, distances, _ = build_tree_and_get_distances(data_idx.T, k=k)
        counts = get_counts_from_tree(tree, data_idx.T, eps, strict=False)
        ptw[0, :] -= digamma(counts)

    return ptw, ptw.mean()


def conditional_mutual_information(
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

    N: int = data.shape[1]
    psi_k: float = digamma(k)

    idxs_xz: tuple[int, ...] = idxs_x + idxs_z
    idxs_yz: tuple[int, ...] = idxs_y + idxs_z
    idxs_joint: tuple[int, ...] = idxs_x + idxs_y + idxs_z

    ptw: NDArray[np.floating] = np.full((1, N), psi_k)

    distances: NDArray[np.floating]
    _, distances, _ = build_tree_and_get_distances(data=data[idxs_joint, :], k=k)

    counter: int = 0
    for idxs in (idxs_z, idxs_xz, idxs_yz):
        tree, _, _ = build_tree_and_get_distances(data[idxs, :], k=k)
        counts: NDArray[np.integer] = get_counts_from_tree(
            tree, data[idxs, :], distances[:, -1]
        )

        if counter == 0:
            ptw[0, :] += digamma(counts + 1)
        else:
            ptw[0, :] -= digamma(counts + 1)
        counter += 1

    return ptw, ptw.mean()

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


def build_tree_and_get_distances(
    data: NDArray[np.floating], k: int, p: float = np.inf
) -> tuple[cKDTree, NDArray[np.floating], NDArray[np.integer]]:
    """
    Builds the KNN tree and returns the indices and distances between each point and it's k-nearest neighbors.

    Parameters
    ----------
    data : NDArray[np.floating]
        Data array of shape (n_variables, n_samples)
    k : int
        Number of nearest neighbors

    Returns
    -------
    cKDTree
        The KNN tree constructed from data.
    NDArray[np.floating]
        The indices of each of the k-nearest neighbors.
    NDArray[np.integer]
        The distances to each k-nearest neighbors (using the max norm).

    """
    tree = cKDTree(data.T)

    distances: NDArray[np.floating]
    indices: NDArray[np.integer]
    distances, indices = tree.query(data.T, k=k + 1, p=p)

    return tree, distances, indices


def get_counts_from_tree(
    tree: cKDTree,
    data: NDArray[np.floating],
    eps: NDArray[np.floating],
    strict: bool = True,
    p: float = np.inf,
) -> NDArray[np.integer]:
    if strict is True:
        eps_ = np.nextafter(eps, -np.inf)
    else:
        eps_ = eps
    counts: NDArray[np.integer] = np.array(
        [
            tree.query_ball_point(x_i, eps_i, return_length=True, p=p) - 1
            for x_i, eps_i in zip(data.T, eps_)
        ]
    )

    return counts

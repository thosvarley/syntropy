import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma
from scipy.spatial import cKDTree


def check_idxs(idxs: tuple[int, ...], N: int) -> tuple[int, ...]:
    """

    Parameters
    ----------
    idxs : tuple[int, ...]

    N : int


    Returns
    -------
    tuple[int, ...]


    """
    if idxs[0] == -1:
        idxs_ = tuple(i for i in range(N))
    else:
        idxs_ = idxs

    return idxs_


def build_tree_and_get_distances(
    data: NDArray[np.floating], k: int
) -> tuple[cKDTree, NDArray[np.floating], NDArray[np.integer]]:
    """

    Parameters
    ----------
    data : NDArray[np.floating]

    k : int


    Returns
    -------
    tuple[cKDTree, NDArray[np.floating], NDArray[np.integer]]


    """
    tree = cKDTree(data.T)

    distances: NDArray[np.floating]
    indices: NDArray[np.integer]
    distances, indices = tree.query(data.T, k=k + 1, p=np.inf)

    return tree, distances, indices


def get_counts_from_tree(
    tree: cKDTree, data: NDArray[np.floating], eps: NDArray[np.floating]
) -> NDArray[np.integer]:
    counts: NDArray[np.integer] = np.array(
        [
            tree.query_ball_point(x_i, eps_i, return_length=True, p=np.inf) - 1
            for x_i, eps_i in zip(data.T, eps)
        ]
    )

    return counts

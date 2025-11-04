#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 18:11:42 2025

@author: thosvarley
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma
from scipy.spatial import cKDTree
from utils import check_idxs, build_tree_and_get_distances, get_counts_from_tree

def differential_entropy(
    data: NDArray[np.floating], k: int, idxs: tuple[int, ...] = (-1,)
) -> tuple[NDArray[np.floating], float]:
    """

        Parameters
        ----------
        data : NDArray[np.floating]

        k : int

        idxs : tuple[int, ...]
    :

        Returns
        -------
        tuple[NDArray[np.floating], float]


    """
    idxs_: tuple[int, ...] = check_idxs(idxs, data.shape[0])

    d: int = len(idxs_)
    N: int = data.shape[1]

    psi_k: float = digamma(k)
    psi_N: float = digamma(N)

    _, distances, _ = build_tree_and_get_distances(data[idxs_], k=k)

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
        return mutual_information_1(idxs_x=idxs_x, idxs_y=idxs_y, k=k, data=data)
    else:
        return mutual_information_2(idxs_x=idxs_x, idxs_y=idxs_y, k=k, data=data)


def mutual_information_1(
    idxs_x: tuple[int, ...], idxs_y: tuple[int, ...], k: int, data: NDArray[np.floating]
) -> tuple[NDArray[np.floating], float]:
    """
    Computes the Kraskov, Stogbauer, Grassberger estimate of the bivariate mutual information
    using the first algorithm presented in:

    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information.
    Physical Review E, 69(6), 066138.
    https://doi.org/10.1103/PhysRevE.69.066138

    Parameters
    ----------
    idxs_x : tuple[int, ...]

    idxs_y : tuple[int, ...]

    k : int

    data : NDArray[np.floating]


    Returns
    -------
    tuple[NDArray[np.floating], float]


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
    using the second algorithm presented in:

    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information.
    Physical Review E, 69(6), 066138.
    https://doi.org/10.1103/PhysRevE.69.066138



    Parameters
    ----------
    idxs_x : tuple[int, ...]

    idxs_y : tuple[int, ...]

    k : int

    data : NDArray[np.floating]


    Returns
    -------
    tuple[NDArray[np.floating], float]


    """
    idxs_xy: tuple[int, ...] = idxs_x + idxs_y

    N: int = data.shape[1]

    psi_k: float = digamma(k)
    psi_N: float = digamma(N)

    _, distances, indices = build_tree_and_get_distances(data[idxs_xy, :], k=k)
    neighbors: NDArray[np.integer] = indices[:, 1:]
    
    ptw: NDArray[np.floating] = np.full(N, psi_k - (1/k) + psi_N)
    for idxs in (idxs_x, idxs_y):

        data_idx: NDArray[np.floating] = data[idxs, :].T
        eps: NDArray[np.floating] = np.repeat(-np.inf, N)
        
        for j in range(k):
            norm: NDArray[np.floating] = np.linalg.norm(data_idx - data_idx[neighbors[:, j]], ord=np.inf, axis=1)
            eps = np.maximum(norm, eps)

        tree, distances, _ = build_tree_and_get_distances(data_idx.T, k=k)
        counts = get_counts_from_tree(tree, data_idx.T, eps)
        ptw -= digamma(counts)

    return ptw, ptw.mean() 

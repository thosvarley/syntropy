import numpy as np
from .utils import get_marginal_distribution

from typing import Any

DiscreteDist = dict[tuple[Any, ...], float]

def shannon_entropy(joint_distribution: DiscreteDist) -> tuple[dict, float]:
    """
    Computes the Shannon entropy of the distribution :math:`P(x)`.

    .. math::
        H(X) = -\\sum_{x} P(x) \\log P(x)

    To compute the entropy of a subset of the variables in the joint
    distribution, use the :func:`syntropy.discrete.utils.get_marginals` function from the utils
    library.

    Parameters
    ----------
    joint_distribution : dict
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    ptw : dict
        The pointwise entropy for each state in the joint distribution.
    avg : float
        The average entropy

    """

    ptw: dict = {
        state: -np.log2(joint_distribution[state])
        for state in joint_distribution.keys()
        if joint_distribution[state] > 0.0
    }
    avg: float = sum(joint_distribution[state] * ptw[state] for state in ptw.keys())

    return ptw, avg


def conditional_entropy(
    idxs_x: tuple[int, ...], idxs_y: tuple[int, ...], joint_distribution: DiscreteDist
) -> tuple[dict, float]:
    """
    Computes the conditional entropy of X given Y.

    :math:`H(X|Y) = H(X,Y) - H(Y)`

    Parameters
    ----------
    idxs_x : tuple
        The indices of the variables to compute the entropy on.
    idxs_y : tuple
        The indicies of the variables to contintue on.
    joint_distribution : dict
        DESCRIPTION.joint_distribution : dict
            The joint probability distribution.
            Keys are tuples corresponding to the state of each element.
            The valules are the probabilities.

    Returns
    -------
    ptw : dict
        The pointwise entropy for each state in the joint distribution.
    avg : float
        The average entropy

    """

    Nx: int = len(idxs_x)

    idxs_xy: tuple[int, ...] = idxs_x + idxs_y

    marginal_xy: DiscreteDist = get_marginal_distribution(idxs_xy, joint_distribution)
    marginal_y: DiscreteDist = get_marginal_distribution(idxs_y, joint_distribution)

    ptw: dict = {}
    avg: float = 0

    for state in marginal_xy.keys():
        if marginal_xy[state] > 0.0:
            p_y: float = marginal_y[state[Nx:]]
            h: float = -np.log2(marginal_xy[state] / p_y)
            ptw[((state[:Nx]), (state[Nx:]))] = h

            avg += marginal_xy[state] * h

    return ptw, avg


def mutual_information(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    joint_distribution: DiscreteDist,
) -> tuple[dict, float]:
    """
    Computes the mutual information between X and Y.

    :math:`I(X;Y) = H(X) + H(Y) - H(X,Y)`

    Parameters
    ----------
    idxs_x : tuple
        The indices of the X variable(s).
    idxs_y : tuple
        The indices of the Y variable(s).
    joint_distribution : dict
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.


    Returns
    -------
    ptw : dict
        The pointwise mutual information for each state in the joint distribution.
    avg : float
        The average mutual information

    """

    Nx: int = len(idxs_x)

    idxs_xy: tuple[int, ...] = idxs_x + idxs_y

    marginal_xy: DiscreteDist = get_marginal_distribution(idxs_xy, joint_distribution)
    marginal_y: DiscreteDist = get_marginal_distribution(idxs_y, joint_distribution)
    marginal_x: DiscreteDist = get_marginal_distribution(idxs_x, joint_distribution)

    ptw: dict = {}
    avg: float = 0.0

    for state in marginal_xy.keys():
        if marginal_xy[state] > 0.0:
            p_x: float = marginal_x[state[:Nx]]
            p_y: float = marginal_y[state[Nx:]]

            mi = np.log2(marginal_xy[state] / (p_x * p_y))

            ptw[((state[:Nx]), (state[Nx:]))] = mi

            avg += mi * marginal_xy[state]

    return ptw, avg


def conditional_mutual_information(
    idxs_x: tuple, idxs_y: tuple, idxs_z: tuple, joint_distribution: DiscreteDist
) -> tuple[dict, float]:
    """
    Computes the mutual information between X and Y condioned on Z.

    :math:`I(X,Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)`

    Parameters
    ----------
    idxs_x : tuple
        The indices of the X variable(s).
    idxs_y : tuple
        The indices of the Y variable(s).
    idxs_z : tuple
        The indices of the variables to condition on.
    joint_distribution : dict
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    ptw : dict
        The pointwise mutual information for each state in the joint distribution.
    avg : float
        The average mutual information

    """

    Nx: int = len(idxs_x)

    joint: tuple[int, ...] = idxs_x + idxs_y

    ptw_xz, avg_xz = conditional_entropy(idxs_x, idxs_z, joint_distribution)
    ptw_yz, avg_yz = conditional_entropy(idxs_y, idxs_z, joint_distribution)
    ptw_xyz, avg_xyz = conditional_entropy(joint, idxs_z, joint_distribution)

    avg: float = avg_xz + avg_yz - avg_xyz
    ptw: dict = {}

    for state in ptw_xyz.keys():
        sx = state[0][:Nx]
        sy = state[0][Nx:]

        sxz = (sx, state[1])
        syz = (sy, state[1])

        ptw[(sx, sy, state[1])] = ptw_xz[sxz] + ptw_yz[syz] - ptw_xyz[state]

    return ptw, avg


def kullback_leibler_divergence(
    posterior_distribution: DiscreteDist, prior_distribution: DiscreteDist
) -> tuple[dict, float]:
    """
    Computes the Kullback-Leibler divergence from a prior distribution P(X) and
    and posterior distribution Q(X).

    :math:`D_{KL}(P||Q) = \\sum_{x} P(x) \\log \\frac{P(x)}{Q(x)}`

    Parameters
    ----------
    posterior_distribution : dict
        The joint distribution of the posterior distribution P(X).
    prior_distribution : dict
        The joint distribution of the prior distribution Q(X)

    Returns
    -------
    ptw : dict
        The pointwise Kullback-Leibler divergence for each state in the joint distribution.
    avg : float
        The average Kullback-Leibler divergence.

    """

    assert set(prior_distribution.keys()).issuperset(
        set(posterior_distribution.keys())
    ), "The support set of the prior must be a superset of the posterior"

    avg: float = 0
    ptw: dict = {state: 0 for state in posterior_distribution.keys()}
    for state in posterior_distribution.keys():
        log_ratio: float = np.log2(
            posterior_distribution[state] / prior_distribution[state]
        )

        avg += posterior_distribution[state] * log_ratio
        ptw[state] = log_ratio

    return ptw, avg

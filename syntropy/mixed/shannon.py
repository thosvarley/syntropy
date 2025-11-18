import numpy as np
from numpy.typing import NDArray

from .gaussian import differential_entropy, local_differential_entropy


def shannon_entropy(
    discrete_vars: NDArray[np.integer], continuous_vars: NDArray[np.floating]
) -> tuple[NDArray[np.floating], float]:
    """

    Parameters
    ----------
    discrete_vars : NDArray[np.integer]

    continuous_vars : NDArray[np.floating]


    Returns
    -------
    tuple[NDArray[np.floating], float]


    """
    assert discrete_vars.shape[1] == continuous_vars.shape[1], (
        "The discrete and continuous variables must have the same number of samples."
    )

    unq, counts = np.unique(discrete_vars, axis=1, return_counts=True)
    probs = counts / counts.sum()

    avg = -np.sum(probs * np.log(probs))
    ptw = np.zeros(continuous_vars.shape[1])

    conditional_entropy = 0.0
    for i in range(unq.shape[1]):
        state = unq[:, [i]]
        mask = np.all(discrete_vars == state, axis=0)

        ptw[mask] -= np.log(probs[i])

        cov_conditional = np.cov(continuous_vars[:, mask], ddof=0)

        ptw[mask] += local_differential_entropy(
            continuous_vars[:, mask], cov_conditional
        )
        conditional_entropy += probs[i] * differential_entropy(cov_conditional)

    avg += conditional_entropy

    return ptw, avg


def conditional_entropy(
    discrete_vars: NDArray[np.integer],
    continuous_vars: NDArray[np.floating],
    conditional: str = "discrete",
) -> tuple[NDArray[np.floating], float]:
    """
    For a discrete, potetially multivariate, random variable Y and a continuous, potentially multivariate random variable X, this estimator exploits the identity:

    .. math::
        H(X,Y) = H(Y|X) + H(Y)
    
    Where :math:'H(Y)' can be computed with the usual :math:'-\\sum P(x)\\log P(x)'

    Parameters
    ----------
    discrete_vars : NDArray[np.integer]

    continuous_vars : NDArray[np.floating]

    conditional : str


    Returns
    -------
    tuple[NDArray[np.floating], float]


    """
    # %%
    assert discrete_vars.shape[1] == continuous_vars.shape[1], (
        "The discrete and continuous variables must have the same number of samples."
    )
    assert conditional in {
        "discrete",
        "continuous",
    }, "Please specify whether the conditioning variable is discrete or continuous."

    ptw_joint, avg_joint = shannon_entropy(
        discrete_vars=discrete_vars, continuous_vars=continuous_vars
    )

    if conditional == "continuous":
        cov = np.cov(continuous_vars, ddof=0)
        ptw_marginal = local_differential_entropy(continuous_vars, cov=cov)
        avg_marginal = differential_entropy(cov)
    elif conditional == "discrete":
        unq, counts = np.unique(discrete_vars, axis=1, return_counts=True)
        probs = counts / counts.sum()
        ptw_marginal = np.zeros(discrete_vars.shape[1])
        avg_marginal = -np.sum(probs * np.log(probs))

        for i in range(unq.shape[1]):
            state = unq[:, [i]]
            mask = np.all(discrete_vars == state, axis=0)

            ptw_marginal[mask] = -np.log(probs[i])

    avg = avg_joint - avg_marginal
    ptw = ptw_joint - ptw_marginal
    # %%

    return ptw, avg


def mutual_information(
    discrete_vars: NDArray[np.integer], continuous_vars: NDArray[np.floating]
) -> tuple[NDArray[np.floating], float]:
    """

    Parameters
    ----------
    discrete_vars : NDArray[np.integer]

    continuous_vars : NDArray[np.floating]


    Returns
    -------
    tuple[NDArray[np.floating], float]


    """
    assert discrete_vars.shape[1] == continuous_vars.shape[1], (
        "The discrete and continuous variables must have the same number of samples."
    )

    cov = np.cov(continuous_vars, ddof=0)

    avg = differential_entropy(cov)
    ptw = local_differential_entropy(continuous_vars, cov)

    unq, counts = np.unique(discrete_vars, axis=1, return_counts=True)
    probs = counts / counts.sum()

    conditional_entropy = 0.0
    for i in range(unq.shape[1]):
        state = unq[:, [i]]
        mask = np.all(discrete_vars == state, axis=0)

        cov_conditional = np.cov(continuous_vars[:, mask], ddof=0)
        conditional_entropy += probs[i] * differential_entropy(cov_conditional)

        local_conditional = local_differential_entropy(
            continuous_vars[:, mask], cov_conditional
        )
        ptw[mask] -= local_conditional

    avg -= conditional_entropy

    return avg, ptw

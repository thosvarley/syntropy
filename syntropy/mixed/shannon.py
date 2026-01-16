import numpy as np
from numpy.typing import NDArray
from .. import gaussian
from .. import knn


def shannon_entropy(
    discrete_vars: NDArray[np.integer],
    continuous_vars: NDArray[np.floating],
    continuous_estimator: str = "gaussian",
    k: int = 5,
) -> tuple[NDArray[np.floating], float]:
    """
    For discrete :math:`X` and continuous :math:`Y`, leverages the identity 

    .. math::
        H(X,Y) = H(X) + H(Y | X)

    Parameters
    ----------
    discrete_vars : NDArray[np.integer]
        Numpy array of the discrete variables, of shape (n_variables, n_samples)
    continuous_vars : NDArray[np.floating]
        Numpy array of the continuous variables, of shape (n_variables, n_samples)
    continuous_estimator : str
        Whether to use a Gaussian or KNN-based estimator of the continuous entropy. Options are "gaussian" or "knn".
    k : int (optional)
        If a KNN-based estimator is selected, sets the k-value. Defaults to 5.

    Returns
    -------
    NDArray[np.floating]
        The local entropy for each sample.
    floating
        The expected entropy over all samples.

    """
    assert discrete_vars.shape[1] == continuous_vars.shape[1], (
        "The discrete and continuous variables must have the same number of samples."
    )
    assert continuous_estimator in {"gaussian", "knn"}, "Continuous estimator options are 'gaussian' or 'knn'."

    unq: NDArray[np.integer]
    counts: NDArray[np.integer]
    unq, counts = np.unique(discrete_vars, axis=1, return_counts=True)

    probs: NDArray[np.floating] = counts / counts.sum()

    avg: float = -np.sum(probs * np.log(probs))
    ptw: NDArray[np.floating] = np.zeros((1, continuous_vars.shape[1]))

    conditional_entropy: float = 0.0
    for i in range(unq.shape[1]):
        state: NDArray[np.integer] = unq[:, [i]]
        mask: NDArray[np.bool_] = np.all(discrete_vars == state, axis=0)

        ptw[:, mask] -= np.log(probs[i])

        cov_conditional: NDArray[np.floating] = np.cov(continuous_vars[:, mask], ddof=0)

        ptw[:, mask] += gaussian.local_differential_entropy(
            continuous_vars[:, mask], cov_conditional
        )
        conditional_entropy += probs[i] * gaussian.differential_entropy(cov_conditional)

    avg += conditional_entropy

    return ptw, avg


def conditional_entropy(
    discrete_vars: NDArray[np.integer],
    continuous_vars: NDArray[np.floating],
    conditional: str = "discrete",
) -> tuple[NDArray[np.floating], float]:
    """
    For discrete :math:`X` and continuous :math:`Y`, leverages the identity 

    .. math::
        H(X,Y) = H(X) + H(Y | X)

    Either the discrete variables or the continuous variables can be conditioning variable (in which case the other is conditioned).

    Parameters
    ----------
    discrete_vars : NDArray[np.integer]
        Numpy array of the discrete variables, of shape (n_variables, n_samples)
    continuous_vars : NDArray[np.floating]
        Numpy array of the continuous variables, of shape (n_variables, n_samples)
    conditional : str
        Wheter to condition the discrete variables on the continuous, or vice versa.


    Returns
    -------
    NDArray[np.floating]
        The local entropy for each sample.
    floating
        The expected entropy over all samples.

    """
    # %%
    assert discrete_vars.shape[1] == continuous_vars.shape[1], (
        "The discrete and continuous variables must have the same number of samples."
    )
    assert conditional in {
        "discrete",
        "continuous",
    }, "Please specify whether the conditioning variable is discrete or continuous."

    ptw_joint: NDArray[np.floating]
    avg_joint: float
    ptw_joint, avg_joint = shannon_entropy(
        discrete_vars=discrete_vars, continuous_vars=continuous_vars
    )

    if conditional == "continuous":
        cov: NDArray[np.floating] = np.cov(continuous_vars, ddof=0)
        ptw_marginal: NDArray[np.floating] = gaussian.local_differential_entropy(
            continuous_vars, cov=cov
        )
        avg_marginal: float = gaussian.differential_entropy(cov)
    elif conditional == "discrete":
        unq: NDArray[np.integer]
        counts: NDArray[np.integer]
        unq, counts = np.unique(discrete_vars, axis=1, return_counts=True)
        probs: NDArray[np.floating] = counts / counts.sum()

        ptw_marginal: NDArray[np.floating] = np.zeros((1, discrete_vars.shape[1]))
        avg_marginal: float = -np.sum(probs * np.log(probs))

        for i in range(unq.shape[1]):
            state: NDArray[np.integer] = unq[:, [i]]
            mask: NDArray[np.bool_] = np.all(discrete_vars == state, axis=0)

            ptw_marginal[:, mask] = -np.log(probs[i])

    avg: float = avg_joint - avg_marginal
    ptw: NDArray[np.floating] = ptw_joint - ptw_marginal
    # %%

    return ptw, avg


def mutual_information(
    discrete_vars: NDArray[np.integer], continuous_vars: NDArray[np.floating]
) -> tuple[NDArray[np.floating], float]:
    """
    Returns the local and average mutual information between two (potentially multivariate) discrete and continuous random variables.

    Parameters
    ----------
    discrete_vars : NDArray[np.integer]
        Numpy array of the discrete variables, of shape (n_variables, n_samples)
    continuous_vars : NDArray[np.floating]
        Numpy array of the continuous variables, of shape (n_variables, n_samples)


    Returns
    -------
    NDArray[np.floating]
        The local entropy for each sample.
    floating
        The expected entropy over all samples.


    """
    assert discrete_vars.shape[1] == continuous_vars.shape[1], (
        "The discrete and continuous variables must have the same number of samples."
    )

    cov: NDArray[np.floating] = np.cov(continuous_vars, ddof=0)

    avg: float = gaussian.differential_entropy(cov)
    ptw: NDArray[np.floating] = gaussian.local_differential_entropy(
        continuous_vars, cov
    )

    unq: NDArray[np.integer]
    counts: NDArray[np.integer]
    unq, counts = np.unique(discrete_vars, axis=1, return_counts=True)
    probs: NDArray[np.floating] = counts / counts.sum()

    conditional_entropy = 0.0
    for i in range(unq.shape[1]):
        state: NDArray[np.integer] = unq[:, [i]]
        mask: NDArray[np.bool_] = np.all(discrete_vars == state, axis=0)

        cov_conditional: NDArray[np.floating] = np.cov(continuous_vars[:, mask], ddof=0)
        conditional_entropy += probs[i] * gaussian.differential_entropy(cov_conditional)

        local_conditional: NDArray[np.floating] = gaussian.local_differential_entropy(
            continuous_vars[:, mask], cov_conditional
        )
        ptw[:, mask] -= local_conditional

    avg -= conditional_entropy

    return ptw, avg

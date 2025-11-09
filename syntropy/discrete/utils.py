# import pickle
import numpy as np
import itertools as it

def make_powerset(iterable):
    """
    A utility function for quickly making powersets,

    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    """
    xs: list = list(iterable)
    # note we return an iterator rather than a list
    return it.chain.from_iterable(it.combinations(xs, n) for n in range(len(xs) + 1))


def clean_distribution(joint_distribution: dict[tuple, float]) -> dict:
    """
    A utility function to remove states with 0 probability

    Parameters
    ----------
    joint_distribution: dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    dict
        The joint probability distribution with zero-probability elements removed.

    """
    return {
        key: joint_distribution[key]
        for key in joint_distribution.keys()
        if joint_distribution[key] > 0.0
    }


def reduce_state(state: tuple, source: tuple) -> tuple:
    """
    A utility function for reducing tuples
    to just the elements in the source.

    Parameters
    ----------
    state : tuple
        The particular state of each variable.
    source : tuple
        The indices of the variable to remove.

    Returns
    -------
    tuple
        The reduced state consisting only of those
        elements indexed in the source variable.

    """

    return tuple(state[i] for i in source)


def construct_joint_distribution(data: np.ndarray) -> dict[tuple, float]:
    """
    Given a channels x time, discrete Numpy array, computes
    the probability distribution that describes the data.


    Parameters
    ----------
    data : np.ndarray
        The data: assumed to be in elements x time format.

    Returns
    -------
    dict
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    """

    assert data.dtype != "float", "The array must be discrete-valued variables."

    N0: int
    N1: int
    N0, N1 = data.shape

    unq: np.ndarray
    counts: np.ndarray
    unq, counts = np.unique(data, return_counts=True, axis=-1)

    return {tuple(unq[:, i]): counts[i] / counts.sum() for i in range(unq.shape[1])}


def get_marginal_distribution(
    idxs: tuple, joint_distribution: dict[tuple, float]
) -> dict:
    """
    Returns the marginal distribution of the variables
    indexed by the idxs tuple. The opposite of the
    marginalize_out() function.

    Parameters
    ----------
    idxs : tuple
        The indices of the variable to retain.
    joint_distribution: dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    dict
        The marginal joint probability distribution object.

    """

    states: list = list(joint_distribution.keys())

    reduced_states: set = {reduce_state(state, idxs) for state in states}
    reduced_distribution: dict = {r_state: 0.0 for r_state in reduced_states}

    for r_state in reduced_states:
        reduced_distribution[r_state] = sum(
            joint_distribution[state]
            for state in joint_distribution.keys()
            if reduce_state(state, idxs) == r_state
        )

    return reduced_distribution


def marginalize_out(idxs: tuple, joint_distribution: dict[tuple, float]) -> dict:
    """
    Returns a distribution with the variables indexed by
    idxs marginalized out.

    Parameters
    ----------
    idxs : tuple
        The indices of the variables to be marginalized out.
    joint_distribution: dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    dict
        A joint probability distribution dictionary.

    """

    N: int = len(list(joint_distribution.keys())[0])

    residuals: tuple = tuple(i for i in range(N) if i not in idxs)

    return get_marginal_distribution(residuals, joint_distribution)


def get_all_marginal_distributions(
    joint_distribution: dict[tuple, float],
) -> dict[tuple, dict]:
    """
    Computes the set of all marginal probability distributions.
    If the original distribution has variables:

        :math:`P(X_1, X_2, X_3)`

    Returns a dictionary of dictionaries for each:
        :math:`P(X_1,), P(X_2,), P(X_3,), P(X_1, X_2), P(X_1, X_3), P(X_2, X_3), P(X_1,X_2,X_3)`


    Parameters
    ----------
    joint_distribution: dict[tuple, float][tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    dict[tuple, dict]
        A dictionary of dictionaries: each key is a set of marginals,
        each value is the associated marginal distribution .

    """

    N: int = len(list(joint_distribution.keys())[0])

    sources: list = list(make_powerset(range(N)))
    sources.remove(())

    marginal_dict: dict = {
        source: get_marginal_distribution(source, joint_distribution)
        for source in sources
    }

    return marginal_dict



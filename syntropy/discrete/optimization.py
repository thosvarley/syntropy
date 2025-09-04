import numpy as np
import itertools as it
from syntropy.discrete.utils import get_marginal_distribution, reduce_state


def constrained_maximum_entropy_distributions(
    joint_distribution: dict,
    marginal_constraints: list = [(None,)],
    order: int = -1,
    max_iters: int = 10_000,
    tol: float = 1e-6,
) -> dict[tuple, float]:
    """
    Uses the iterated proportional fitting (IPF) algorithm to find
    the maximum-entropy distribution consistant with given constraints.

    If the marginal constrains are given, uses those.
    If order is given, finds the maximum entropy distribution consistant
    with all marginals of the given order.

    Parameters
    ----------
    joint_distribution : dict
        The joint distribution from which to compute the marginal constraints.
    marginal_constraints : list, optional
        A list of tuples: each tuple corresponds to a set of marginals to constrain. The default is (None,).
    order : int, optional
        The order of marginals to fix. The default is -1.
    max_iters : int, optional
        The maximum number of iterations the algorithm can run for. The default is 10_000.
    tol : float, optional
        The value below which further refinements of the maximum entropy distribution are stopped.
        The default is 1e-6.

    Returns
    -------
    dict[tuple, float]
        The optimized distribution consistent with the given marginal constraints.

    """

    assert (marginal_constraints == [(None,)]) ^ (
        order == -1
    ), "Give just one: fixed constraints or marginal order."

    N = len(list(joint_distribution.keys())[0])

    if order != -1:
        marginal_constraints = list(it.combinations(range(N), r=order))

    marginals = {
        m: get_marginal_distribution(m, joint_distribution)
        for m in marginal_constraints
    }

    first_order_marginals = [
        get_marginal_distribution((i,), joint_distribution) for i in range(N)
    ]
    first_order_states = [list(m.keys()) for m in first_order_marginals]
    # Evil tuple unpacking. This works but I hate it.
    first_order_states = [
        tuple(item for tuple in first_order_states[x] for item in tuple)
        for x in range(len(first_order_states))
    ]

    maxent_states = list(it.product(*first_order_states))

    if (
        order == 1
    ):  # If the order is 1, return the product of the first-order marginals.
        maxent = {}

        for state in maxent_states:
            maxent[state] = np.prod(
                [first_order_marginals[i][(state[i],)] for i in range(N)]
            )

        # return maxent
    else:  # Otherwise do the IPF algorithm.
        maxent = {state: 1 / len(maxent_states) for state in maxent_states}

        for _ in range(max_iters):

            prev = maxent.copy()
            for m in marginal_constraints:

                target_marg = marginals[m]
                maxent_marg = get_marginal_distribution(m, maxent)

                scaling_factors = {
                    key: (
                        target_marg[key] / maxent_marg[key]
                        if maxent_marg[key] > 0
                        else 0
                    )
                    for key in target_marg.keys()
                }

                maxent = {
                    state: (
                        maxent[state] * scaling_factors[reduce_state(state, m)]
                        if reduce_state(state, m) in scaling_factors
                        else maxent[state]
                    )
                    for state in maxent.keys()
                }
                maxent = {
                    state: maxent[state] / sum(maxent.values())
                    for state in maxent.keys()
                }

            if sum([abs(prev[key] - maxent[key]) for key in maxent.keys()]) < tol:
                break

    return maxent

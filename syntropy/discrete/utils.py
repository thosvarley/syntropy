# import pickle
import numpy as np
import networkx as nx
import itertools as it
from ..lattices import LATTICE_2, LATTICE_3, LATTICE_4

BOTTOM_2: tuple = ((0,), (1,))
BOTTOM_3: tuple = ((0,), (1,), (2,))
BOTTOM_4: tuple = ((0,), (1,), (2,), (3,))

PATHS_2: dict = nx.shortest_paths.shortest_path_length(
    LATTICE_2, source=None, target=BOTTOM_2
)
LAYERS_2: dict = {
    val: {key for key in PATHS_2.keys() if PATHS_2[key] == val}
    for val in set(PATHS_2.values())
}

PATHS_3: dict = nx.shortest_paths.shortest_path_length(
    LATTICE_3, source=None, target=BOTTOM_3
)
LAYERS_3: dict = {
    val: {key for key in PATHS_3.keys() if PATHS_3[key] == val}
    for val in set(PATHS_3.values())
}

PATHS_4: dict = nx.shortest_paths.shortest_path_length(
    LATTICE_4, source=None, target=BOTTOM_4
)
LAYERS_4: dict = {
    val: {key for key in PATHS_4.keys() if PATHS_4[key] == val}
    for val in set(PATHS_4.values())
}

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


def local_precompute_sources(joint_distribution: dict[tuple, float]) -> dict:
    """
    A utility function that computes the local entropy of each subset of
    elements. This speeds up the computation using the hmin function
    considerably,

    Parameters
    ----------
    joint_distribution: dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    dict
        For each state, the joint entropy of every subset.

    """

    N: int = len(list(joint_distribution.keys())[0])

    sources: list = list(make_powerset(range(N)))
    sources.remove(())

    local_entropies: dict = {
        state: {source: {} for source in sources} for state in joint_distribution.keys()
    }

    state: tuple
    for state in local_entropies.keys():

        source: tuple
        for source in local_entropies[state]:

            # For a given state x, and a given source s, computes the total probability mass of the joint
            # distribution consistent with the state of the source.
            # E.g. if state = (0,0,0) and source = (0,1), probability_mass = P(X0=0 AND X1=0)
            probability_mass = sum(
                [
                    joint_distribution[key]
                    for key in joint_distribution.keys()
                    if reduce_state(key, source) == reduce_state(state, source)
                ]
            )

            if probability_mass > 0:
                local_entropies[state][source] = -np.log2(probability_mass)
            else:
                local_entropies[state][
                    source
                ] = 0  # Set log2(0) to 0 since impossible events contain no information.

    return local_entropies


def hmin_discrete_redundancy(
    atom: tuple, state: tuple, sources: dict, joint_distribution: dict[tuple, float]
) -> float:
    """
    For a collection of sources :math:`\\alpha=\\{a_1, a_2, \\ldots, a_k\\}`, computes
    the redundnat entropy shared by all sources as:


    :math:`h_{\\cap}^{min}(\\alpha) = \\min_{i}h(a_i)`

    See:
        Finn, C., & Lizier, J. T. (2020).
        Generalised Measures of Multivariate Information Content.
        Entropy, 22(2), Article 2.
        https://doi.org/10.3390/e22020216

    Parameters
    ----------
    atom : tuple
        The partial entropy atom.
    state : tuple
        The state of the system.
    sources : dict
        The pre-computed local entropies constructed by the
        precompute_local_entropies() function.
    joint_distribution: dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    float
        The redundant entropy shared by every source in the atom.

    """

    if joint_distribution[state] == 0:
        return 0
    else:
        return min(sources[state][source] for source in atom)


def imin_discrete_redundancy(
    atom: tuple,
    state: tuple,
    inputs: tuple,
    target: tuple,
    sources: dict,
    joint_distribution: dict[tuple, float],
) -> float:
    """
    For a collection of sources :math:`\\alpha = \\{a_{1}, a_{2}, \\ldots, a_{k}\\}` and a target :math:`t` the redundancy is defined as:

    :math:`i_{min}(\\alpha;t) = \\min_{i}h(a_i) - \\min_{i}h(a_i|t)`

    See:
        Finn, C., & Lizier, J. T. (2020).
        Generalised Measures of Multivariate Information Content.
        Entropy, 22(2), Article 2.
        https://doi.org/10.3390/e22020216


    Parameters
    ----------
    atom : tuple
        The partial information atom.
    state : tuple
        The joint-state of the systems.
    inputs : tuple
        The indices of the input variables.
    target : tuple
        The indices of the target variable. May be multivariate.
    sources : dict
        The pre-computed local entropies
        for the joint state of the source and the target.
    joint_distribution: dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    float
        The local redundant mutual information.

    """

    i_plus: float = np.inf
    i_minus: float = np.inf

    h_target: float = sources[state][target]

    for source in atom:

        input_source: tuple = tuple(inputs[x] for x in source)
        h_input_source: float = sources[state][input_source]

        if i_plus > h_input_source:
            i_plus = h_input_source

        joint_source: tuple = tuple(sorted(input_source + target))

        h_conditional: float = sources[state][joint_source] - h_target

        if i_minus > h_conditional:
            i_minus = h_conditional

    return i_plus - i_minus


def hsx_discrete_redundancy(
    atom: tuple, state: tuple, joint_distribution: dict[tuple, float]
) -> float:
    """
    Computes the redundant entropy shared by a set of sources using the :math:`h_{sx}` function.

    For a collection of sources :math:`\\alpha = \\{a_{1}, a_{2}, \\ldots, a_{k}\\}`,
    the redundancy is defined as

    :math:`h^{sx}_{\cap}(\\alpha) = -\\log_{2} P(a_{1} \\cup a_{2} \\cup \\ldots \\cup a_{k})`.

    See:
        Varley, T. F., Pope, M., Maria Grazia, P., Joshua, F., & Sporns, O. (2023).
        Partial entropy decomposition reveals higher-order information structures in human brain activity.
        Proceedings of the National Academy of Sciences, 120(30), e2300888120.
        https://doi.org/10.1073/pnas.2300888120

    Parameters
    ----------
    atom : tuple
        The partial entropy atom.
    state : tuple
        The state of the system.
    joint_distribution: dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    float
        The redundant entropy shared by every source in the atom..

    """
    if joint_distribution[state] == 0:
        return 0.0
    else:
        state_set: set = set()

        source: tuple
        for source in atom:

            state_source: tuple = reduce_state(state, source)
            state_set.update(
                {
                    key
                    for key in joint_distribution.keys()
                    if reduce_state(key, source) == state_source
                }
            )

        redundant_entropy: float = -np.log2(
            sum(joint_distribution[s] for s in state_set)
        )

        return redundant_entropy


def isx_discrete_redundancy(
    atom: tuple,
    state: tuple,
    inputs: tuple,
    target: tuple,
    joint_distribution: dict[tuple, float],
) -> float:
    """
    For a collection of sources :math:`\\alpha = \\{a_{1}, a_{2}, \\ldots, a_{k}\\}` and a target :math:`t` the redundancy is defined as:

    :math:`i_{sx}(\\alpha;t) = \\log_{2}\\frac{P(t) - P(t \\cap (\\bar{a}_1 \\cap \\ldots \\cap \\bar{a}_k)}{1 - P(\\bar{a}_1 \\cap \\ldots \\cap \\bar{a}_{k})} - \\log_2 P(t)`

    Parameters
    ----------
    atom : tuple
        The partial information atom.
    state : tuple
        The joint-state of the systems.
    inputs : tuple
        The indices of the input variables.
    target : tuple
        The indices of the target variable. May be multivariate.
    sources : dict
        The pre-computed local entropies
        for the joint state of the source and the target.
    joint_distribution: dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    float
        The redundant mutual informations.

    """

    # i_sx^+
    # -log2(1 / a1 U a2 U ... U ak)

    state_set: set = set()
    source: tuple
    for source in atom:

        input_source = tuple(inputs[x] for x in source)
        state_source: tuple = reduce_state(state, input_source)
        state_set.update(
            {
                key
                for key in joint_distribution.keys()
                if reduce_state(key, input_source) == state_source
            }
        )

    i_plus = np.log2(1 / sum(joint_distribution[x] for x in state_set))

    # i_sx^-
    # -log2( P(t) / t ^ (a1 U a2 U ... U ak))
    target_state: tuple = reduce_state(state, target)
    p_target: float = sum(
        [
            joint_distribution[key]
            for key in joint_distribution.keys()
            if reduce_state(key, target) == target_state
        ]
    )

    state_set: set = set()
    for source in atom:

        input_source = tuple(inputs[x] for x in source)
        state_source: tuple = reduce_state(state, input_source)
        state_set.update(
            {
                key
                for key in joint_distribution.keys()
                if reduce_state(key, input_source) == state_source
            }
        )

    state_set = {key for key in state_set if reduce_state(key, target) == target_state}

    i_minus = np.log2(p_target / sum([joint_distribution[x] for x in state_set]))

    return i_plus - i_minus


def mobius_inversion(
    decomposition: str,
    joint_distribution: dict[tuple, float],
    redundancy: str,
    inputs: tuple = (None,),
    target: tuple = (None,),
) -> tuple[dict, dict]:
    """
    Computes the Mobius inversion on the antichain lattice.

    Parameters
    ----------
    decomposition : str
        Which decomposition to use:
            For partial information decomposition use "pid".
            For partial entropy decomposition use "ped".
    joint_distribution: dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.
    redundancy : str
        The redundancy function.
    inputs : tuple, optional
        The indicies of the inputs. The default is (None,)
    target : tuple, optional
        The (potentially multivariate) indices of the target.
        The default is (None,).

    Returns
    -------
    (dict, dict)
        The pointwise and average partial information atom dictionaries.

    """


    decomposition_lower = decomposition.lower()
    assert decomposition_lower in {
        "pid",
        "ped",
    }, "You must specify a decomposition: PID or PED."

    if decomposition_lower == "pid":
        assert redundancy in {
            "isx",
            "imin",
        }, "The implemented redundancy functions are 'imin' and 'hsx'."
        assert target != (None,), "You must specify a target."
        if redundancy == "isx":
            redundancy_function = isx_discrete_redundancy
        elif redundancy == "imin":
            redundancy_function = imin_discrete_redundancy
        total_str: str = "total_information"
        partial_str: str = "pi"
    elif decomposition_lower == "ped":
        assert redundancy in {
            "hsx",
            "hmin",
        }, "The implemented redundancy functions are 'hxs' and 'hmin'."
        if redundancy == "hsx":
            redundancy_function = hsx_discrete_redundancy
        elif redundancy == "hmin":
            redundancy_function = hmin_discrete_redundancy
        total_str: str = "total_entropy"
        partial_str: str = "pe"

    if inputs == (None,):  # If no inputs are specified, all elements are inputs
        N = {len(x) for x in joint_distribution.keys()}.pop()
    else:
        N = len(inputs)

    if N == 2:
        lattice, bottom, layers = LATTICE_2, BOTTOM_2, LAYERS_2
    elif N == 3:  #    |
        lattice, bottom, layers = LATTICE_3, BOTTOM_3, LAYERS_3
    else:  #                      |
        lattice, bottom, layers = LATTICE_4, BOTTOM_4, LAYERS_4

    # Pre-computing all the local h_{\partial}(source) values
    # this saves a lot of time.

    if "min" in redundancy:
        sources: dict = local_precompute_sources(joint_distribution)

    ptw: dict = {state: dict() for state in joint_distribution.keys()}

    for state in joint_distribution.keys():
        for layer in layers:
            for atom in layers[layer]:
                if decomposition_lower == "pid":

                    if redundancy == "imin":
                        args = {
                            "atom": atom,
                            "state": state,
                            "inputs": inputs,
                            "target": target,
                            "sources": sources,
                            "joint_distribution": joint_distribution,
                        }
                    elif redundancy == "isx":
                        args = {
                            "atom": atom,
                            "state": state,
                            "inputs": inputs,
                            "target": target,
                            "joint_distribution": joint_distribution,
                        }

                    lattice.nodes[atom][total_str] = redundancy_function(**args)
                elif decomposition_lower == "ped":

                    if redundancy == "hmin":
                        args = {
                            "atom": atom,
                            "state": state,
                            "sources": sources,
                            "joint_distribution": joint_distribution,
                        }
                    elif redundancy == "hsx":
                        args = {
                            "atom": atom,
                            "state": state,
                            "joint_distribution": joint_distribution,
                        }

                    lattice.nodes[atom][total_str] = redundancy_function(**args)

                if atom == bottom:
                    lattice.nodes[atom][partial_str] = lattice.nodes[atom][total_str]
                else:
                    lattice.nodes[atom][partial_str] = lattice.nodes[atom][
                        total_str
                    ] - sum(
                        [
                            lattice.nodes[d][partial_str]
                            for d in lattice.nodes[atom]["descendants"]
                        ]
                    )

        local_ptw: dict = {
            node: lattice.nodes[node][partial_str] for node in lattice.nodes
        }
        ptw[state] = local_ptw

    avg = {}
    for node in lattice.nodes:
        avg[node] = sum(
            [joint_distribution[state] * ptw[state][node] for state in ptw.keys()]
        )

    return ptw, avg

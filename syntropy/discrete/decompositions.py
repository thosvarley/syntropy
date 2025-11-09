import numpy as np
import networkx as nx
from typing import Callable
from .utils import make_powerset, reduce_state

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
def partial_information_decomposition(
    redundancy: str,
    inputs: tuple,
    target: tuple,
    joint_distribution: dict[tuple[int, ...], float],
) -> (dict, dict):
    """
    Computes the partial information decomposition for up to four input variables
    onto one (potentially joint) target variable.

    The available redundancy functions are :math:`i_{\\min}` from Finn and Lizier and :math:`i_{sx}` from Makkeh et al..

    .. math:: 

        i_{\\min}(\\alpha;t) &= \\min h(\\alpha_i) - \min h(\\alpha_i|t) \\\\
           i_{sx}(\\alpha;t) &= \\log\\frac{P(t)-P(t\\cap(\\alpha_1\\cup ...\\cup\\alpha_k))}{1-P(\\bar\\alpha_1\\cap ...\\cap\\alpha_N)}
    
    Parameters
    ----------
    redundancy : str
        The redundancy function.
        "imin" for the Finn and Lizier measure
        "isx" for the Makkeh et al., measure
    inputs : tuple
        The set of up to four input elements.
    target : tuple
        The set of target elements.
        If len(target) > 1, then the target is the joint
        state of all elements
    joint_distribution : dict[tuple,float]
        The joint distribution object.

    Returns
    -------
    ptw : dict
        A dictionary of dictionaries,
        The outer dictionary has one key for each joint state.
        Each inner dictionary is the lookup of partial information atoms.
    avg : dict
        The expected value for each partial information atom.

    References
    ----------
    Williams, P. L., & Beer, R. D. (2010). 
    Nonnegative Decomposition of Multivariate Information. 
    arXiv:1004.2515 [Math-Ph, Physics:Physics, q-Bio]. 
    http://arxiv.org/abs/1004.2515

    Finn, C., & Lizier, J. T. (2018).
    Pointwise Partial Information Decomposition Using
    the Specificity and Ambiguity Lattices.
    Entropy, 20(4), Article 4.
    https://doi.org/10.3390/e20040297

    Makkeh, A., Gutknecht, A. J., & Wibral, M. (2021).
    Introducing a differentiable measure of pointwise
    shared information.
    Physical Review E, 103(3), 032149.
    https://doi.org/10.1103/PhysRevE.103.032149
    
    """

    ptw, avg = mobius_inversion(
        decomposition="pid",
        joint_distribution=joint_distribution,
        inputs=inputs,
        redundancy=redundancy,
        target=target,
    )

    return ptw, avg


def partial_entropy_decomposition(
    redundancy: str, joint_distribution: dict[tuple[int, ...], float]
) -> (dict, dict):
    """
    Computes the partial entropy decomposition of a joint distribution
    with up to four elements.

    The available redundancy functions are h_min from Finn and Lizier
    and h_sx from Varley et al.,
    
    .. math:: 
        h_{\\min}(\\alpha) &= \\min(\\alpha_i) \\\\
           h_{sx}(\\alpha) &= \\log\\frac{1}{P(\\alpha_1\\cup ... \\cup\\alpha_N)}

    Parameters
    ----------
    redundancy : str
        The redundnacy function to use.
        "hmin" for the measure from Finn and Lizier,
        "hsx" for the measure from Varley et al.,
    joint_distribution : dict[tuple,float]
        The joint distribution object.

    Returns
    -------
    ptw : dict[tuple, dict]
        A dictionary of dictionaries,
        The outer dictionary has one key for each joint state.
        Each inner dictionary is the lookup of partial entropy atoms.
    avg : dict[tuple, float]
        The expected value for each partial entropy atom.

    References
    ----------
    Finn, C., & Lizier, J. T. (2020).
    Generalised Measures of Multivariate Information Content.
    Entropy, 22(2), Article 2.
    https://doi.org/10.3390/e22020216

    Varley, T. F., Pope, M., Maria Grazia, P., Joshua, F., & Sporns, O. (2023).
    Partial entropy decomposition reveals higher-order
    information structures in human brain activity.
    Proceedings of the National Academy of Sciences,
    120(30), e2300888120.
    https://doi.org/10.1073/pnas.2300888120

    """

    ptw, avg = mobius_inversion(
        decomposition="ped",
        joint_distribution=joint_distribution,
        redundancy=redundancy,
    )

    return ptw, avg


def generalized_information_decomposition(
    redundancy: str,
    posterior_distribution: dict[tuple[int, ...], float],
    prior_distribution: dict[tuple[int, ...], float],
) -> (dict, dict):
    """
    Computes the generalized information decomposition from Varley et al.
    The GID is a decomposition of the Kullback-Leibler divergence of a
    posterior distribution from a prior distribution.

    Available redundnacy functions are "hmin" and "hsx". See
    the documentation for the partial_entropy_decomposition() function
    for details.

    Parameters
    ----------
    redundancy : str
        The localizable redundancy function.
        Options are: hmin and hsx.
    posterior_distribution : dict[tuple, float]
        The posterior distribution.
        The support set of this distribution must be a subset of
        the supppirt set of the prior distribution.
    prior_distribution : dict[tuple, float]
        The prior distribution.

    Returns
    -------
    ptw : dict[tuple, dict]
        A dict of dicts.
        The local Kullback-Leibler divergence for each atom for each state.
    avg : dict[tuple, float]
        The average Kullback-Leibler divergence for each atom.

    References
    ----------
    Varley, T. F. (2024).
    Generalized decomposition of multivariate information.
    PLOS ONE, 19(2), e0297128.
    https://doi.org/10.1371/journal.pone.0297128

    """
    assert set(prior_distribution.keys()).issuperset(
        set(posterior_distribution.keys())
    ), (
        "The support set of the prior must be a superset of the support set of the posterior."
    )
    assert redundancy.lower() in {
        "hmin",
        "hsx",
    }, "The supported redundancy functions are hmin and hsx."

    ptw_prior, _ = mobius_inversion(
        decomposition="ped",
        joint_distribution=prior_distribution,
        redundancy=redundancy,
    )

    ptw_posterior, _ = mobius_inversion(
        decomposition="ped",
        joint_distribution=posterior_distribution,
        redundancy=redundancy,
    )

    nodes = list(ptw_prior[list(prior_distribution.keys())[0]].keys())

    ptw = {
        state: {
            node: ptw_prior[state][node] - ptw_posterior[state][node] for node in nodes
        }
        for state in posterior_distribution.keys()
    }

    avg = {}
    for node in nodes:
        avg[node] = sum(
            [posterior_distribution[state] * ptw[state][node] for state in ptw.keys()]
        )

    return ptw, avg


def representational_complexity(avg: dict, comparator: Callable = min) -> float:
    """
    Computes the representational complexity of a given partial information or entropy lattice.
    The representational complexity is a measure of how
    much partial information atoms of a given degree of synergy
    contribute to the overall mutual information or entropy.


    Parameters
    ----------
    avg : dict[tuple, float]
        The dictionary of partial information/entropy atoms.
        Returned from any of the above functions.
    comparator : function, optional
        Whether to consider the minimum complexity of an atom.
        or the maximum complexity of an atom.
        Options are: min, max, np.min, np.max.
        The default is min, following the original work
        by Ehrlich et al.,.

    Returns
    -------
    float
        The representational complexity.

    References
    ----------
    Ehrlich, D. A., Schneider, A. C., Priesemann, V., Wibral, M., & Makkeh, A. (2023).
    A Measure of the Complexity of Neural Representations
    based on Partial Information Decomposition.
    Transactions on Machine Learning Research.
    https://openreview.net/forum?id=R8TU3pfzFr

    """

    assert comparator in (min, max, np.min, np.max), "The comparator must be min or max"

    rc: float = 0.0

    for atom in avg.keys():
        rc += avg[atom] * comparator(len(source) for source in atom)

    return rc / sum(avg.values())

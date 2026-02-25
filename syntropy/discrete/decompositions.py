import numpy as np
import math
import networkx as nx
from typing import Callable, Any
from .utils import make_powerset, reduce_state
from .shannon import mutual_information
from ..lattices import load_lattice, mobius_inversion

Atom = tuple[tuple[int, ...], ...]
DiscreteDist = dict[tuple[Any, ...], float]
Sources = dict[tuple[Any, ...], dict[tuple[int, ...], float]]


def local_precompute_sources(joint_distribution: DiscreteDist) -> Sources:
    """
    A utility function that computes the local entropy of each subset of
    elements. This speeds up the computation using the hmin function
    considerably,

    Parameters
    ----------
    joint_distribution: DiscreteDist
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

    state: tuple[Any, ...]
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
                local_entropies[state][source] = -math.log2(probability_mass)
            else:
                local_entropies[state][source] = (
                    0  # Set log2(0) to 0 since impossible events contain no information.
                )

    return local_entropies


def hmin_discrete_redundancy(
    atom: tuple, state: tuple[Any, ...], sources: dict, joint_distribution: DiscreteDist
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
    atom : Atom
        The partial entropy atom.
    state : tuple[Any, ...]
        The state of the system.
    sources : Sources
        The pre-computed local entropies constructed by the
        precompute_local_entropies() function.
    joint_distribution: DiscreteDist
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
        return min(sources[state][a] for a in atom)


def hsx_discrete_redundancy(
    atom: Atom, state: tuple[Any, ...], joint_distribution: DiscreteDist
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
    atom : Atom
        The partial entropy atom.
    state : tuple[Any, ...]
        The state of the system.
    joint_distribution: DiscreteDist
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

        redundant_entropy: float = -math.log2(
            sum(joint_distribution[s] for s in state_set)
        )

        return redundant_entropy


def mmi_discrete_redundancy(
    atom: Atom,
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    joint_distribution: DiscreteDist,
    single_target_flag: bool = True,
) -> float:
    """

    Parameters
    ----------
    atom : Atom
        The partial information or integrated information atom.

    inputs : tuple[int, ...]
        The indices of the input elements.

    target : tuple[int, ...]
        The indices of the target element(s)

    joint_distribution: DiscreteDist
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    single_target_flag : bool
        Whether the do a single-target PID or a multi-target Phi-ID.


    Returns
    -------
    float


    """
    mn: float = np.inf
    if single_target_flag is True:  # PID
        atom_inputs = tuple(tuple(inputs[x] for x in a) for a in atom)
        for idxs_x in atom_inputs:
            _, mi = mutual_information(
                idxs_x=idxs_x, idxs_y=target, joint_distribution=joint_distribution
            )
            if mi < mn:
                mn = mi
    elif single_target_flag is False:  # Phi-ID
        atom_inputs = tuple(tuple(inputs[x] for x in a) for a in atom[0])
        atom_target = tuple(tuple(target[x] for x in a) for a in atom[1])
        for idxs_x in atom_inputs:
            for idxs_y in atom_target:
                _, mi = mutual_information(
                    idxs_x=idxs_x, idxs_y=idxs_y, joint_distribution=joint_distribution
                )
                if mi < mn:
                    mn = mi
    return mn


def ipm_discrete_redundancy(
    atom: Atom,
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    state: tuple[Any, ...],
    sources: Sources,
    joint_distribution: DiscreteDist,
    single_target_flag: bool = True,
) -> float:
    """
    For a collection of sources :math:`\\alpha = \\{a_{1}, a_{2}, \\ldots, a_{k}\\}` and a target :math:`t` the redundancy is defined as:

    :math:`i_{min}(\\alpha;t) = \\min_{i}h(a_i) - \\min_{i}h(a_i|t)`

    For a pair of atoms
    Parameters
    ----------
    atom : Atom
        The partial information or integrated information atom.

    inputs : tuple[int, ...]
        The indices of the input elements.

    target : tuple[int, ...]
        The indices of the target element(s)

    state : tuple[Any, ...]

    sources : Sources

    joint_distribution: DiscreteDist
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    single_target_flag : bool
        Whether the do a single-target PID or a multi-target Phi-ID.


    Returns
    -------
    float


    References
    ----------
    Finn, C., & Lizier, J. T. (2020).
    Generalised Measures of Multivariate Information Content.
    Entropy, 22(2), Article 2.
    https://doi.org/10.3390/e22020216


    """
    if single_target_flag is True:
        atom_inputs: Atom = tuple(tuple(inputs[x] for x in a) for a in atom)
        mn_inputs: float = hmin_discrete_redundancy(
            atom=atom_inputs,
            state=state,
            sources=sources,
            joint_distribution=joint_distribution,
        )
        h_target: float = sources[state][target]
        mn_joint: float = hmin_discrete_redundancy(
            atom=tuple(tuple(set(a + target)) for a in atom_inputs),
            state=state,
            sources=sources,
            joint_distribution=joint_distribution,
        )
        lmi: float = mn_inputs + h_target - mn_joint
    elif single_target_flag is False:
        atom_inputs: Atom = tuple(tuple(inputs[x] for x in a) for a in atom[0])
        atom_target: Atom = tuple(tuple(target[x] for x in a) for a in atom[1])

        mn_inputs: float = hmin_discrete_redundancy(
            atom=atom_inputs,
            state=state,
            sources=sources,
            joint_distribution=joint_distribution,
        )
        mn_target: float = hmin_discrete_redundancy(
            atom=atom_target,
            state=state,
            sources=sources,
            joint_distribution=joint_distribution,
        )
        mn_joint: float = np.inf
        for s1 in atom_inputs:
            for s2 in atom_target:
                mn_joint = min(mn_joint, sources[state][tuple(set(s1 + s2))])
        lmi: float = mn_inputs + mn_target - mn_joint

    return lmi


def isx_discrete_redundancy(
    atom: Atom,
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    state: tuple[Any, ...],
    sources: Sources,
    joint_distribution: DiscreteDist,
    single_target_flag: bool = True,
) -> float:
    """
    Computes the redundant entropy shared by a set of sources using the :math:`h_{sx}` function.

    For a collection of sources :math:`\\alpha = \\{a_{1}, a_{2}, \\ldots, a_{k}\\}`,
    the redundancy is defined as

    :math:`h^{sx}_{\cap}(\\alpha) = -\\log_{2} P(a_{1} \\cup a_{2} \\cup \\ldots \\cup a_{k})`.

    Parameters
    ----------
    atom : Atom
        The partial information or integrated information atom.

    inputs : tuple[int, ...]
        The indices of the input elements.

    target : tuple[int, ...]
        The indices of the target element(s)

    state : tuple[Any, ...]
        The state of the system.

    sources : Sources
        A dictionary of dictionaries.
        The first level is all system states.
        The second level is the local entropies of all subsets of
        the system in that state.

    joint_distribution: DiscreteDist
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    single_target_flag : bool
        Whether the do a single-target PID or a multi-target Phi-ID.


    Returns
    -------
    float


    References
    ----------
    Varley, T. F., Pope, M., Maria Grazia, P., Joshua, F., & Sporns, O. (2023).
    Partial entropy decomposition reveals higher-order information structures in human brain activity.
    Proceedings of the National Academy of Sciences, 120(30), e2300888120.
    https://doi.org/10.1073/pnas.2300888120

    """
    if single_target_flag is True:  # PID
        atom_inputs = tuple(tuple(inputs[x] for x in a) for a in atom)
        sx_inputs = hsx_discrete_redundancy(
            atom=atom_inputs, state=state, joint_distribution=joint_distribution
        )
        h_target = sources[state][target]
        sx_joint = hsx_discrete_redundancy(
            atom=tuple(tuple(set(a + target)) for a in atom_inputs),
            state=state,
            joint_distribution=joint_distribution,
        )
        lmi: float = sx_inputs + h_target - sx_joint
    elif single_target_flag is False:  # PED
        atom_inputs: Atom = tuple(tuple(inputs[x] for x in a) for a in atom[0])
        atom_target: Atom = tuple(tuple(target[x] for x in a) for a in atom[1])

        sx_inputs: float = hsx_discrete_redundancy(
            atom=atom_inputs, state=state, joint_distribution=joint_distribution
        )
        sx_target: float = hsx_discrete_redundancy(
            atom=atom_target, state=state, joint_distribution=joint_distribution
        )
        atom_joint = []
        for s1 in atom_inputs:
            for s2 in atom_target:
                atom_joint.append(tuple(set(s1 + s2)))
        atom_joint: Atom = tuple(atom_joint)
        sx_joint: float = hsx_discrete_redundancy(
            atom=atom_joint, state=state, joint_distribution=joint_distribution
        )

        lmi: float = sx_inputs + sx_target - sx_joint

    return lmi


def _pid(
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    joint_distribution: DiscreteDist,
    redundancy_function: str,
    single_target_flag: bool = True,
) -> Any:
    """
    A utility function that computes the actual PID/PhiID depending on the state of single_target_flag.

    Parameters
    ----------
    inputs : tuple[int, ...]
        The indices of the input elements.

    target : tuple[int, ...]
        The indices of the target element(s)

    joint_distribution: DiscreteDist
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    redundancy_function : str
        The localizable redundancy function.
        Options are: hmin and hsx.

    single_target_flag : bool
        Whether the do a single-target PID or a multi-target Phi-ID.


    Returns
    -------
    Any


    """
    assert redundancy_function in {"mmi", "ipm", "isx"}, (
        "The available redundancy functions are Finn and Lizier's i_pm, Makkeh et al.'s i_sx, and the minimum mutual information."
    )

    num_inputs = len(inputs)
    assert num_inputs in (2, 3, 4), (
        "Currently, syntropy only supports PIDs on 2, 3, and 4 inputs."
    )
    num_target = len(target)
    if single_target_flag is False:
        assert num_target in (2, 3), (
            "Currently syntropy only supports \Phi-IDs on 2 and 3 targets."
        )
        lattice: nx.DiGraph = load_lattice(num_inputs=num_inputs, num_target=num_target)
    elif single_target_flag is True:
        lattice: nx.DiGraph = load_lattice(num_inputs=num_inputs)

    kwargs: dict[str, Any] = {
        "inputs": inputs,
        "target": target,
        "joint_distribution": joint_distribution,
        "single_target_flag": single_target_flag,
    }

    if redundancy_function == "mmi":
        redundancy_func: Callable = mmi_discrete_redundancy

        result = mobius_inversion(
            redundancy_func=redundancy_func, lattice=lattice.copy(), kwargs=kwargs
        )
        avg: dict[Atom, float] = {
            node: result.nodes[node]["pi"] for node in result.nodes
        }
        return avg

    elif redundancy_function in {"isx", "ipm"}:
        sources: Sources = local_precompute_sources(joint_distribution)
        kwargs["sources"] = sources
        if redundancy_function == "isx":
            redundancy_func: Callable = isx_discrete_redundancy
        elif redundancy_function == "ipm":
            redundancy_func: Callable = ipm_discrete_redundancy

        ptw: dict = {}
        state_mapping: dict = {}
        for state in joint_distribution.keys():
            kwargs["state"] = state
            result = mobius_inversion(redundancy_func, lattice.copy(), kwargs)

            state_inputs = tuple(state[i] for i in inputs)
            state_targets = tuple(state[i] for i in target)

            ptw[(state_inputs, state_targets)] = {
                node: result.nodes[node]["pi"] for node in result.nodes
            }

            state_mapping[(state_inputs, state_targets)] = state

        atoms: list[Atom] = list(lattice.nodes)
        avg: dict = {}
        for a in atoms:
            avg[a] = np.sum(
                [
                    joint_distribution[state_mapping[state]] * ptw[state][a]
                    for state in ptw.keys()
                ]
            )

        return ptw, avg


def partial_information_decomposition(
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    joint_distribution: DiscreteDist,
    redundancy_function: str,
) -> Any:
    """
    Computes the partial information decomposition for up to four input variables onto one (potentially joint) target variable.

    The available redundancy functions are :math:`MMI`, :math:`i_{pm}` from Finn and Lizier and :math:`i_{sx}` from Makkeh et al..

    .. math:: 

        i_{pm}(\\alpha;t) &= \\min h(\\alpha_i) - \min h(\\alpha_i|t) \\\\
           i_{sx}(\\alpha;t) &= \\log\\frac{P(t)-P(t\\cap(\\alpha_1\\cup ...\\cup\\alpha_k))}{1-P(\\bar\\alpha_1\\cap ...\\cap\\alpha_N)} \\\\
           i_{MMI}(\\alpha;t) &= \\min_i I(\\alpha_i;T) 
    
    Parameters
    ----------
    inputs : tuple[int, ...]
        The indices of the input elements.

    target : tuple[int, ...]
        The indices of the target element(s)
        
    joint_distribution: DiscreteDist
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.
        
    redundancy_function : str
        The localizable redundancy function.
        Options are: hmin and hsx.
        

    Returns
    -------
    Any


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
    return _pid(
        inputs=inputs,
        target=target,
        joint_distribution=joint_distribution,
        redundancy_function=redundancy_function,
        single_target_flag=True,
    )


def integrated_information_decomposition(
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    joint_distribution: DiscreteDist,
    redundancy_function: str,
) -> Any:
    """
    Computes the integrated information decomposition introduced by Rosas, Mediano, et al.
    The PhiID relaxes the requirement of only having a single target, and instead allows for 
    redundant-redundant, synergistic-synergistic, etc interactions. 

    Available redundancy functions are: 
        i_{pm}(\\alpha;\\beta) &= \\min_i h(\\alpha_i) + \\min_i h(\\beta_i) - \min h(\\alpha_i, \\beta_i) \\\\
                i_{tsx}(\\alpha;\\beta) &= h_{sx}(\\alpha) + h_{sx}(\\beta) - h_{sx}(\\alpha\\cap\\beta)
                i_{MMI}(\\alpha;\\beta) &= \\min_{ij} I(\\alpha_i;\\beta_j) 


    Parameters
    ----------
    inputs : tuple[int, ...]
        The indices of the input elements.
    target : tuple[int, ...]
        The indices of the target element(s)
    joint_distribution: DiscreteDist
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.
    redundancy_function : str
        The localizable redundancy function.
        Options are: hmin and hsx.


    Returns
    -------
    Any

    References
    ----------
    Mediano, P. A. M., Rosas, F. E., Luppi, A. I., Carhart-Harris, R. L., Bor, D., Seth, A. K., & Barrett, A. B. (2025). Toward a unified taxonomy of information dynamics via Integrated Information Decomposition. Proceedings of the National Academy of Sciences, 122(39), e2423297122. https://doi.org/10.1073/pnas.2423297122

    Rosas, F. E., Mediano, P. A. M., Jensen, H. J., Seth, A. K., Barrett, A. B., Carhart-Harris, R. L., & Bor, D. (2020). Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data. PLOS Computational Biology, 16(12), Article 12. https://doi.org/10.1371/journal.pcbi.1008289

    Varley, T. F. (2023). Decomposing past and future: Integrated information decomposition based on shared probability mass exclusions. PLOS ONE, 18(3), e0282950. https://doi.org/10.1371/journal.pone.0282950

    """
    return _pid(
        inputs=inputs,
        target=target,
        joint_distribution=joint_distribution,
        redundancy_function=redundancy_function,
        single_target_flag=False,
    )


def partial_entropy_decomposition(
    joint_distribution: DiscreteDist,
    redundancy_function: str,
) -> tuple[dict[tuple[int, ...], dict[Atom, float]], dict[Atom, float]]:
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
    joint_distribution: DiscreteDist
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    redundancy_function : str
        The localizable redundancy function.
        Options are: hmin and hsx.

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
    N: int = len(next(iter(joint_distribution)))
    assert N in (2, 3, 4), (
        "Currently, syntropy only supports PEDs on 2, 3, and 4-dimensional distributions."
    )

    assert redundancy_function in {"hsx", "hmin"}, (
        "The available redundancy functions are Varley's h_sx and Finn and Lizier's h_min."
    )

    kwargs: dict[str, Any] = {"joint_distribution": joint_distribution}
    sources: Sources = local_precompute_sources(joint_distribution)

    if redundancy_function == "hsx":
        redundancy_func = hsx_discrete_redundancy
    elif redundancy_function == "hmin":
        redundancy_func = hmin_discrete_redundancy
        kwargs["sources"] = sources

    lattice: nx.DiGraph = load_lattice(num_inputs=N)

    ptw: dict[tuple[int, ...], dict[Atom, float]] = dict()
    for state in joint_distribution.keys():
        kwargs["state"] = state

        result = mobius_inversion(
            redundancy_func=redundancy_func, lattice=lattice, kwargs=kwargs
        )
        ptw[state] = {node: result.nodes[node]["pi"] for node in result.nodes}

    avg: dict[Atom, float] = dict()
    for a in lattice.nodes:
        avg[a] = np.sum([joint_distribution[key] * ptw[key][a] for key in ptw.keys()])

    return ptw, avg


def generalized_information_decomposition(
    posterior_distribution: DiscreteDist,
    prior_distribution: DiscreteDist,
    redundancy_function: str,
) -> (dict, dict):
    """
    Computes the generalized information decomposition from Varley et al.
    The GID is a decomposition of the Kullback-Leibler divergence of a
    posterior distribution from a prior distribution.

    Available redundancy functions are "hmin" and "hsx". See
    the documentation for the partial_entropy_decomposition() function
    for details.

    Parameters
    ----------
    posterior_distribution : DiscreteDist
        The posterior distribution.
        The support set of this distribution must be a subset of
        the supppirt set of the prior distribution.

    prior_distribution : DicreteDist
        The prior distribution.

    redundancy_function : str
        The localizable redundancy function.
        Options are: hmin and hsx.

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
    assert redundancy_function.lower() in {
        "hmin",
        "hsx",
    }, "The supported redundancy functions are hmin and hsx."

    ptw_prior, _ = partial_entropy_decomposition(
        joint_distribution=prior_distribution,
        redundancy_function=redundancy_function,
    )

    ptw_posterior, _ = partial_entropy_decomposition(
        joint_distribution=posterior_distribution,
        redundancy_function=redundancy_function,
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

import numpy as np
import networkx as nx

from .shannon import (
    local_differential_entropy,
    mutual_information,
)

from .utils import check_cov
from ..utils import make_powerset
from ..lattices import load_lattice, mobius_inversion
from numpy.typing import NDArray
from typing import Callable, Any

Atom = tuple[tuple[int, ...], ...]


# %%
def unpack_atom(atom: Atom) -> set[int, ...]:
    """
    A utitlity function to unpack tuples.
    """
    varset: set[int, ...] = set()
    for s in {*atom}:
        varset.update(set(s))

    return varset


def local_precompute_sources(
    data: NDArray[np.floating], cov: NDArray[np.floating] | None = None
):
    """
    A utility function that pre-computes the local entropies
    of every subset of the data. This speeds up the PID/GID/PED
    by orders of magnitude.

    Parameters
    ----------
    data : NDArray[np.floating]
        The data in channels x samples format.
    cov : NDArray[np.floating], optional
        The covariance matrix of the data.
        The default is COV_NULL.

    Returns
    -------
    dict
        A dictionary of the local entropies for every source.

    """
    cov_: NDArray[np.floating] = check_cov(cov, data)

    N: int = data.shape[0]

    joint: tuple[int, ...] = tuple(i for i in range(N))
    sources: list[tuple[int, ...]] = list(make_powerset(joint))
    sources.remove(())

    return {
        source: local_differential_entropy(data[source, :], cov_[source, :][:, source])
        for source in sources
    }


def hmin_differential_redundancy(
    atom: Atom,
    sources: dict,
) -> NDArray[np.floating]:
    """
    For a collection of sources :math:`\\alpha=\\{a_1, a_2, \\ldots, a_k\\}`, computes
    the redundnat entropy shared by all sources as:

    :math:`h_{\\cap}^{min}(\\alpha) = \\min_{i}(a_{i})`

    Parameters
    ----------
    atom : tuple
        The partial information atom. In the form :math:`((a_1,),(a_2,)\\ldots)`.
    sources : dict
        The pre-computed collection of sources returned by 
        `precompute_local_entropies`.

    Returns
    -------
    NDArray[np.floating]
        The local redundancies for each sample.

    References
    ----------
    Finn, C., & Lizier, J. T. (2020).
    Generalised Measures of Multivariate Information Content.
    Entropy, 22(2), Article 2.
    https://doi.org/10.3390/e22020216


    """
    N: int = sources[atom[0]].shape[1]
    h_min: NDArray[np.floating] = np.repeat(np.inf, N)
    source: tuple = tuple()

    for source in atom:
        # i+ = min(h(source))
        h_min = np.minimum(h_min, sources[source])

    return h_min


def mmi_differential_redundancy(
    atom: Atom,
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    cov: NDArray[np.floating],
    single_target_flag: bool = True,
) -> float:
    mn: float = np.inf
    if single_target_flag is True:
        for idxs_x in atom:
            mi = mutual_information(idxs_x=idxs_x, idxs_y=target, cov=cov)
            if mi < mn:
                mn = mi
    elif single_target_flag is False:
        atom_inputs = atom[0]
        atom_targets = list()

        for x in atom[1]:
            atom_targets.append(tuple(target[i] for i in x))
        atom_targets = tuple(atom_targets)

        for idxs_x in atom_inputs:
            for idxs_y in atom_targets:
                mi = mutual_information(idxs_x=idxs_x, idxs_y=idxs_y, cov=cov)
                if mi < mn:
                    mn = mi

    return mn


def ipm_differential_redundancy(
    atom: Atom,
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    sources: dict[tuple[int, ...], NDArray[np.floating]],
    single_target_flag: bool = True,
) -> NDArray[np.floating]:
    """

    Parameters
    ----------
    atom : Atom
        The partial information or integrated information atom.

    inputs : tuple[int, ...]
        The indices of the input elements.

    target : tuple[int, ...]
        The indices of the target element(s)

    sources : dict[tuple[int, ...], NDArray[np.floating]]
        The local entropies retruned by `precompute_local_entropies`

    single_target_flag : bool
        Whether to do a single-target PID or multi-target PhiID.

    Returns
    -------
    NDArray[np.floating]


    """
    joint: tuple[int, ...] = inputs + target
    num_inputs: int = len(inputs)
    num_target: int = len(target)

    target_: tuple[int, ...] = tuple(
        i for i in range(num_inputs, num_inputs + num_target)
    )

    if single_target_flag is True:
        mn_inputs = hmin_differential_redundancy(atom=atom, sources=sources)
        mn_target = sources[target_]
        mn_joint = hmin_differential_redundancy(
            tuple(x + target_ for x in atom), sources=sources
        )
    elif single_target_flag is False:
        atom_inputs = atom[0]
        atom_targets = list()
        for x in atom[1]:
            atom_targets.append(tuple(target_[i] for i in x))
        atom_targets = tuple(atom_targets)

        mn_inputs = hmin_differential_redundancy(atom=atom_inputs, sources=sources)
        mn_target = hmin_differential_redundancy(atom=atom_targets, sources=sources)
        mn_joint = np.repeat(np.inf, repeats=mn_inputs.shape[1])
        for s1 in atom_inputs:
            for s2 in atom_targets:
                joint = s1 + s2
                mn_joint = np.minimum(mn_joint, sources[joint])

    diff = mn_inputs + mn_target - mn_joint

    return diff


def _pid(
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    data: NDArray[np.floating] | None,
    cov: NDArray[np.floating] | None = None,
    redundancy_function: str = "ipm",
    single_target_flag: bool = True,
) -> dict[Atom, float] | tuple[dict[Atom, float], dict[Atom, NDArray[np.floating]]]:
    """
    A utility function that computes the guts of the PID/PhiID, depending 
    the value of single_target_flag. 

    Parameters
    ----------
    inputs : tuple[int, ...]
        The indices of the input elements. 
        
    target : tuple[int, ...]
        The indices of the target elements. 
        
    data : NDArray[np.floating]
        The numpy array, assumed to be in channels x time format.
    
    cov : NDArray[np.floating], optional
        The covariance matrix. If none is not provided, it is computed
        from the data directly.
        
    redundancy_function : str
        The redundancy function. 
        Options are `ipm` and `mmi`

    single_target_flag : bool
        Whether to do the PID or PhiID. 

    Returns
    -------
    dict[Atom, float] | tuple[dict[Atom, float], dict[Atom, NDArray[np.floating]]]


    """
    assert redundancy_function in {"ipm", "mmi"}, (
        "The available redundancy functions are Finn and Lizier's ipm and the minimum mutual information."
    )
    if redundancy_function == "ipm":
        assert data is not None, (
            "You must provide data for the ipm redundancy function."
        )

    cov_ = check_cov(cov, data)

    joint = inputs + target
    num_inputs = len(inputs)
    assert num_inputs in (2, 3, 4), (
        "Currently, syntropy only supports PIDs on 2, 3, and 4 inputs."
    )

    num_target = len(target)
    if single_target_flag is False:
        assert num_target in (2, 3), (
            "Currently syntropy only supports \Phi-IDs on 2, and 3 targets."
        )
        lattice: nx.DiGraph = load_lattice(num_inputs=num_inputs, num_target=num_target)
    elif single_target_flag is True:
        lattice: nx.DiGraph = load_lattice(num_inputs=num_inputs)

    kwargs: dict = {
        "inputs": inputs,
        "target": target,
        "single_target_flag": single_target_flag,
    }

    if redundancy_function == "mmi":
        redundancy_func: Callable = mmi_differential_redundancy
        kwargs["cov"] = cov_
        result: nx.DiGraph = mobius_inversion(redundancy_func, lattice, kwargs)

        avg: dict[Atom, float] = {
            node: result.nodes[node]["pi"] for node in result.nodes
        }

        return avg
    elif redundancy_function == "ipm":
        redundancy_func: Callable = ipm_differential_redundancy
        sources = local_precompute_sources(
            data=data[joint, :], cov=cov_[np.ix_(joint, joint)]
        )
        kwargs["sources"] = sources
        result: nx.DiGraph = mobius_inversion(redundancy_func, lattice, kwargs)

        ptw: dict[Atom, NDArray[np.floating]] = {
            node: result.nodes[node]["pi"] for node in result.nodes
        }
        avg: dict[Atom, float] = {key: ptw[key].mean() for key in ptw.keys()}

        return ptw, avg


def partial_information_decomposition(
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    data: NDArray[np.floating] | None,
    cov: NDArray[np.floating] | None = None,
    redundancy_function: str = "ipm",
    ) -> Any:
    """
    Computes the partial information decomposition for up to four input variables onto one (potentially joint) target variable.

    The available redundancy functions are :math:`MMI` and :math:`i_{\\min}` from Finn and Lizier. 

    .. math::

        i_{pm}(\\alpha;t) &= \\min h(\\alpha_i) - \min h(\\alpha_i|t) \\\\
           i_{MMI}(\\alpha;t) &= \\min_i I(\\alpha_i;T) 
   
    Parameters
    ----------
    inputs : tuple[int, ...]
        The indices of the input elements. 
        
    target : tuple[int, ...]
        The indices of the target elements. 
        
    data : NDArray[np.floating]
        The numpy array, assumed to be in channels x time format.
    
    cov : NDArray[np.floating], optional
        The covariance matrix. If none is not provided, it is computed
        from the data directly.
        
    redundancy_function : str
        The redundancy function. 
        Options are `ipm` and `mmi`

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
    
    """
    return _pid(
        inputs=inputs,
        target=target,
        data=data,
        cov=cov,
        redundancy_function=redundancy_function,
        single_target_flag=True,
    )


def integrated_information_decomposition(
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    data: NDArray[np.floating] | None,
    cov: NDArray[np.floating] | None = None,
    redundancy_function: str = "ipm",
) -> Any:
    """
    Computes the integrated information decomposition introduced by Rosas, Mediano, et al.
    The PhiID relaxes the requirement of only having a single target, and instead allows for 
    redundant-redundant, synergistic-synergistic, etc interactions. 

    Available redundancy functions are: 
    
    .. math::

        i_{pm}(\\alpha;\\beta) &= \\min_i h(\\alpha_i) + \\min_i h(\\beta_i) - \min h(\\alpha_i, \\beta_i) \\\\
                i_{MMI}(\\alpha;\\beta) &= \\min_{ij} I(\\alpha_i;\\beta_j) 

    Parameters
    ----------
    inputs : tuple[int, ...]
        The indices of the input elements. 
        
    target : tuple[int, ...]
        The indices of the target elements. 
        
    data : NDArray[np.floating]
        The numpy array, assumed to be in channels x time format.
    
    cov : NDArray[np.floating], optional
        The covariance matrix. If none is not provided, it is computed
        from the data directly.
        
    redundancy_function : str
        The redundancy function. 
        Options are `ipm` and `mmi`
        

    Returns
    -------
    Any
        
    References
    ----------
    Mediano, P. A. M., Rosas, F. E., Luppi, A. I., Carhart-Harris, R. L., Bor, D., Seth, A. K., & Barrett, A. B. (2025). Toward a unified taxonomy of information dynamics via Integrated Information Decomposition. Proceedings of the National Academy of Sciences, 122(39), e2423297122. https://doi.org/10.1073/pnas.2423297122

    Rosas, F. E., Mediano, P. A. M., Jensen, H. J., Seth, A. K., Barrett, A. B., Carhart-Harris, R. L., & Bor, D. (2020). Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data. PLOS Computational Biology, 16(12), Article 12. https://doi.org/10.1371/journal.pcbi.1008289
    """
    return _pid(
        inputs=inputs,
        target=target,
        data=data,
        cov=cov,
        redundancy_function=redundancy_function,
        single_target_flag=False,
    )


def partial_entropy_decomposition(
    data: NDArray[np.floating],
    inputs: tuple[int, ...] | None = None,
    cov: NDArray[np.floating] | None = None,
) -> tuple[dict[Atom, NDArray[np.floating]], dict[Atom, float]]:
    """
    Computes the partial entropy decomposition of a joint distribution
    with up to four elements. Uses the Gaussian hmin estimator. See:


    Parameters
    ----------
    inputs : tuple[int, ...]
        The indices of the elements to analyze.
    data : NDArray[np.floating]
        The numpy array, assumed to be in channels x time format.
    cov : NDArray[np.floating], optional
        The covariance matrix. If none is not provided, it is computed
        from the data directly.

    Returns
    -------
    ptw : dict
        The dictionary of local values for each partial entropy atom.
        The local values are represented by a numpy array of the same
        length as the data array.
    avg : dict
        The dictionary of expected values for each partial entropy atom.

    References
    ----------
    Finn, C., & Lizier, J. T. (2020).
    Generalised Measures of Multivariate Information Content.
    Entropy, 22(2), Article 2.
    https://doi.org/10.3390/e22020216

    """

    cov_: NDArray[np.floating] = check_cov(cov, data)

    if inputs is None:
        inputs = tuple(i for i in range(data.shape[0]))

    num_inputs: int = len(inputs)
    lattice = load_lattice(num_inputs=num_inputs)
    sources: dict[tuple[int, ...], NDArray[np.floating]] = local_precompute_sources(
        data=data, cov=cov_
    )

    kwargs = {"sources": sources}

    result = mobius_inversion(hmin_differential_redundancy, lattice, kwargs)

    ptw = {node: result.nodes[node]["pi"] for node in result.nodes}
    avg = {key: ptw[key].mean() for key in ptw.keys()}

    return ptw, avg


def generalized_information_decomposition(
    inputs: tuple[int, ...],
    data: NDArray[np.floating],
    cov_posterior: NDArray[np.floating],
    cov_prior: NDArray[np.floating],
) -> tuple[dict[Atom, NDArray[np.floating]], dict[Atom, float]]:
    """
    Computes the generalized information decomposition from Varley et al.
    The GID is a decomposition of the Kullback-Leibler divergence of a
    posterior distribution from a prior distribution.

    The available redundancy function is the Gaussian hmin.


    Parameters
    ----------
    inputs : tuple[int, ...]
        The indices of the elements to analyze.
    data : NDArray[np.floating]
        The numpy array, assumed to be in channels x time format.
    cov_posterior : NDArray[np.floating]
        The covariance matrix that defines the prior distribution.
    cov_prior : NDArray[np.floating]
        The covariance matrix that defines the posterior distribution.

    Returns
    -------
    ptw : dict
        The dictionary of local values for each partial entropy atom.
        The local values are represented by a numpy array of the same
        length as the data array.
    avg : dict
        The dictionary of expected values for each partial entropy atom.

    References
    ----------
    Varley, T. F. (2024).
    Generalized decomposition of multivariate information.
    PLOS ONE, 19(2), e0297128.
    https://doi.org/10.1371/journal.pone.0297128
    """

    ptw_prior: dict[Atom, NDArray[np.floating]]
    ptw_prior, _ = partial_entropy_decomposition(
        data=data, inputs=inputs, cov=cov_prior
    )
    ptw_posterior: dict[Atom, NDArray[np.floating]]
    ptw_posterior, _ = partial_entropy_decomposition(
        data=data, inputs=inputs, cov=cov_posterior
    )
    ptw: dict[Atom, NDArray[np.floating]] = {
        key: ptw_prior[key] - ptw_posterior[key] for key in ptw_prior.keys()
    }
    avg: dict[Atom, float] = {key: ptw[key].mean() for key in ptw.keys()}

    return ptw, avg


def representational_complexity(
    avg: dict[Atom, float], comparator: Callable = min
) -> float:
    """
    Computes the representational complexity of a given partial information or entropy lattice.
    The representational complexity is a measure of how
    much partial information atoms of a given degree of synergy
    contribute to the overall mutual information or entropy.

    Parameters
    ----------
    avg : dict
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
    A Measure of the Complexity of Neural Representations based on Partial Information Decomposition.
    Transactions on Machine Learning Research.
    https://openreview.net/forum?id=R8TU3pfzFr

    """

    assert comparator in (min, max, np.min, np.max), "The comparator must be min or max"

    rc: float = 0.0

    atom: Atom
    for atom in avg.keys():
        rc += avg[atom] * comparator(len(source) for source in atom)

    return rc / sum(avg.values())


def idep_partial_information_decomposition(
    inputs: tuple[tuple[int, ...], tuple[int, ...]],
    target: tuple[int, ...],
    cov: NDArray[np.floating] | None, 
    data: NDArray[np.floating] | None = None
) -> dict[str, float]:
    """
    Computes the I_dep partial information decomposition for Gaussian systems
    using the dependency constraint method from Kay & Ince (2018).

    Currently only supports 2 predictors (univariate or multivariate).

    Adapted from: 
        https://github.com/robince/partial-info-decomp/blob/master/calc_pi_Idep_mvn.m


    Parameters
    ----------
    inputs: tuple[tuple[int, ...], tuple[int,...]],
        The indices of the two predictor variables/sets.
        Must have length 2.
    target : tuple[int, ...]
        The indices of the target variable(s).
    data : NDArray[np.floating] | None
        The data in channels x samples format. Optional if cov provided.
    cov : NDArray[np.floating] | None
        The covariance matrix. If None, computed from data.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: 'unq0', 'unq1', 'red', 'syn'

    References
    ----------
    Kay, J. W., & Ince, R. A. A. (2018).
    Exact Partial Information Decompositions for Gaussian Systems
    Based on Dependency Constraints. Entropy, 20(4), 240.
    https://doi.org/10.3390/e20040240
    """

    assert len(inputs) == 2, "I_dep currently only supports 2 predictors"
    
    if cov is None:
        assert data is not None, "You must provide something"
        cov = np.cov(data, ddof=0)

    # Extract indices
    input_0 = inputs[0]
    input_1 = inputs[1]

    n0, n1, nt = len(input_0), len(input_1), len(target)

    # Extract block covariances
    C_00 = cov[np.ix_(input_0, input_0)]
    C_11 = cov[np.ix_(input_1, input_1)]
    C_tt = cov[np.ix_(target, target)]
    C_01 = cov[np.ix_(input_0, input_1)]
    C_0t = cov[np.ix_(input_0, target)]
    C_1t = cov[np.ix_(input_1, target)]

    # Cholesky decomposition (upper triangular)
    # Note: np.linalg.cholesky returns lower triangular, so we transpose
    chol_00 = np.linalg.cholesky(C_00).T
    chol_11 = np.linalg.cholesky(C_11).T
    chol_tt = np.linalg.cholesky(C_tt).T

    # Compute P, Q, R using Kay & Ince Appendix D (Equation A5)
    # P = inv(chol_xx)' * Cxy * inv(chol_yy)
    inv_chol_00 = np.linalg.inv(chol_00)
    inv_chol_11 = np.linalg.inv(chol_11)
    inv_chol_tt = np.linalg.inv(chol_tt)

    P = inv_chol_00.T @ C_01 @ inv_chol_11
    Q = inv_chol_00.T @ C_0t @ inv_chol_tt
    R = inv_chol_11.T @ C_1t @ inv_chol_tt

    # Build standardized covariance matrix for MI calculations

    # Compute basic mutual informations on ORIGINAL covariance
    mi_x0_y = mutual_information(idxs_x=input_0, idxs_y=target, cov=cov)
    mi_x1_y = mutual_information(idxs_x=input_1, idxs_y=target, cov=cov)
    mi_x01_y = mutual_information(idxs_x=input_0 + input_1, idxs_y=target, cov=cov)

    # Compute edge values using TRUE identity matrices
    edge_b = mi_x0_y

    # Edge i: using identity matrices (Kay & Ince Table 9)
    eye_n1 = np.eye(n1)
    eye_n2 = np.eye(nt)

    numerator_det = np.linalg.slogdet(eye_n1 - R @ Q.T @ Q @ R.T)[1]
    denom1_det = np.linalg.slogdet(eye_n2 - Q.T @ Q)[1]
    denom2_det = np.linalg.slogdet(eye_n2 - R.T @ R)[1]

    edge_i = 0.5 * (numerator_det - denom1_det - denom2_det) - mi_x1_y

    # Edge k: build standardized Sigma_Z and compute its determinant
    Sigma_Z = np.zeros((n0 + n1 + nt, n0 + n1 + nt))
    Sigma_Z[:n0, :n0] = np.eye(n0)
    Sigma_Z[n0 : n0 + n1, n0 : n0 + n1] = np.eye(n1)
    Sigma_Z[n0 + n1 :, n0 + n1 :] = np.eye(nt)
    Sigma_Z[:n0, n0 : n0 + n1] = P
    Sigma_Z[n0 : n0 + n1, :n0] = P.T
    Sigma_Z[:n0, n0 + n1 :] = Q
    Sigma_Z[n0 + n1 :, :n0] = Q.T
    Sigma_Z[n0 : n0 + n1, n0 + n1 :] = R
    Sigma_Z[n0 + n1 :, n0 : n0 + n1] = R.T

    k_num = 0.5 * np.linalg.slogdet(np.eye(n1) - P.T @ P)[1]
    k_den = 0.5 * np.linalg.slogdet(Sigma_Z)[1]
    edge_k = k_num - k_den - mi_x1_y

    # Unique information from X0 is minimum across all edges adding X0Y
    unq0 = min(edge_b, edge_i, edge_k)

    # Derive other components
    red = mi_x0_y - unq0
    unq1 = mi_x1_y - red
    syn = mi_x01_y - mi_x1_y - unq0

    return {
        "unq0": unq0,
        "unq1": unq1,
        "red": red,
        "syn": syn,
        # # For debugging/analysis
        # 'I_X0_Y': mi_x0_y,
        # 'I_X1_Y': mi_x1_y,
        # 'I_X0X1_Y': mi_x01_y,
        # 'edges': {'b': edge_b, 'i': edge_i, 'k': edge_k}
    }

import numpy as np
import networkx as nx
from .shannon import local_differential_entropy

from .utils import COV_NULL, make_powerset
from ..lattices import LATTICE_2, LATTICE_3, LATTICE_4
from numpy.typing import NDArray

Atom = tuple[tuple[int, ...], ...]
BOTTOM_2: Atom = ((0,), (1,))
BOTTOM_3: Atom = ((0,), (1,), (2,))
BOTTOM_4: Atom = ((0,), (1,), (2,), (3,))

PATHS_2: dict = nx.shortest_paths.shortest_path_length(
    LATTICE_2, source=None, target=BOTTOM_2
)
LAYERS_2: dict[int, int] = {
    val: {key for key in PATHS_2.keys() if PATHS_2[key] == val}
    for val in set(PATHS_2.values())
}

PATHS_3: dict[int, int] = nx.shortest_paths.shortest_path_length(
    LATTICE_3, source=None, target=BOTTOM_3
)
LAYERS_3: dict[int, int] = {
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
def unpack_atom(atom: tuple) -> set:
    """
    A utitlity function to unpack tuples.
    """
    varset: set = set()
    for s in {*atom}:
        varset.update(set(s))

    return varset


def local_precompute_sources(data: NDArray[np.floating], cov: NDArray[np.floating] = COV_NULL):
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
    if cov[0][0] == -1:
        cov = np.cov(data, ddof=0.0)

    N: int = data.shape[0]
    joint: tuple = tuple(i for i in range(N))
    sources: list = list(make_powerset(joint))
    sources.remove(())

    return {
        source: local_differential_entropy(data[source, :], cov[source, :][:, source])
        for source in sources
    }


def hmin_differential_redundancy(
    atom: tuple, sources: dict, data: NDArray[np.floating], cov: NDArray[np.floating] = COV_NULL
) -> NDArray[np.floating]:
    """
    For a collection of sources :math:`\\alpha=\\{a_1, a_2, \\ldots, a_k\\}`, computes
    the redundnat entropy shared by all sources as:

    :math:`h_{\\cap}^{min}(\\alpha) = \\min_{i}(a_{i})`

    See:
        Finn, C., & Lizier, J. T. (2020).
        Generalised Measures of Multivariate Information Content.
        Entropy, 22(2), Article 2.
        https://doi.org/10.3390/e22020216

    Parameters
    ----------
    atom : tuple
        The partial information atom. In the form :math:`((a_1,),(a_2,)\\ldots)`.
    sources : dict
        The pre-computed collection of sources.
    data : NDArray[np.floating]
        The data, assumed to be in sources x samples format.

    Returns
    -------
    NDArray[np.floating]
        The local redundancies for each sample.


    """
    h_min: NDArray[np.floating] = np.repeat(np.inf, data.shape[1])
    source: tuple = tuple()

    for source in atom:
        # i+ = min(h(source))
        h_min = np.minimum(h_min, sources[source])

    return h_min


def imin_differential_redundancy(
    atom: tuple,
    sources: dict,
    inputs: tuple,
    target: tuple,
    data: NDArray[np.floating],
    cov: NDArray[np.floating] = COV_NULL,
) -> NDArray[np.floating]:
    """
    Computes the differential redundnacy between a partial information atom :math:`\\alpha=\\{a_i,\\ldots,a_k\\}` and a (potentially multivariate) target. Uses a Gaussian estimator for the local entropies.

    :math:`i_{\\cap}^{min}(\\alpha;t) = \\min_{i}h(a_i) - \\min_{i}h(a_i|t)`

    This is NOT the I_min from Williams and Beer - it is the local redundancy function from Finn and Lizier.

    See:
        Finn, C., & Lizier, J. T. (2018).
        Pointwise Partial Information Decomposition Using
        the Specificity and Ambiguity Lattices.
        Entropy, 20(4), Article 4.
        https://doi.org/10.3390/e20040297

    Parameters
    ----------
    atom : tuple
        The partial information atom. In the form :math:`((a_1,),(a_2,)\\ldots)`.
    sources : dict
        The pre-computed collection of sources.
    inputs : tuple
        The indicies of the inputs - one index per element of the tuple.
    target : tuple
        The (potentially multivariate) indices of the collective target.
    data : NDArray[np.floating]
        The data, assumed to be in channels x samples format.

    Returns
    -------
    NDArray[np.floating]
        The local differnetial redundancy for each frame.

    """
    assert data.shape[0] == (len(inputs) + len(target))
    if cov[0][0] == -1:
        cov = np.cov(data, ddof=0)

    i_plus: NDArray[np.floating] = np.repeat(np.inf, data.shape[1])
    i_minus: NDArray[np.floating] = np.repeat(np.inf, data.shape[1])

    h_target: NDArray[np.floating] = sources[
        target
    ]  # No need to transform it. It is the variable idx.

    source: tuple = tuple()
    for (
        source
    ) in atom:  # All atom idxs need to be transformed into source variable idxs.
        # Atom idx to source variable idx conversion.
        # Most of the time this is redundnat, but in case someone tries to call
        # the redundancy function directly from a big time series array, this
        # conversion is important for safety.
        source_inputs: tuple = tuple(inputs[x] for x in source)
        joint_inputs: tuple = tuple(sorted(source_inputs + target))

        h_joint: NDArray[np.floating] = sources[joint_inputs]

        h_conditional: NDArray[np.floating] = h_joint - h_target

        # i+ = min(h(source))
        i_plus = np.minimum(i_plus, sources[source_inputs])
        # i- = min(h(source|target))
        i_minus = np.minimum(i_minus, h_conditional)

    i_min: NDArray[np.floating] = i_plus - i_minus

    return i_min


def mobius_inversion(
    decomposition: str,
    data: NDArray[np.floating],
    inputs: tuple,
    target: tuple = (None,),
    cov: NDArray[np.floating] = COV_NULL,
) -> tuple[dict, dict]:
    """
    Computes the Mobius inversion on a lattice, given a redundancy function.

    See:
        Williams, P. L., & Beer, R. D. (2010).
        Nonnegative Decomposition of Multivariate Information.
        arXiv:1004.2515 [Math-Ph, Physics:Physics, q-Bio].
        http://arxiv.org/abs/1004.2515

    Parameters
    ----------
    decomposition : str
        Whehter to do a PID or PED. Options: "pid", "ped".
    data : NDArray[np.floating]
        The data, assumed to be in channels x samples format.
    inputs : tuple
        The variables to be decomposed.
    target : tuple, optional
        If doing a PID, the indices of the target.
        If doing a PED, leave false.
        The default is (None,).
    cov : NDArray[np.floating], optional
        The covariance matrix that defines the multivairate distribution.
        If left empty, it is computed from the data. The default is COV_NULL.

    Returns
    -------
    ptw : dict
        The pointwise values of each atom in the decomposition for each sample.
    avg : dict
        The expected values of each atom in the decomposition.

    """

    decomposition_lower: str = decomposition.lower()
    assert decomposition_lower in {
        "pid",
        "ped",
    }, "You must specify a decomposition: PID or PED."

    if decomposition_lower == "pid":
        assert target != (None,), "You must specify a target."

    if cov[0][0] == -1:
        cov = np.cov(data, ddof=0)

    if len(inputs) == 2:
        lattice, bottom, layers = LATTICE_2, BOTTOM_2, LAYERS_2
    elif len(inputs) == 3:  #    |
        lattice, bottom, layers = LATTICE_3, BOTTOM_3, LAYERS_3
    else:  #                      |
        lattice, bottom, layers = LATTICE_4, BOTTOM_4, LAYERS_4

    # Pre-computing all the local h_{\partial}(source) values
    # this saves a lot of time.

    sources: dict = local_precompute_sources(data=data, cov=cov)

    total_str: str = ""
    partial_str: str = ""

    if decomposition_lower == "pid":
        total_str = "total_information"
        partial_str = "pi"
    elif decomposition_lower == "ped":
        total_str = "total_entropy"
        partial_str = "pe"

    for layer in layers:
        for atom in layers[layer]:
            if decomposition_lower == "pid":
                lattice.nodes[atom][total_str] = imin_differential_redundancy(
                    atom, sources, inputs, target, data, cov
                )
            elif decomposition_lower == "ped":
                lattice.nodes[atom][total_str] = hmin_differential_redundancy(
                    atom, sources, data, cov
                )

            if atom == bottom:
                lattice.nodes[atom][partial_str] = lattice.nodes[atom][total_str]
            else:
                lattice.nodes[atom][partial_str] = lattice.nodes[atom][total_str] - sum(
                    [
                        lattice.nodes[d][partial_str]
                        for d in lattice.nodes[atom]["descendants"]
                    ]
                )

    ptw: dict = {node: lattice.nodes[node][partial_str] for node in lattice.nodes}
    avg: dict = {
        node: lattice.nodes[node][partial_str].mean() for node in lattice.nodes
    }

    return (ptw, avg)

def partial_information_decomposition(
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    data: NDArray[np.floating],
    cov: NDArray[np.floating] = COV_NULL,
) -> tuple[dict, dict]:
    """
    The pointwise and average partial information decomposition
    using the Gaussian imin function. See:
    
    See:
        Finn, C., & Lizier, J. T. (2018).
        Pointwise Partial Information Decomposition Using the Specificity and Ambiguity Lattices.
        Entropy, 20(4), Article 4.
        https://doi.org/10.3390/e20040297

    Parameters
    ----------
    inputs : tuple[int, ...]
        The indices of the input variables.
    target : tuple[int, ...]
        The indices of the target variable(s).
    data : NDArray[np.floating]
        The data in channels x time format.
    cov : NDArray[np.floating]
        The covariance matrix of the data.
        The default is COV_NULL.

    Returns
    -------
    ptw : dict
        The pointwise PID for every frame in the data.
    avg : dict
        The average PID for the data

    """

    if cov[0][0] == -1:
        cov = np.cov(data, ddof=0)
    else:
        assert cov.shape[0] == data.shape[0], (
            "The covariance matrix must have the same dimensionality as the data."
        )

    N_inputs: int = len(inputs)
    N_target: int = len(target)
    joint: tuple = inputs + target

    ptw, avg = mobius_inversion(
        decomposition="pid",
        data=np.vstack((data[inputs, :], data[target, :])),
        inputs=tuple(i for i in range(N_inputs)),
        target=tuple(i + N_inputs for i in range(N_target)),
        cov=cov[np.ix_(joint, joint)],
    )

    return ptw, avg


def partial_entropy_decomposition(
    inputs: tuple[int, ...],
    data: NDArray[np.floating],
    cov: NDArray[np.floating] = COV_NULL,
) -> tuple[dict, dict]:
    """
    Computes the partial entropy decomposition of a joint distribution
    with up to four elements. Uses the Gaussian hmin estimator. See:

        Finn, C., & Lizier, J. T. (2020).
        Generalised Measures of Multivariate Information Content.
        Entropy, 22(2), Article 2.
        https://doi.org/10.3390/e22020216


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

    """
    if cov[0][0] == -1:
        cov = np.cov(data, ddof=0)
    else:
        assert cov.shape[0] == data.shape[0], (
            "The covariance matrix must have the same dimensionality as the data."
        )

    N_inputs: int = len(inputs)

    ptw, avg = mobius_inversion(
        decomposition="ped",
        data=data[inputs, :],
        inputs=tuple(i for i in range(N_inputs)),
        cov=cov[np.ix_(inputs, inputs)],
    )

    return ptw, avg


def generalized_information_decomposition(
    inputs: tuple[int, ...],
    data: NDArray[np.floating],
    cov_posterior: NDArray[np.floating],
    cov_prior: NDArray[np.floating],
) -> tuple[dict, dict]:
    """
    Computes the generalized information decomposition from Varley et al.
    The GID is a decomposition of the Kullback-Leibler divergence of a
    posterior distribution from a prior distribution.

    The available redundancy function is the Gaussian hmin.

    See:
        Varley, T. F. (2024).
        Generalized decomposition of multivariate information.
        PLOS ONE, 19(2), e0297128.
        https://doi.org/10.1371/journal.pone.0297128

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

    """

    N_inputs: int = len(inputs)

    ptw_prior, _ = mobius_inversion(
        decomposition="ped",
        data=data[inputs, :],
        inputs=tuple(i for i in range(N_inputs)),
        cov=cov_prior[np.ix_(inputs, inputs)],
    )

    ptw_posterior, _ = mobius_inversion(
        decomposition="ped",
        data=data[inputs, :],
        inputs=tuple(i for i in range(N_inputs)),
        cov=cov_posterior[np.ix_(inputs, inputs)],
    )

    ptw: dict = {key: ptw_prior[key] - ptw_posterior[key] for key in ptw_prior.keys()}
    avg: dict = {key: ptw[key].mean() for key in ptw.keys()}

    return ptw, avg


def representational_complexity(avg: dict, comparator=min) -> float:
    """
    Computes the representational complexity of a given partial information or entropy lattice.
    The representational complexity is a measure of how
    much partial information atoms of a given degree of synergy
    contribute to the overall mutual information or entropy.

    See:
        Ehrlich, D. A., Schneider, A. C., Priesemann, V., Wibral, M., & Makkeh, A. (2023).
        A Measure of the Complexity of Neural Representations
        based on Partial Information Decomposition.
        Transactions on Machine Learning Research.
        https://openreview.net/forum?id=R8TU3pfzFr



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

    """

    assert comparator in (min, max, np.min, np.max), "The comparator must be min or max"

    rc: float = 0.0

    for atom in avg.keys():
        rc += avg[atom] * comparator(len(source) for source in atom)

    return rc / sum(avg.values())

import numpy as np
import networkx as nx
import itertools as it
from numpy.typing import NDArray

from ..lattices import LATTICE_2, LATTICE_3, LATTICE_4

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

COV_NULL = np.array([[-1]])

# %% LIBRARY


def check_cov(
    cov: NDArray[np.floating], data: NDArray[np.floating]
) -> NDArray[np.floating]:
    if cov[0, 0] == -1:
        cov_ = np.cov(data, ddof=0.0)
    else:
        cov_ = cov.copy()

    return cov_


def make_powerset(iterable):
    """
    Computes the powerset of a collection of elements.

    :math:`\\mathcal{P}(\\{X_1,X_2,X_3\\}) \\to (\\{\\}, \\{X_1\\}, \\{X_2\\}, \\{X_3\\}, \\{X_1,X_2\\}, \\{X_1,X_3\\}, \\{X_1,X_2,X_3\\} )`

    """
    xs: list = list(iterable)
    # note we return an iterator rather than a list
    return it.chain.from_iterable(it.combinations(xs, n) for n in range(len(xs) + 1))


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
    data : np.ndarray
        The data in channels x samples format.
    cov : np.ndarray, optional
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
    atom: tuple, sources: dict, data: np.ndarray, cov: np.ndarray = COV_NULL
) -> np.ndarray:
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
    data : np.ndarray
        The data, assumed to be in sources x samples format.

    Returns
    -------
    np.ndarray
        The local redundancies for each sample.


    """
    h_min: np.ndarray = np.repeat(np.inf, data.shape[1])
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
    data: np.ndarray,
    cov: np.ndarray = COV_NULL,
) -> np.ndarray:
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
    data : np.ndarray
        The data, assumed to be in channels x samples format.

    Returns
    -------
    np.ndarray
        The local differnetial redundancy for each frame.

    """
    assert data.shape[0] == (len(inputs) + len(target))
    if cov[0][0] == -1:
        cov = np.cov(data, ddof=0)

    i_plus: np.ndarray = np.repeat(np.inf, data.shape[1])
    i_minus: np.ndarray = np.repeat(np.inf, data.shape[1])

    h_target: np.ndarray = sources[
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

        h_joint: np.ndarray = sources[joint_inputs]

        h_conditional: np.ndarray = h_joint - h_target

        # i+ = min(h(source))
        i_plus = np.minimum(i_plus, sources[source_inputs])
        # i- = min(h(source|target))
        i_minus = np.minimum(i_minus, h_conditional)

    i_min: np.ndarray = i_plus - i_minus

    return i_min


def mobius_inversion(
    decomposition: str,
    data: np.ndarray,
    inputs: tuple,
    target: tuple = (None,),
    cov: np.ndarray = COV_NULL,
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
    data : np.ndarray
        The data, assumed to be in channels x samples format.
    inputs : tuple
        The variables to be decomposed.
    target : tuple, optional
        If doing a PID, the indices of the target.
        If doing a PED, leave false.
        The default is (None,).
    cov : np.ndarray, optional
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


def correlation_to_mutual_information(cov: np.ndarray) -> np.ndarray:
    """
    Converts a Pearson correlation matrix to a Guassian mutual
    information matrix based on the identity:

    :math:`I(X;Y) = \\frac{-\\log(1-(r_{XY})^2)}{2}`

    Where :math:`r_{XY}` is the Pearson correlation coefficient between :math:`X` and :math:`Y`.

    Also works for a covariance matrix if the processes have 0 mean
    and unit variance.

    Parameters
    ----------
    cov : np.ndarray
        A covariance matrix.

    Returns
    -------
    np.ndarray
        The equivalent mutual information matrix.

    """
    mi = -np.log(1 - (cov**2)) / 2.0
    np.fill_diagonal(mi, np.nan)

    return mi

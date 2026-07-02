import pathlib
import pickle
import networkx as nx

lattice_path = pathlib.Path(__file__).parent


def load_lattice(num_inputs: int, num_target: int = 1) -> nx.DiGraph:
    """
    Loads a pre-computed redundancy lattice from disk.

    Parameters
    ----------
    num_inputs : int
        The number of input elements the lattice was built for.
    num_target : int, optional
        The number of (potentially joint) target elements. If 1, a
        single-target PID lattice is loaded; if greater than 1, the
        corresponding multi-target Phi-ID lattice is loaded. The default
        is 1.

    Returns
    -------
    nx.DiGraph
        The redundancy lattice, as a directed graph of information atoms.

    References
    ----------
    Williams, P. L., & Beer, R. D. (2010).
    Nonnegative Decomposition of Multivariate Information.
    arXiv:1004.2515 [Math-Ph, Physics:Physics, q-Bio].
    http://arxiv.org/abs/1004.2515

    """
    if num_target == 1:
        path = lattice_path / f"pi_lattice_{str(num_inputs)}.pickle"
    elif num_target > 1:
        path = lattice_path / f"pi_lattice_{str(num_inputs) + str(num_target)}.pickle"

    with open(path, "rb") as f:
        lattice = pickle.load(f)

    return lattice

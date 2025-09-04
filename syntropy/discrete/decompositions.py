import numpy as np
from syntropy.discrete.utils import mobius_inversion


def partial_information_decomposition(
    redundancy: str,
    inputs: tuple,
    target: tuple,
    joint_distribution: dict[tuple, float],
) -> (dict, dict):
    """
    Computes the partial information decomposition for up to four input variables
    onto one (potentially joint) target variable.

    The available redundancy functions are i_min from Finn and Lizier
    and i_sx from Makkeh et al.. See:

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
    redundancy: str, joint_distribution: dict[tuple, float]
) -> (dict, dict):
    """
    Computes the partial entropy decomposition of a joint distribution
    with up to four elements.

    The available redundancy functions are h_min from Finn and Lizier
    and h_sx from Varley et al., See:

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

    """

    ptw, avg = mobius_inversion(
        decomposition="ped",
        joint_distribution=joint_distribution,
        redundancy=redundancy,
    )

    return ptw, avg


def generalized_information_decomposition(
    redundancy: str, posterior_distribution: dict[tuple,float], prior_distribution: dict[tuple,float]
) -> (dict, dict):
    """
    Computes the generalized information decomposition from Varley et al.
    The GID is a decomposition of the Kullback-Leibler divergence of a
    posterior distribution from a prior distribution.

    Available redundnacy functions are "hmin" and "hsx". See
    the documentation for the partial_entropy_decomposition() function
    for details.

    See:
        Varley, T. F. (2024).
        Generalized decomposition of multivariate information.
        PLOS ONE, 19(2), e0297128.
        https://doi.org/10.1371/journal.pone.0297128


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

    """
    assert set(prior_distribution.keys()).issuperset(
        set(posterior_distribution.keys())
    ), "The support set of the prior must be a superset of the support set of the posterior."
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

    """

    assert comparator in (min, max, np.min, np.max), "The comparator must be min or max"

    rc: float = 0.0

    for atom in avg.keys():
        rc += avg[atom] * comparator(len(source) for source in atom)

    return rc / sum(avg.values())

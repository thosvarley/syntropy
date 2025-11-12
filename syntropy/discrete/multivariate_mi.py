import numpy as np
import itertools as it
from scipy.special import comb
from .utils import (
    reduce_state,
    get_marginal_distribution,
    marginalize_out,
    get_all_marginal_distributions,
)
from .shannon import kullback_leibler_divergence, shannon_entropy
from .optimization import constrained_maximum_entropy_distributions

binom_lookup = {N: {k: comb(N, k, exact=True) for k in range(N)} for N in range(16)}


def total_correlation(
    joint_distribution: dict[tuple[int, ...], float],
) -> (tuple[dict, float], float):
    """
    Computes the average and pointwise total correlations:

    .. math::

        TC(X) &= D_{KL}(P(X) || \\prod_{i=1}^{N}P(X_i) \\\\
              &= \\sum_{i=1}^{N}H(X_i) - H(X)

    Parameters
    ----------
    joint_distribution : dict[tuple[int, ...]], float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    ptw : dict[tuple, float]
        The pointwise TC .
    avg : float
        The average TC.
    
    References
    ----------
    Watanabe, S. (1960). Information Theoretical Analysis of Multivariate Correlation.
    IBM Journal of Research and Development, 4(1), Article 1.
    https://doi.org/10.1147/rd.41.0066

    Tononi, G., Sporns, O., & Edelman, G. M. (1994). 
    A measure for brain complexity: Relating functional segregation and integration in the nervous system. 
    Proceedings of the National Academy of Sciences, 91(11), Article 11. 
    https://doi.org/10.1073/pnas.91.11.5033
    
    """

    maxent_distribution = constrained_maximum_entropy_distributions(
        joint_distribution, order=1
    )

    ptw, avg = kullback_leibler_divergence(joint_distribution, maxent_distribution)

    return ptw, avg


def k_wms(
    k: int, joint_distribution: dict[tuple[int, ...], float]
) -> (tuple[dict, float], float):
    """
    S-information, DTC, and negative O-information can all be written in a general form:

    .. math::

        WMS^{k}(X) = (N-k)TC(X) - \\sum_{i=1}^{N}TC(X^{-i})

    Parameters
    ----------
    k : int
        The scale parameter
    joint_distribution : dict[tuple[int, ...], float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    ptw : dict[tuple[int, ...], float]
        The pointwise w-WMS .
    avg : float
        The average k-WMS.

    References
    ----------
    Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
    Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
    Communications Biology, 6(1), Article 1.
    https://doi.org/10.1038/s42003-023-04843-w

    """

    states: list[tuple[int, ...]] = list(joint_distribution.keys())
    N: int = len(states[0])

    ptw_tc: dict
    avg_tc: float

    ptw_tc, avg_tc = total_correlation(joint_distribution)

    avg_whole: float = (N - k) * avg_tc
    avg_sum_parts: float = 0.0

    ptw_whole = {state: ptw_tc[state] * (N - k) for state in states}
    ptw_sum_parts = {state: 0 for state in states}

    for i in range(N):
        residuals: tuple = tuple(j for j in range(N) if j != i)

        reduced_distribution: dict = marginalize_out((i,), joint_distribution)

        ptw_r: dict
        avg_r: dict
        ptw_r, avg_r = total_correlation(reduced_distribution)

        avg_sum_parts += avg_r

        for state in ptw_r:
            full_states = {s for s in states if reduce_state(s, residuals) == state}

            for full_state in full_states:
                ptw_sum_parts[full_state] += ptw_r[state]

    ptw: dict = {state: ptw_whole[state] - ptw_sum_parts[state] for state in states}
    avg: float = avg_whole - avg_sum_parts

    return ptw, avg


def s_information(
    joint_distribution: dict[tuple[int, ...], float],
) -> (tuple[dict, float], float):
    """
    Computes the local and expected S-information for the joint distribution.

    .. math:: 

        \\Sigma(X) &= \\sum_{i=1}^{N}I(X_i;X^{-i}) \\\\
                   &= N\\times TC(X) - \\sum_{i=1}^{N}TC(X^{-i}) \\\\
                   &= TC(X) + DTC(X)
        
    Parameters
    ----------
    joint_distribution : dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    ptw : dict[tuple, float]
        The pointwise S-information .
    avg : float
        The average S-information.

    References
    ----------
    Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
    Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information.
    Physical Review E, 100(3), Article 3.
    https://doi.org/10.1103/PhysRevE.100.032305

    Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
    Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
    Communications Biology, 6(1), Article 1.
    https://doi.org/10.1038/s42003-023-04843-w

    """

    ptw: dict
    avg: float
    ptw, avg = k_wms(k=0, joint_distribution=joint_distribution)

    return ptw, avg


def dual_total_correlation(
    joint_distribution: dict[tuple[int, ...], float],
) -> (tuple[dict, float], float):
    """
    Computes the local and expected dual total correlations for the joint distribution.

    .. math:: 

        DTC(X) &= H(X) - \\sum_{i=1}^{N}H(X_i|X^{-i}) \\\\
               &= (N-1)\\times TC(X) - \\sum_{i=1}^{N}TC(X^{-i})

    Parameters
    ----------
    joint_distribution : dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    ptw : dict[tuple, float]
        The pointwise DTC .
    avg : float
        The average DTC.

    References
    ----------
    Abdallah, S. A., & Plumbley, M. D. (2012).
    A measure of statistical complexity based on predictive information with application to finite spin systems.
    Physics Letters A, 376(4), 275–281.
    https://doi.org/10.1016/j.physleta.2011.10.066

    Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
    Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information.
    Physical Review E, 100(3), Article 3.
    https://doi.org/10.1103/PhysRevE.100.032305

    """
    ptw: dict
    avg: float
    ptw, avg = k_wms(k=1, joint_distribution=joint_distribution)

    return ptw, avg


def o_information(
    joint_distribution: dict[tuple[int, ...], float],
) -> (tuple[dict, float], float):
    """
    Computes the local and expected O-informations for the joint distribution.
    O-information quantifies the balance between redundancy (positive values) and synergy (negative values) in multivariate information.

    .. math::

        \\Omega(X) &= (2-N)TC(X) + \\sum_{i=1}^{N}TC(X^{-i}) \\\\
                   &= TC(X) - DTC(X)

    Parameters
    ----------
    joint_distribution : dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    ptw : dict[tuple, float]
        The pointwise O-information .
    avg : float
        The average O-information.
    
    References
    ----------
    Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
    Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information.
    Physical Review E, 100(3), Article 3.
    https://doi.org/10.1103/PhysRevE.100.032305

    Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
    Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
    Communications Biology, 6(1), Article 1.
    https://doi.org/10.1038/s42003-023-04843-w

    """

    ptw: dict
    avg: float
    ptw, avg = k_wms(k=2, joint_distribution=joint_distribution)

    return {state: -ptw[state] for state in ptw.keys()}, -avg


def co_information(
    joint_distribution: dict[tuple[int, ...], float],
) -> (tuple[dict, float], float):
    """
    Computes the cO-information, the third generalization of bivariate mutual information. Unlike total correlation and dual total correlation, the cO-information can be negative and is difficult to interpret.

    .. math::
        Co(X) = \\sum_{\\xi\\subseteq X}(-1)^{|\\xi|}H(\\xi)

    Parameters
    ----------
    joint_distribution : dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    ptw : dict[tuple, float]
        The pointwise cO-information.
    avg : float
        The average cO-information.

    References
    ----------
    Bell, A. J. (2003, April).
    The Co-information lattice.
    4th International Symposium on Independent Component Analysis and
    Blind Signal Separation, Nara, Japan.
    https://www.semanticscholar.org/paper/THE-CO-INFORMATION-LATTICE-Bell/25a0cd8d486d5ffd204485685226f189e6eadd4d

    """

    # Get the lattice of marginal distributions.
    marginals: dict = get_all_marginal_distributions(joint_distribution)
    # Convert them to local entropies as a batch.
    h_marginals = {key: shannon_entropy(marginals[key])[0] for key in marginals}

    ptw: dict = {state: 0.0 for state in joint_distribution.keys()}
    avg: float = 0.0

    for state in joint_distribution.keys():
        for source in h_marginals.keys():
            sign = (-1) ** len(source)

            ptw[state] -= sign * h_marginals[source][reduce_state(state, source)]

        avg += joint_distribution[state] * ptw[state]

    return ptw, avg


def tse_complexity(
    joint_distribution: dict[tuple[int, ...], float], num_samples
) -> float:
    """
    The Tononi-Sporns-Edelman neural complexity measure, which provides a measure of the balance between integration and segregation across scales.

    .. math::

        TSE(X) &= \\sum_{k=1}^{\\lfloor N/2\\rfloor} \\bigg\\langle I(X^{k}_j;X^{-k}_j) \\bigg\\rangle_{j} \\\\
               &= \\sum_{k=2}^{N}\\bigg[\\bigg(\\frac{k}{N}\\bigg)TC(X) - \\langle TC(X^{k}_{j}) \\rangle_{j}  \\bigg] 

    Runtimes scale very badly with system size (as it requires brute-forcing) all possible bipartitions of the system. If the system is too large, a sub-sampling approach is taken: at each scale, num_samples are drawn from the space of bipartitions.

    Parameters
    ----------
    joint_distribution : dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.
    num_samples : int
        The number of samples to do for each subset size..

    Returns
    -------
    float
        The TSE complexity. No local complexity is computed. .

    References
    ----------
    Tononi, G., Sporns, O., & Edelman, G. M. (1994).
    A measure for brain complexity: Relating functional segregation and integration in the nervous system.
    Proceedings of the National Academy of Sciences, 91(11), Article 11.
    https://doi.org/10.1073/pnas.91.11.5033

    Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
    Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
    Communications Biology, 6(1), Article 1.
    https://doi.org/10.1038/s42003-023-04843-w

    """
    N: int = len(list(joint_distribution.keys())[0])

    tc_whole: float = total_correlation(joint_distribution)[1]

    null_tcs: np.ndarray = np.array(
        [(float(i) / float(N)) * tc_whole for i in range(1, N + 1)]
    )
    exp_tcs: np.ndarray = np.zeros(null_tcs.shape[0])
    exp_tcs[-1] = tc_whole

    for k in range(1, N):  # For each layer of the TSE-leaf.
        binom: int = binom_lookup[N][k]

        if (
            binom > num_samples
        ):  # if N-choose-k is greater than the pre-specified number of samples:
            # Samples is a set to avoid repeats.
            samples: set = {
                tuple(sorted(tuple(np.random.choice(N, size=k, replace=False))))
                for i in range(num_samples)
            }
        else:
            samples: set = set(it.combinations(range(N), k))

        tcs: float = 0.0

        sample: tuple
        for sample in samples:
            marginal: dict = get_marginal_distribution(sample, joint_distribution)
            tcs += total_correlation(marginal)[1]

        exp_tcs[k - 1] = tcs / len(samples)

    return (null_tcs - exp_tcs).sum()


def description_complexity(
    joint_distribution: dict[tuple[int, ...], float],
) -> (tuple[dict, float], float):
    """
    The description complexity was proposed by Tononi and Sporns as a
    heuristic, easy-to-compute approximation of the full TSE-Complexity.
    Later shown by Varley et al., to be directly proportional to
    the dual total correlation.

    .. math::

        C(X) = \\frac{DTC(X)}{N}


    Where :math:`N` is the number of elements in :math:`X`.

    Parameters
    ----------
    joint_distribution : dict[tuple, float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.

    Returns
    -------
    ptw : dict[tuple, float]
        The pointwise description complexity.
    avg : float
        The average description complexity.

    References
    ----------
    Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
    Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
    Communications Biology, 6(1), Article 1.
    https://doi.org/10.1038/s42003-023-04843-w

    """

    N = float(len(list(joint_distribution.keys())[0]))

    ptw: dict
    avg: float
    ptw, avg = k_wms(k=1, joint_distribution=joint_distribution)

    avg /= N
    ptw = {key: ptw[key] / N for key in ptw.keys()}

    return ptw, avg


def connected_information(
    joint_distribution: dict[tuple[int, ...], float], maximum_order: int = -1
) -> list[float]:
    """
    Returns the connected information profile from Schneidman et al.,
    which decomposes the total correlation into contributing parts of
    different orders:

    .. math::
        TC(X) = \\sum_{k=2}^{N}TC^{k}(X)

    Where the `k` superscript refers to the maximum-entropy distribution that preserves all marginals of order `k`.

    One of the few measures that can reliably distinguish between the
    JAMES_DYADIC and JAMES_TRIADIC distributions.

    Parameters
    ----------
    joint_distribution : dict[tuple[int, ...], float]
        The joint probability distribution.
        Keys are tuples corresponding to the state of each element.
        The valules are the probabilities.
    maximum_order : int, optional
        The highest order of marginals to sweep.
        The default sweeps all.

    Returns
    -------
    list[float]
        The connected information profile.

    References
    ----------
    Schneidman, E., Still, S., Berry, M. J., & Bialek, W. (2003).
    Network Information and Connected Correlations.
    Physical Review Letters, 91(23), 238701.
    https://doi.org/10.1103/PhysRevLett.91.238701

    """

    N: int = len(list(joint_distribution.keys())[0])
    if maximum_order == -1:
        maximum_order = N
    
    profile: list[float] = []

    for order in range(1, maximum_order + 1):
        maxent: dict[tuple[int, ...], float] = (
            constrained_maximum_entropy_distributions(
                joint_distribution=joint_distribution, order=order
            )
        )
        profile.append(shannon_entropy(maxent)[1])

    profile = [profile[i-1] - profile[i] for i in range(1, len(profile))]

    return profile

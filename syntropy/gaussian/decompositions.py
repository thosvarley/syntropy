import numpy as np
from syntropy.gaussian.utils import COV_NULL, mobius_inversion
from numpy.typing import NDArray


def partial_information_decomposition(
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    data: NDArray[np.floating],
    cov: NDArray[np.floating] = COV_NULL,
) -> tuple[dict, dict]:
    """
     The pointwise and average partial information decomposition
     using the Gaussian imin function. See:

         Finn, C., & Lizier, J. T. (2018).
         Pointwise Partial Information Decomposition Using
         the Specificity and Ambiguity Lattices.
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
        cov=cov_prior[np.ix_(inputs, inputs)],
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

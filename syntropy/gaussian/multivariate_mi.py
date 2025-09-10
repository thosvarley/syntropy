import numpy as np
from syntropy.gaussian.utils import COV_NULL
from syntropy.gaussian.shannon import local_differential_entropy


def local_total_correlation(
    data: np.ndarray, cov: np.ndarray, inputs: tuple = (-1,)
) -> np.ndarray:
    """
    The local total correlation.

    See:
        Scagliarini, T., Marinazzo, D., Guo, Y., Stramaglia, S., & Rosas, F. E. (2022).
        Quantifying high-order interdependencies on individual patterns via the local O-information:
            Theory and applications to music analysis.
        Physical Review Research, 4(1), 013184.
        https://doi.org/10.1103/PhysRevResearch.4.013184


    Parameters
    ----------
    data : np.ndarray
        The data in channels x samples format.
    inptuts: tuple
        The indices of the channels to include.
    Returns
    -------
    np.ndarray
        The local total correaltion for each frame.

    """
    assert (
        cov.shape[0] == data.shape[0]
    ), "The data and covariance matrix must have the same dimensionality"

    if inputs[0] == -1:
        _inputs = tuple(i for i in range(data.shape[0]))
    else:
        _inputs = inputs
    N = len(_inputs)

    whole = local_differential_entropy(data[_inputs, :], cov[_inputs, :][:, _inputs])

    sum_parts = np.zeros_like(whole)
    for i in range(N):
        sum_parts += local_differential_entropy(data[_inputs[i], :])

    return sum_parts - whole


def total_correlation(cov: np.ndarray, inputs: tuple = (-1,)) -> float:
    """
    The expected total correlation.

    See:
        Watanabe, S. (1960). Information Theoretical Analysis of Multivariate Correlation.
        IBM Journal of Research and Development, 4(1), Article 1.
        https://doi.org/10.1147/rd.41.0066

        Pascual-Marqui, R. D., Kochi, K., & Kinoshita, T. (2025).
        Total/dual correlation/coherence, redundancy/synergy, complexity, and O-information for real and complex valued multivariate data
        (No. arXiv:2507.08773). arXiv.
        https://doi.org/10.48550/arXiv.2507.08773


    Parameters
    ----------
    cov : np.ndarray
        The covariance matrix.
    inputs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    float
        The expected total correlation.
    """
    if inputs[0] == -1:
        _inputs = tuple(i for i in range(cov.shape[0]))
    else:
        _inputs = inputs

    _cov: np.ndarray = cov[_inputs, :][:, _inputs]

    # Converting to a correlation/coherence matrix.
    diag: np.ndarray = np.sqrt(np.diag(_cov))
    d_inv: np.ndarray = np.diag(1.0 / diag)

    corr = d_inv @ _cov @ d_inv

    return -np.linalg.slogdet(corr)[1] / 2


def local_k_wms(
    k: int, data: np.ndarray, cov: np.ndarray, inputs: tuple = (-1,)
) -> np.ndarray:
    """

    Parameters
    ----------
    k : int
        The integer value that defines whether one is computing
        S-info, DTC, or negative O-information.
    data : np.ndarray
        The data in channels x samples format.
    cov : np.ndarray, optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    inputs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    np.ndarray.
        The series of local k_{wms}.

    """
    if inputs[0] == -1:
        _inputs: tuple = tuple(i for i in range(data.shape[0]))
    else:
        _inputs = inputs

    N: int = len(_inputs)
    whole = (N - k) * local_total_correlation(
        data[_inputs, :], cov[_inputs, :][:, _inputs]
    )

    sum_parts = np.zeros_like(whole)

    for i in range(N):
        idxs = tuple(_inputs[j] for j in range(N) if j != i)
        sum_parts += local_total_correlation(data[idxs, :], cov[idxs, :][:, idxs])

    return whole - sum_parts


def k_wms(k: int, cov: np.ndarray, inputs: tuple = (-1,)) -> float:
    """
    A utility function that computes the generalized form
    of the O-information, S-information, and DTC.

    :math:`K_{WMS}(X) = (N-k)TC(X) - \\sum_{i=1}^{N} TC(X^{-i})`

    Parameters
    ----------
    k : int
        The integer value that defines whether one is computing
        S-info, DTC, or negative O-information.
    cov : np.ndarray
        The covariance matrix that defines the distribution.
    inputs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    float

    """

    if inputs[0] == -1:
        _inputs: tuple = tuple(i for i in range(cov.shape[0]))
    else:
        _inputs = inputs

    N: int = len(_inputs)

    whole: float = (N - k) * total_correlation(cov[_inputs, :][:, _inputs])
    sum_parts: float = 0.0

    for i in range(N):

        idxs: tuple = tuple(_inputs[j] for j in range(N) if j != i)
        sum_parts += total_correlation(cov[idxs, :][:, idxs])

    return whole - sum_parts


def local_s_information(
    data: np.ndarray, cov: np.ndarray = COV_NULL, inputs: tuple = (-1,)
) -> np.ndarray:
    """
    s(x) = N*tc(x) - \sum tc(x^-i)

    Parameters
    ----------
    data : np.ndarray
        The data in channels x samples format.
    cov : np.ndarray, optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    inputs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    np.ndarray.
        The series of local S-information.
    """

    if cov[0][0] == -1:
        cov = np.cov(data, ddof=0)

    return local_k_wms(k=0, data=data, cov=cov, inputs=inputs)


def s_information(cov: np.ndarray, inputs: tuple = (-1,)) -> float:
    """
    See:
        Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
        Quantifying High-order Interdependencies via Multivariate
            Extensions of the Mutual Information.
        Physical Review E, 100(3), Article 3.
        https://doi.org/10.1103/PhysRevE.100.032305

        Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
        Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
        Communications Biology, 6(1), Article 1.
        https://doi.org/10.1038/s42003-023-04843-w

    Parameters
    ----------
    data : np.ndarray
        The data in channels x samples format.
    cov : np.ndarray
        The covariance matrix that defines the distribution.
    inputs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Return:
    -------
    float
        The expected S-information.

    """

    return k_wms(k=0, cov=cov, inputs=inputs)


def local_dual_total_correlation(
    data: np.ndarray, cov: np.ndarray = COV_NULL, inputs: tuple = (-1)
) -> np.ndarray:
    """
    dtc(x) = (N-1)tc(x) + \sum tc(x^-i)

    Parameters
    ----------
    data : np.ndarray
        The data in channels x samples format.
    cov : np.ndarray, optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    inputs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    np.ndarray.
        The series of local dual total correlations.
    """

    if cov[0][0] == -1:
        cov = np.cov(data, ddof=0)

    return local_k_wms(k=1, data=data, cov=cov, inputs=inputs)


def dual_total_correlation(cov: np.ndarray, inputs: tuple = (-1,)) -> float:
    """
    DTC(X) = (N-1)TC(X) + \sum TC(X^-i)

    See:
        Abdallah, S. A., & Plumbley, M. D. (2012).
        A measure of statistical complexity based on predictive
            information with application to finite spin systems.
        Physics Letters A, 376(4), 275–281.
        https://doi.org/10.1016/j.physleta.2011.10.066

        Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
        Quantifying High-order Interdependencies via Multivariate
            Extensions of the Mutual Information.
        Physical Review E, 100(3), Article 3.
        https://doi.org/10.1103/PhysRevE.100.032305

    Parameters
    ----------
    data : np.ndarray
        The data in channels x samples format.
    cov : np.ndarray
        The covariance matrix that defines the distribution. .
    inputs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    float
        The expected dual total correlation.

    """

    return k_wms(k=1, cov=cov, inputs=inputs)


def local_o_information(
    data: np.ndarray, cov: np.ndarray = COV_NULL, inputs: tuple = (-1,)
) -> np.ndarray:
    """

    See:
        Scagliarini, T., Marinazzo, D., Guo, Y., Stramaglia, S., & Rosas, F. E. (2022).
        Quantifying high-order interdependencies on individual patterns via the local O-information: Theory and applications to music analysis.
        Physical Review Research, 4(1), 013184.
        https://doi.org/10.1103/PhysRevResearch.4.013184

        Pope, M., Varley, T. F., Grazia Puxeddu, M., Faskowitz, J., & Sporns, O. (2025).
        Time-varying synergy/redundancy dominance in the human cerebral cortex. Journal of Physics: Complexity, 6(1), 015015.
        https://doi.org/10.1088/2632-072X/adbaa9

    Parameters
    ----------
    data : np.ndarray
        The data in channels x samples format.
    cov : np.ndarray, optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    inputs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    np.ndarray.
        The series of local O-informations.
    """

    return -local_k_wms(k=2, data=data, cov=cov, inputs=inputs)


def o_information(cov: np.ndarray, inputs: tuple = (-1,)) -> float:
    """
    See:
        Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
        Quantifying High-order Interdependencies via Multivariate
        Extensions of the Mutual Information.
        Physical Review E, 100(3), Article 3.
        https://doi.org/10.1103/PhysRevE.100.032305

        Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
        Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
        Communications Biology, 6(1), Article 1.
        https://doi.org/10.1038/s42003-023-04843-w

    Parameters
    ----------
    cov : np.ndarray, optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    inputs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    float
        The expected O-information.

    """

    return -k_wms(k=2, cov=cov, inputs=inputs)


def tse_complexity(num_samples: int, cov: np.ndarray) -> float:
    """
    Computes the Tononi-Sporns-Edelman complexity using Gaussian
    estimators.

    See:
        Tononi, G., Sporns, O., & Edelman, G. M. (1994).
        A measure for brain complexity: Relating functional segregation and
            integration in the nervous system.
        Proceedings of the National Academy of Sciences, 91(11), Article 11.
        https://doi.org/10.1073/pnas.91.11.5033


    Parameters
    ----------
    num_samples : int
        The number of sample subsets to compute.
    cov : np.ndarray
        The covariance matrix that defines the distribution.

    Returns
    -------
    float
        The TSE complexity.
    """

    N: int = cov.shape[0]  # Number of channels

    tc_whole: float = total_correlation(cov)  # Global total correlation

    null_tcs: np.ndarray = np.array(
        [(float(i) / float(N)) * tc_whole for i in range(1, N + 1)]
    )
    exp_tcs: np.ndarray = np.zeros(null_tcs.shape[0])
    exp_tcs[-1] = tc_whole

    for k in range(1, N):

        # All of the samples of subsets at scale i
        samples: np.ndarray = np.array(
            [np.random.choice(N, size=k, replace=False) for _ in range(num_samples)]
        )
        samples.sort(axis=-1)

        # No need to run the same subset multiple times.
        samples = np.unique(samples, axis=0)

        exp_tcs[k - 1] = np.mean(
            [total_correlation(cov, inputs=tuple(sample)) for sample in samples]
        )

    return (null_tcs - exp_tcs).sum()


def description_complexity(cov: np.ndarray, inputs: tuple = (-1,)) -> float:
    """
    C(X) = DTC(X) / N

    Parameters
    ----------
    cov : np.ndarray
        The covariance matrix that defines the distribution.
    inputs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    float
        The expected description complexity.

    """

    if inputs[0] == -1:
        N: float = float(cov.shape[0])
    else:
        N: float = float(len(inputs))

    return dual_total_correlation(cov=cov, inputs=inputs) / N


def local_description_complexity(
    data: np.ndarray, cov: np.ndarray = COV_NULL, inputs: tuple = (-1,)
) -> np.ndarray:
    """
    c(x) = dtc(x) / N

    Parameters
    ----------
    data : np.ndarray
        The data in channels x time format.
    cov : np.ndarray, optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    inputs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    np.ndarray.
        The series of local description complexities.

    """

    if inputs[0] == -1:
        N: float = float(data.shape[0])
    else:
        N: float = float(len(inputs))

    return local_k_wms(k=1, data=data, cov=cov, inputs=inputs) / N

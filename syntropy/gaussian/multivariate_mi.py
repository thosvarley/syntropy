import numpy as np
from .utils import COV_NULL
from .shannon import local_differential_entropy
from numpy.typing import NDArray

def local_total_correlation(
    data: NDArray[np.floating], cov: NDArray[np.floating], idxs: tuple[int, ...] = (-1,)
) -> NDArray[np.floating]:
    """
    The local total correlation. Note that this measure can be negative.

    .. math:: 
        tc(x) = \\sum_{i=1}^{N}h(x_i) - h(x)

    Parameters
    ----------
    data : NDArray[np.floating]
        The data in channels x samples format.
    inptuts: tuple
        The indices of the channels to include.
    
    Returns
    -------
    NDArray[np.floating]
        The local total correaltion for each frame.

    References
    ----------
    Scagliarini, T., Marinazzo, D., Guo, Y., Stramaglia, S., & Rosas, F. E. (2022).
    Quantifying high-order interdependencies on individual patterns via the local O-information: Theory and applications to music analysis.
    Physical Review Research, 4(1), 013184.
    https://doi.org/10.1103/PhysRevResearch.4.013184

    Pope, M., Varley, T. F., Grazia Puxeddu, M., Faskowitz, J., & Sporns, O. (2025). 
    Time-varying synergy/redundancy dominance in the human cerebral cortex. 
    Journal of Physics: Complexity, 6(1), 015015. 
    https://doi.org/10.1088/2632-072X/adbaa9

    """

    assert cov.shape[0] == data.shape[0], (
        "The data and covariance matrix must have the same dimensionality"
    )

    if idxs[0] == -1:
        _idxs = tuple(i for i in range(data.shape[0]))
    else:
        _idxs = idxs
    N = len(_idxs)

    whole = local_differential_entropy(data[_idxs, :], cov[np.ix_(_idxs, _idxs)])

    sum_parts = np.zeros_like(whole)
    for i in range(N):
        sum_parts += local_differential_entropy(data[_idxs[i], :])

    return sum_parts - whole


def total_correlation(
    cov: NDArray[np.floating], idxs: tuple[int, ...] = (-1,)
) -> float:
    """
    The expected total correlation.

    .. math::

        TC(X) &= D_{KL}(P(X) || \\prod_{i=1}^{N}P(X_i) \\\\
              &= \\sum_{i=1}^{N}H(X_i) - H(X)

    For Gaussian random variables, the estimator is:

        .. math::
            \\hat{TC}(X) = \\frac{-\log R}{2}

    Where :math:`R` is the Pearson correlation matrix.

    Parameters
    ----------
    cov : NDArray[np.floating]
        The covariance matrix.
    idxs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    float
        The expected total correlation.
    
    References
    ----------
    Watanabe, S. (1960). Information Theoretical Analysis of Multivariate Correlation.
    IBM Journal of Research and Development, 4(1), Article 1.
    https://doi.org/10.1147/rd.41.0066
    
    Tononi, G., Sporns, O., & Edelman, G. M. (1994). 
    A measure for brain complexity: Relating functional segregation and integration in the nervous system. 
    Proceedings of the National Academy of Sciences, 91(11), Article 11. 
    https://doi.org/10.1073/pnas.91.11.5033

    Pascual-Marqui, R. D., Kochi, K., & Kinoshita, T. (2025).
    Total/dual correlation/coherence, redundancy/synergy, complexity, and O-information for real and complex valued multivariate data
    (No. arXiv:2507.08773). arXiv.
    https://doi.org/10.48550/arXiv.2507.08773

    """
    if idxs[0] == -1:
        _idxs = tuple(i for i in range(cov.shape[0]))
    else:
        _idxs = idxs

    _cov: NDArray[np.floating] = cov[np.ix_(_idxs, _idxs)]

    # Converting to a correlation/coherence matrix.
    diag: NDArray[np.floating] = np.sqrt(np.diag(_cov))
    d_inv: NDArray[np.floating] = np.diag(1.0 / diag)

    corr: NDArray[np.floating] = d_inv @ _cov @ d_inv

    return -np.linalg.slogdet(corr)[1] / 2


def local_k_wms(
    k: int,
    data: NDArray[np.floating],
    cov: NDArray[np.floating],
    idxs: tuple[int, ...] = (-1,),
) -> NDArray[np.floating]:
    """
    A utility function that computes the local generalized form
    of the O-information, S-information, and DTC.

    .. math:: 
        k_{WMS}(x) = (N-k)tc(x) - \\sum_{i=1}^{N} tc(x^{-i})

    Parameters
    ----------
    k : int
        The integer value that defines whether one is computing
        S-info, DTC, or negative O-information.
    data : NDArray[np.floating]
        The data in channels x samples format.
    cov : NDArray[np.floating], optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    idxs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    NDArray[np.floating].
        The series of local k_{wms}.

    """
    if idxs[0] == -1:
        _idxs: tuple[int, ...] = tuple(i for i in range(data.shape[0]))
    else:
        _idxs = idxs

    N: int = len(_idxs)
    whole = (N - k) * local_total_correlation(data[_idxs, :], cov[np.ix_(_idxs, _idxs)])

    sum_parts = np.zeros_like(whole)

    for i in range(N):
        idxs = tuple(_idxs[j] for j in range(N) if j != i)
        sum_parts += local_total_correlation(data[idxs, :], cov[np.ix_(idxs, idxs)])

    return whole - sum_parts


def k_wms(k: int, cov: NDArray[np.floating], idxs: tuple[int, ...] = (-1,)) -> float:
    """
    S-information, DTC, and negative O-information can all be written in a general form:

    .. math::

        WMS^{k}(X) = (N-k)TC(X) - \\sum_{i=1}^{N}TC(X^{-i})

    Parameters
    ----------
    k : int
        The integer value that defines whether one is computing
        S-info, DTC, or negative O-information.
    cov : NDArray[np.floating]
        The covariance matrix that defines the distribution.
    idxs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    float

    """

    if idxs[0] == -1:
        _idxs: tuple[int, ...] = tuple(i for i in range(cov.shape[0]))
    else:
        _idxs = idxs

    N: int = len(_idxs)

    whole: float = (N - k) * total_correlation(cov[np.ix_(_idxs, _idxs)])
    sum_parts: float = 0.0

    for i in range(N):
        idxs_residual: tuple[int, ...] = tuple(_idxs[j] for j in range(N) if j != i)
        sum_parts += total_correlation(cov[idxs_residual, :][:, idxs_residual])

    return whole - sum_parts


def local_s_information(
    data: NDArray[np.floating],
    cov: NDArray[np.floating] = COV_NULL,
    idxs: tuple[int, ...] = (-1,),
) -> NDArray[np.floating]:
    """
    Compute local S-information using Gaussian estimation.
    
    .. math::
        \\sigma(X) &= \\sum_{i=1}^{N}i(x_i;x^{-i}) \\\\
                   &= N\\times tc(x) - \\sum_{i=1}^{N}tc(x^{-i}) \\\\
                   &= tc(x) + dtc(x)

    Parameters
    ----------
    data : NDArray[np.floating]
        The data in channels x samples format.
    cov : NDArray[np.floating], optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    idxs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    NDArray[np.floating].
        The series of local S-information.

    """

    if cov[0][0] == -1:
        cov = np.cov(data, ddof=0)

    return local_k_wms(k=0, data=data, cov=cov, idxs=idxs)


def s_information(cov: NDArray[np.floating], idxs: tuple[int, ...] = (-1,)) -> float:
    """
    Compute S-information using Gaussian estimation.

    .. math:: 

        \\Sigma(X) &= \\sum_{i=1}^{N}I(X_i;X^{-i}) \\\\
                   &= N\\times TC(X) - \\sum_{i=1}^{N}TC(X^{-i}) \\\\
                   &= TC(X) + DTC(X)
    
    Parameters
    ----------
    data : NDArray[np.floating]
        The data in channels x samples format.
    cov : NDArray[np.floating]
        The covariance matrix that defines the distribution.
    idxs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns:
    -------
    float
        The expected S-information.

    References
    ----------
    Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
    Quantifying High-order Interdependencies via Multivariate
    Extensions of the Mutual Information.
    Physical Review E, 100(3), Article 3.
    https://doi.org/10.1103/PhysRevE.100.032305

    Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
    Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
    Communications Biology, 6(1), Article 1.
    https://doi.org/10.1038/s42003-023-04843-w

    """

    return k_wms(k=0, cov=cov, idxs=idxs)


def local_dual_total_correlation(
    data: NDArray[np.floating],
    cov: NDArray[np.floating] = COV_NULL,
    idxs: tuple[int, ...] = (-1,),
) -> NDArray[np.floating]:
    """
    Computes the dual total correlation using Gaussian estimation. Note that this measure can be negative.
    
    .. math:: 

        dtc(x) &= h(x) - \\sum_{i=1}^{N}h(x_i|x^{-i}) \\\\
               &= (N-1)\\times tc(x) - \\sum_{i=1}^{N}tc(x^{-i})

    Parameters
    ----------
    data : NDArray[np.floating]
        The data in channels x samples format.
    cov : NDArray[np.floating], optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    idxs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    NDArray[np.floating].
        The series of local dual total correlations.

    """

    if cov[0][0] == -1:
        cov = np.cov(data, ddof=0)

    return local_k_wms(k=1, data=data, cov=cov, idxs=idxs)


def dual_total_correlation(
    cov: NDArray[np.floating], idxs: tuple[int, ...] = (-1,)
) -> float:
    """
    Computes the dual total correlation using Gaussian estimation.

    .. math:: 

        DTC(X) &= H(X) - \\sum_{i=1}^{N}H(X_i|X^{-i}) \\\\
               &= (N-1)\\times TC(X) - \\sum_{i=1}^{N}TC(X^{-i})
    
    Parameters
    ----------
    data : NDArray[np.floating]
        The data in channels x samples format.
    cov : NDArray[np.floating]
        The covariance matrix that defines the distribution. .
    idxs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    float
        The expected dual total correlation.

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

    return k_wms(k=1, cov=cov, idxs=idxs)


def local_o_information(
    data: NDArray[np.floating],
    cov: NDArray[np.floating] = COV_NULL,
    idxs: tuple[int, ...] = (-1,),
) -> NDArray[np.floating]:
    """
    Computes the local O-information for each sample using Gaussian estimation.

    .. math::

        \\omega(x) &= (2-N)tc(x) + \\sum_{i=1}^{N}tc(x^{-i}) \\\\
                   &= tc(x) - dtc(x)
    
    Parameters
    ----------
    data : NDArray[np.floating]
        The data in channels x samples format.
    cov : NDArray[np.floating], optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    idxs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    NDArray[np.floating].
        The series of local O-informations.

    References
    ----------
    Scagliarini, T., Marinazzo, D., Guo, Y., Stramaglia, S., & Rosas, F. E. (2022).
    Quantifying high-order interdependencies on individual patterns via the local O-information: Theory and applications to music analysis.
    Physical Review Research, 4(1), 013184.
    https://doi.org/10.1103/PhysRevResearch.4.013184

    Pope, M., Varley, T. F., Grazia Puxeddu, M., Faskowitz, J., & Sporns, O. (2025).
    Time-varying synergy/redundancy dominance in the human cerebral cortex. Journal of Physics: Complexity, 6(1), 015015.
    https://doi.org/10.1088/2632-072X/adbaa9

    """

    return -local_k_wms(k=2, data=data, cov=cov, idxs=idxs)


def o_information(cov: NDArray[np.floating], idxs: tuple[int, ...] = (-1,)) -> float:
    """
    Compute O-information using Gaussian estimation.
    O-information quantifies the balance between redundancy (positive values) and synergy (negative values) in multivariate information.
    
    .. math::

        \\Omega(X) &= (2-N)TC(X) + \\sum_{i=1}^{N}TC(X^{-i}) \\\\
                   &= TC(X) - DTC(X)

    Parameters
    ----------
    cov : NDArray[np.floating], optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    idxs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    float
        The expected O-information.

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

    return -k_wms(k=2, cov=cov, idxs=idxs)


def tse_complexity(num_samples: int, cov: NDArray[np.floating]) -> float:
    """
    Computes the Tononi-Sporns-Edelman complexity using Gaussian
    estimators.
    
    .. math::

        TSE(X) &= \\sum_{k=1}^{\\lfloor N/2\\rfloor} \\bigg\\langle I(X^{k}_j;X^{-k}_j) \\bigg\\rangle_{j} \\\\
               &= \\sum_{k=2}^{N}\\bigg[\\bigg(\\frac{k}{N}\\bigg)TC(X) - \\langle TC(X^{k}_{j}) \\rangle_{j}  \\bigg] 

    Runtimes scale very badly with system size (as it requires brute-forcing) all possible bipartitions of the system. If the system is too large, a sub-sampling approach is taken: at each scale, num_samples are drawn from the space of bipartitions.

    Parameters
    ----------
    num_samples : int
        The number of sample subsets to compute.
    cov : NDArray[np.floating]
        The covariance matrix that defines the distribution.

    Returns
    -------
    float
        The TSE complexity.

    References
    ----------
    Tononi, G., Sporns, O., & Edelman, G. M. (1994).
    A measure for brain complexity: Relating functional segregation and integration in the nervous system.
    Proceedings of the National Academy of Sciences, 91(11), Article 11.
    https://doi.org/10.1073/pnas.91.11.5033

    """

    N: int = cov.shape[0]  # Number of channels

    tc_whole: float = total_correlation(cov)  # Global total correlation

    null_tcs: NDArray[np.floating] = np.array(
        [(float(i) / float(N)) * tc_whole for i in range(1, N + 1)]
    )
    exp_tcs: NDArray[np.floating] = np.zeros(null_tcs.shape[0])
    exp_tcs[-1] = tc_whole

    for k in range(1, N):
        # All of the samples of subsets at scale i
        samples: NDArray[np.floating] = np.array(
            [np.random.choice(N, size=k, replace=False) for _ in range(num_samples)]
        )
        samples.sort(axis=-1)

        # No need to run the same subset multiple times.
        samples = np.unique(samples, axis=0)

        exp_tcs[k - 1] = np.mean(
            [total_correlation(cov, idxs=tuple(sample)) for sample in samples]
        )

    return (null_tcs - exp_tcs).sum()


def description_complexity(
    cov: NDArray[np.floating], idxs: tuple[int, ...] = (-1,)
) -> float:
    """
    .. math:: 
        C(X) = \\frac{DTC(X)}{N} 

    Parameters
    ----------
    cov : NDArray[np.floating]
        The covariance matrix that defines the distribution.
    idxs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    float
        The expected description complexity.

    References
    ----------
    Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
    Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
    Communications Biology, 6(1), Article 1.
    https://doi.org/10.1038/s42003-023-04843-w
    
    """

    N: float = float(cov.shape[0]) if idxs[0] == -1 else float(len(idxs))

    return dual_total_correlation(cov=cov, idxs=idxs) / N


def local_description_complexity(
    data: NDArray[np.floating],
    cov: NDArray[np.floating] = COV_NULL,
    idxs: tuple[int, ...] = (-1,),
) -> NDArray[np.floating]:
    """

    Computes the local description complexity for each sample using Gaussian estimation.

    .. math::
        c(x) = \\frac{dtc(x)}/N

    Parameters
    ----------
    data : NDArray[np.floating]
        The data in channels x time format.
    cov : NDArray[np.floating], optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    idxs : tuple, optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    NDArray[np.floating].
        The series of local description complexities.

    """

    N: float = float(cov.shape[0]) if idxs[0] == -1 else float(len(idxs))

    return local_k_wms(k=1, data=data, cov=cov, idxs=idxs) / N

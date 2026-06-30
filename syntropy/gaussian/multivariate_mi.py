from __future__ import annotations

import numpy as np
from .shannon import local_differential_entropy, differential_entropy
from numpy.typing import NDArray
from .utils import check_cov, covariance_to_correlation
from ..utils import check_idxs, make_powerset
from .decompositions import local_precompute_sources


def local_total_correlation(
    data: NDArray[np.floating],
    cov: NDArray[np.floating] | None = None,
    idxs: tuple[int, ...] | None = None,
) -> NDArray[np.floating]:
    """
    The local total correlation. Note that this measure can be negative.

    .. math::
        tc(x) = \\sum_{i=1}^{N}h(x_i) - h(x)

    If you wish to use a Gaussian copula estimator, use the transformed data and the correlation matrix returned by the function :func:`utils.copula_transform`.

    Parameters
    ----------
    data : NDArray[np.floating]
        The data in channels x samples format.
    cov : NDArray[np.floating] | None
        The covariance matrix used to compute the local entropies.
        If None is provided, it is inferred from the data.
        The default is None.
    inputs: tuple | None
        The indices of the channels to include.
        If None is provided, all channels are used.
        The default is None.

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

    N: int = len(idxs) if idxs is not None else data.shape[0]
    whole = local_differential_entropy(data=data, cov=cov, idxs=idxs)

    sum_parts = np.zeros_like(whole)
    for i in range(N):
        idx = idxs[i] if idxs is not None else i
        sum_parts += local_differential_entropy(data[idx, :])

    return sum_parts - whole


def total_correlation(
    cov: NDArray[np.floating], idxs: tuple[int, ...] | None = None
) -> float:
    r"""
    The expected total correlation.

    .. math::

        TC(X) &= D_{KL}(P(X) || \prod_{i=1}^{N}P(X_i) \\
              &= \sum_{i=1}^{N}H(X_i) - H(X)

    For Gaussian random variables, the estimator is:

        .. math::
            \hat{TC}(X) = \frac{-\log R}{2}

    Where :math:`R` is the Pearson correlation matrix.
    
    If you wish to use a Gaussian copula estimator, use the correlation matrix returned by the function :func:`utils.copula_transform`. 

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
    idxs_: tuple[int, ...] = check_idxs(idxs, cov)

    cov_: NDArray[np.floating] = cov[np.ix_(idxs_, idxs_)]
    corr: NDArray[np.floating]

    # Checking to see if the covariance matrix has
    # 1s along the diagonal (a correlation matrix).
    if False in np.isclose(cov_.diagonal(), 1):
        # Converting to a correlation/coherence matrix.
        corr = covariance_to_correlation(cov)
    else:
        corr = cov_.copy()

    return -np.linalg.slogdet(corr)[1] / 2


def local_delta_k(
    k: int,
    data: NDArray[np.floating],
    cov: NDArray[np.floating] | None = None,
    idxs: tuple[int, ...] | None = None,
) -> NDArray[np.floating]:
    r"""
    A utility function that computes the local generalized form
    of the O-information, S-information, and DTC.

    .. math::
        \delta_{k}(x) = (N-k)tc(x) - \sum_{i=1}^{N} tc(x^{-i})

    If you wish to use a Gaussian copula estimator, use the transformed data and the correlation matrix returned by the function :func:`utils.copula_transform`.

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
        The series of local :math:`\delta^{k}`.

    """
    idxs_: tuple[int, ...] = check_idxs(idxs, data)
    cov_: NDArray[np.floating] = check_cov(cov, data)

    N: int = len(idxs_)
    whole = (N - k) * local_total_correlation(
        data[idxs_, :], cov_[np.ix_(idxs_, idxs_)]
    )

    sum_parts = np.zeros_like(whole)

    for i in range(N):
        idxs_r = tuple(idxs_[j] for j in range(N) if j != i)
        sum_parts += local_total_correlation(
            data[idxs_r, :], cov_[np.ix_(idxs_r, idxs_r)]
        )

    return whole - sum_parts


def delta_k(
    k: int, cov: NDArray[np.floating], idxs: tuple[int, ...] | None = None
) -> float:
    """
    S-information, DTC, and negative O-information can all be written in a general form:

    .. math::

        WMS^{k}(X) = (N-k)TC(X) - \\sum_{i=1}^{N}TC(X^{-i})

    If you wish to use a Gaussian copula estimator, use the correlation matrix returned by the function :func:`utils.copula_transform`.

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

    idxs_: tuple[int, ...] = check_idxs(idxs, cov)
    N: int = len(idxs_)

    whole: float = (N - k) * total_correlation(cov[np.ix_(idxs_, idxs_)])
    sum_parts: float = 0.0

    for i in range(N):
        idxs_residual: tuple[int, ...] = tuple(idxs_[j] for j in range(N) if j != i)
        sum_parts += total_correlation(cov[idxs_residual, :][:, idxs_residual])

    return whole - sum_parts


def local_s_information(
    data: NDArray[np.floating],
    cov: NDArray[np.floating] | None = None,
    idxs: tuple[int, ...] | None = None,
) -> NDArray[np.floating]:
    """
    Compute local S-information using Gaussian estimation.
    
    .. math::
        \\sigma(X) &= \\sum_{i=1}^{N}i(x_i;x^{-i}) \\\\
                   &= N\\times tc(x) - \\sum_{i=1}^{N}tc(x^{-i}) \\\\
                   &= tc(x) + dtc(x)
    
    If you wish to use a Gaussian copula estimator, use the transformed data and the correlation matrix returned by the function :func:`utils.copula_transform`. 

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

    return local_delta_k(k=0, data=data, cov=cov, idxs=idxs)


def s_information(
    cov: NDArray[np.floating], idxs: tuple[int, ...] | None = None
) -> float:
    """
    Compute S-information using Gaussian estimation.

    .. math:: 

        \\Sigma(X) &= \\sum_{i=1}^{N}I(X_i;X^{-i}) \\\\
                   &= N\\times TC(X) - \\sum_{i=1}^{N}TC(X^{-i}) \\\\
                   &= TC(X) + DTC(X)
    
    If you wish to use a Gaussian copula estimator, use the correlation matrix returned by the function :func:`utils.copula_transform`. 
    
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

    return delta_k(k=0, cov=cov, idxs=idxs)


def local_dual_total_correlation(
    data: NDArray[np.floating],
    cov: NDArray[np.floating] | None = None,
    idxs: tuple[int, ...] | None = None,
) -> NDArray[np.floating]:
    """
    Computes the dual total correlation using Gaussian estimation. Note that this measure can be negative.
    
    .. math:: 

        dtc(x) &= h(x) - \\sum_{i=1}^{N}h(x_i|x^{-i}) \\\\
               &= (N-1)\\times tc(x) - \\sum_{i=1}^{N}tc(x^{-i})
    
    If you wish to use a Gaussian copula estimator, use the transformed data and the correlation matrix returned by the function :func:`utils.copula_transform`. 

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

    return local_delta_k(k=1, data=data, cov=cov, idxs=idxs)


def dual_total_correlation(
    cov: NDArray[np.floating], idxs: tuple[int, ...] | None = None
) -> float:
    r"""
    Computes the dual total correlation using Gaussian estimation.

    .. math::

        DTC(X) &= H(X) - \sum_{i=1}^{N}H(X_i|X^{-i}) \\
               &= (N-1)\times TC(X) - \sum_{i=1}^{N}TC(X^{-i})

    If you wish to use a Gaussian copula estimator, use the correlation matrix returned by the function :func:`utils.copula_transform`.

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

    return delta_k(k=1, cov=cov, idxs=idxs)


def local_o_information(
    data: NDArray[np.floating],
    cov: NDArray[np.floating] | None = None,
    idxs: tuple[int, ...] | None = None,
) -> NDArray[np.floating]:
    """
    Computes the local O-information for each sample using Gaussian estimation.

    .. math::

        \\omega(x) &= (2-N)tc(x) + \\sum_{i=1}^{N}tc(x^{-i}) \\\\
                   &= tc(x) - dtc(x)
    
    If you wish to use a Gaussian copula estimator, use the transformed data and the correlation matrix returned by the function :func:`utils.copula_transform`. 
    
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

    return -local_delta_k(k=2, data=data, cov=cov, idxs=idxs)


def o_information(
    cov: NDArray[np.floating], idxs: tuple[int, ...] | None = None
) -> float:
    """
    Compute O-information using Gaussian estimation.
    O-information quantifies the balance between redundancy (positive values) and synergy (negative values) in multivariate information.
    
    .. math::

        \\Omega(X) &= (2-N)TC(X) + \\sum_{i=1}^{N}TC(X^{-i}) \\\\
                   &= TC(X) - DTC(X)
    
    If you wish to use a Gaussian copula estimator, use the correlation matrix returned by the function :func:`utils.copula_transform`. 

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

    return -delta_k(k=2, cov=cov, idxs=idxs)


def tse_complexity(num_samples: int, cov: NDArray[np.floating]) -> float:
    """
    Computes the Tononi-Sporns-Edelman complexity using Gaussian
    estimators.
    
    .. math::

        TSE(X) &= \\sum_{k=1}^{\\lfloor N/2\\rfloor} \\bigg\\langle I(X^{k}_j;X^{-k}_j) \\bigg\\rangle_{j} \\\\
               &= \\sum_{k=2}^{N}\\bigg[\\bigg(\\frac{k}{N}\\bigg)TC(X) - \\langle TC(X^{k}_{j}) \\rangle_{j}  \\bigg] 

    Runtimes scale very badly with system size (as it requires brute-forcing) all possible bipartitions of the system. If the system is too large, a sub-sampling approach is taken: at each scale, num_samples are drawn from the space of bipartitions.
    
    If you wish to use a Gaussian copula estimator, use the correlation matrix returned by the function :func:`utils.copula_transform`. 

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
    cov: NDArray[np.floating], idxs: tuple[int, ...] | None = None
) -> float:
    """
    .. math::
        C(X) = \\frac{DTC(X)}{N}

    If you wish to use a Gaussian copula estimator, use the correlation matrix returned by the function :func:`utils.copula_transform`.

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

    N: float = float(cov.shape[0]) if idxs[0] is None else float(len(idxs))

    return dual_total_correlation(cov=cov, idxs=idxs) / N


def local_description_complexity(
    data: NDArray[np.floating],
    cov: NDArray[np.floating] | None = None,
    idxs: tuple[int, ...] | None = None,
) -> NDArray[np.floating]:
    """

    Computes the local description complexity for each sample using Gaussian estimation.

    .. math::
        c(x) = \\frac{dtc(x)}/N

    If you wish to use a Gaussian copula estimator, use the transformed data and the correlation matrix returned by the function :func:`utils.copula_transform`.

    Parameters
    ----------
    data : NDArray[np.floating]
        The data in channels x time format.
    cov : NDArray[np.floating], optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    idxs : tuple, optional
        The specific subset of variables to compute the descriptio complexity of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    NDArray[np.floating].
        The series of local description complexities.

    """

    N: float = float(cov.shape[0]) if idxs[0] is None else float(len(idxs))

    return local_delta_k(k=1, data=data, cov=cov, idxs=idxs) / N


def local_co_information(
    data: NDArray[np.floating],
    cov: NDArray[np.floating] | None = None,
    idxs: tuple[int, ...] | None = None,
) -> NDArray[np.floating]:
    """
    Computes the local co-information, the third generalization of bivariate mutual information. Unlike total correlation and dual total correlation, the cO-information can be negative and is difficult to interpret.

    .. math::
        co(X) = \\sum_{\\xi\\subseteq X}(-1)^{|\\xi|}h(\\xi)


    Parameters
    ----------
    data : NDArray[np.floating]
        The data in channels x time format.
    cov : NDArray[np.floating] | None, optional
        The covariance matrix that defines the distribution.
        If unspecified it is computed directly from the data.
    idxs : tuple | None, optional
        The specific subset of variables to compute the co-information of.
        Defaults to computing the measure for the entire covariance matrix.


    Returns
    -------
    NDArray[np.floating]
        The local co-informations.

    """

    cov_: NDArray[np.floating] = check_cov(cov=cov, data=data)
    idxs_: tuple[int, ...] = check_idxs(idxs=idxs, data=data)

    sources: dict[tuple[int, ...], NDArray[np.floating]] = local_precompute_sources(
        data=data[idxs_, :], cov=cov_[np.ix_(idxs_, idxs_)]
    )

    co: NDArray[np.floating] = np.zeros((1, data.shape[1]))

    for source in sources.keys():
        sign: int = (-1) ** len(source)
        co -= sign * sources[source]

    return co


def co_information(cov: NDArray[np.floating], idxs: tuple[int, ...] | None) -> float:
    """
    Computes the average co-information, the third generalization of bivariate mutual information. Unlike total correlation and dual total correlation, the cO-information can be negative and is difficult to interpret.

    .. math::
        Co(X) = \\sum_{\\xi\\subseteq X}(-1)^{|\\xi|}H(\\xi)


    Parameters
    ----------
    cov : NDArray[np.floating]
        The covariance matrix that defines the distribution.
    idxs : tuple | None, optional
        The specific subset of variables to compute the co-information of.
        Defaults to computing the measure for the entire covariance matrix.


    Returns
    -------
    float
        The expected co-information.

    """

    idxs_: tuple[int, ...] = check_idxs(idxs, cov)
    sources: list[tuple[int, ...]] = list(make_powerset(idxs_))
    sources.remove(())

    co: float = 0.0

    for source in sources:
        sign: int = (-1) ** len(source)

        co -= sign * differential_entropy(cov[np.ix_(source, source)])

    return co

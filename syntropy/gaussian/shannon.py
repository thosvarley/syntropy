import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray
from .utils import check_cov

H_SINGLE: float = np.log(np.sqrt(2.0 * np.pi * np.e))
LN_TWO_PI_E: float = np.log(2.0 * np.pi * np.e)
TWO_PI: float = 2.0 * np.pi
SQRT_TWO_PI: float = np.sqrt(TWO_PI)

COV_NULL: NDArray[np.floating] = np.array([[-1.0]])


def differential_entropy(
    cov: NDArray[np.floating], idxs: tuple[int, ...] = (-1,)
) -> float:
    """
    Computes the expected differential entropy of a multivariate
    distribution parameterized by a covariance matrix using a
    Gaussian estimator.

    The differential entropy is given by:

    .. math::

        H(X) = \\int dx P(x)\\log P(x)

    And if :math:`X` is drawn from a k-dimensional Gaussian, it is equal to

    .. math::

        H(X) = \\frac{k}{2}\\log 2\\pi\\textnormal{e} + \\frac{1}{2}\\log|\\Sigma|

    Where :math:`|\\Sigma|` is the determinant of the covariance matrix.

    Parameters
    ----------
    cov : NDArray[np.floating]
        The covariance matrix that defines the distribution.
    idxs : tuple[int, ...], optional
        The specific subset of variables to compute the total correlation of.
        Defaults to computing the TC of the entire covariance matrix.

    Returns
    -------
    float

    """

    if idxs[0] == -1:
        return stats.multivariate_normal(cov=cov, allow_singular=True).entropy()
    else:
        if len(idxs) == 1:
            return H_SINGLE
        else:
            return stats.multivariate_normal(
                cov=cov[np.ix_(idxs, idxs)], allow_singular=True
            ).entropy()


def local_differential_entropy(
    data: NDArray[np.floating], cov: NDArray[np.floating] = COV_NULL
) -> NDArray[np.floating]:
    """
    Computes the framewise differential entropy for a set of variables.

    .. math::
        h(x) = -\\log P(x)

    For data drawn from a k-dimensional Gaussian

    .. math::
        P(x) = (2\\pi)^{-k/2}|\\Sigma|^{-1/2}\\textnormal{e}^{\\frac{-(x - \\mu)^\\mathrm{T} \\Sigma^{-1}(x - \\mu)}{2}}

    Parameters
    ----------
    data : NDArray[np.floating]
        The data in channels x samples format.
    cov : NDArray[np.floating], optional
        The covariance matrix that defines the distribution.
        If none is provided, it is computed from the data object.

    Returns
    -------
    NDArray[np.floating]
        The series of pointwise entropies.

    """
    N: int = data.shape[0]

    cov_: NDArray[np.floating] = check_cov(cov, data)

    if N == 1:
        return -stats.norm.logpdf(
            x=data, loc=data.mean(), scale=data.std()
        )
    else:
        return -(
            stats.multivariate_normal.logpdf(
                x=data.T, mean=data.mean(axis=-1), cov=cov_, allow_singular=True
            )
        )


def conditional_entropy(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    cov: NDArray[np.floating],
) -> float:
    """
    Computes the conditional entropy of X given Y using Gaussian estimation.
    
    .. math::
        H(X|Y) = H(X,Y) - H(Y)
    
    Parameters
    ----------
    idxs_x : tuple
        The indices of the variables to compute the conditional entropy on.
    idxs_y : tuple
        The indices of the conditioning set.
    cov : NDArray[np.floating], optional
        The covariance matrix that defines the distribution.
        If none is provided, it is computed from the data object.

    Returns
    -------
    float

    """

    joint: tuple[int, ...] = idxs_x + idxs_y

    h_joint: float = differential_entropy(cov, joint)
    h_y: float = differential_entropy(cov, idxs_y)

    return h_joint - h_y


def local_conditional_entropy(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    data: NDArray[np.floating],
    cov: NDArray[np.floating] = COV_NULL,
) -> NDArray[np.floating]:
    """
    Computes the local condition entropy for every sample in data using Gaussian estimation.

    .. math::
    
        h(x|y) = h(x,y) - h(y)
    

    Parameters
    ----------
    idxs_x : tuple
        The indices of the variables to compute the conditional entropy on.
    idxs_y : tuple
        The indices of the conditioning set.
    data : NDArray[np.floating]
        The data in channels x samples format.
    cov : NDArray[np.floating], optional
        The covariance matrix that defines the distribution.
        If none is provided, it is computed from the data object.

    Returns
    -------
    NDArray[np.floating]

    """

    cov_: NDArray[np.floating] = check_cov(cov, data)

    joint: tuple[int, ...] = idxs_x + idxs_y

    h_y = local_differential_entropy(data[idxs_y, :], cov_[np.ix_(idxs_y, idxs_y)])
    h_joint = local_differential_entropy(data[joint, :], cov_[np.ix_(joint, joint)])

    return h_joint - h_y


def mutual_information(
    idxs_x: tuple[int, ...], idxs_y: tuple[int, ...], cov: NDArray[np.floating]
) -> float:
    r"""
    Computes the mutual information between two (potentially multivariate) sets of elements.

    .. math::
        I(X;Y) &= H(X) + H(Y) - H(X,Y) \\
               &= H(X) - H(X|Y) \\
               &= H(Y) - H(Y|X) \\
               &= H(X,Y) - H(X|Y) - H(Y|X)

    For Gaussian random variables:

    .. math::
        I(X;Y) = \frac{1}{2}\log\frac{|\Sigma_{X}||\Sigma_{Y}|}{|\Sigma_{XY}|}

    In the particular case where :math:`X` and :math:`Y` are univariate, the mutual information can be computed directly from the Pearson correlation coefficient :math:`r`:

    .. math::
        I(X;Y) = \frac{-\log(1-r^{2})}{2}
    
    If you wish to use a Gaussian copula estimator, use the correlation matrix returned by the function :func:`utils.copula_transform`. 


    Parameters
    ----------
    idxs_x : tuple
        The indices of the source variables. Can be multivariate.
    idxs_y : tuple
        The indices of the idxs_y variable. Can be multivariate.
    cov : NDArray[np.floating]
        The covariance matrix that defines the distribution.

    Returns
    -------
    float

    """
    joint: tuple[int, ...] = idxs_x + idxs_y

    cov_idxs_x: NDArray[np.floating] = cov[np.ix_(idxs_x, idxs_x)]
    cov_idxs_y: NDArray[np.floating] = cov[np.ix_(idxs_y, idxs_y)]
    cov_joint: NDArray[np.floating] = cov[np.ix_(joint, joint)]

    det_idxs_x: float = 0.0
    if len(idxs_x) == 1:
        det_idxs_x += np.linalg.det([[cov_idxs_x]])
    else:
        det_idxs_x += np.linalg.det(cov_idxs_x)

    det_idxs_y: float = 0.0
    if len(idxs_y) == 1:
        det_idxs_y += np.linalg.det([[cov_idxs_y]])
    else:
        det_idxs_y += np.linalg.det(cov_idxs_y)

    det_joint: float = np.linalg.det(cov_joint)

    dets: float = (det_idxs_x * det_idxs_y) / det_joint

    return (np.log(dets) / 2).item()


def local_mutual_information(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    data: NDArray[np.floating],
    cov: NDArray[np.floating] = COV_NULL,
) -> NDArray[np.floating]:
    """
    Computes the local mutual information between X and Y for every sample in data using Gaussian estimation.
    Note that the local mutual information can be negative.
    
    .. math::
        i(x;y) &= h(x) + h(y) - h(x,y) \\\\
               &= \\log\\frac{p(x|y)}{p(x)} \\\\
               &= \\log\\frac{p(y|x)}{p(y)} \\\\
               &= \\log\\frac{p(x,y)}{p(x)p(y)} \\\\
    
    If you wish to use a Gaussian copula estimator, use the transformed data and the correlation matrix returned by the function :func:`utils.copula_transform`. 

    Parameters
    ----------
    idxs_x : tuple
        The indices of the source variables. Can be multivariate.
    idxs_y : tuple
        The indices of the idxs_y variable. Can be multivariate.
    data : NDArray[np.floating]
        The data in channels x samples format.
    cov : NDArray[np.floating], optional
        The covariance matrix that defines the distribution.
        If none is provided, it is computed from the data object.

    Returns
    -------
     NDArray[np.floating]

    """

    cov_: NDArray[np.floating] = check_cov(cov, data)

    joint: tuple[int, ...] = idxs_x + idxs_y

    h_x: NDArray[np.floating] = local_differential_entropy(
        data = data[idxs_x, :], cov = cov_[np.ix_(idxs_x, idxs_x)]
    )
    h_y: NDArray[np.floating] = local_differential_entropy(
        data = data[idxs_y, :], cov = cov_[np.ix_(idxs_y, idxs_y)]
    )
    h_joint: NDArray[np.floating] = local_differential_entropy(
        data = data[joint, :], cov = cov_[np.ix_(joint, joint)]
    )

    return h_x + h_y - h_joint


def conditional_mutual_information(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    idxs_z: tuple[int, ...],
    cov: NDArray[np.floating],
) -> float:
    """
    Computes the expected mutual information between a set of variables X
    and Y, conditional on a third set Z.
    
    .. math:: 
        I(X,Y|Z) &= H(X|Z) + H(Y|Z) - H(X,Y|Z) \\\\
                 &= I(X;Y,Z) - I(X;Z)
    
    If you wish to use a Gaussian copula estimator, use the correlation matrix returned by the function :func:`utils.copula_transform`. 

    Parameters
    ----------
    idxs_x : tuple
        The indices of the X variables. Can be multivariate.
    idxs_y : tuple
        The indices of the Y variable. Can be multivariate.
    idxs_z : tuple
        The indices of the conditioning set. Can be multivariate.
    cov : NDArray[np.floating]
        The covariance matrix that defines the distribution.

    Returns
    -------
    float

    """

    joint: tuple[int, ...] = idxs_x + idxs_y

    h_x_z: float = conditional_entropy(idxs_x, idxs_z, cov)
    h_y_z: float = conditional_entropy(idxs_y, idxs_z, cov)
    h_xy_z: float = conditional_entropy(joint, idxs_z, cov)

    return h_x_z + h_y_z - h_xy_z


def local_conditional_mutual_information(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    idxs_z: tuple[int, ...],
    data: NDArray[np.floating],
    cov: NDArray[np.floating] = COV_NULL,
):
    """
    Computes the local conditional mutual information between
    two sets of variables X and Y, conditional on another set Z.
    
    .. math:: 
    
        i(x,y|z) &= h(x|z) + h(y|z) - h(x,y|z) \\\\
                 &= i(x;y,z) - i(x;z)
    
    If you wish to use a Gaussian copula estimator, use the transformed data and the correlation matrix returned by the function :func:`utils.copula_transform`. 

    Parameters
    ----------
    idxs_x : tuple
        The indices of the X variables. Can be multivariate.
    idxs_y : tuple
        The indices of the Y variable. Can be multivariate.
    idxs_z : tuple
        The indices of the conditioning set. Can be multivariate.
    data : NDArray[np.floating]
         The data in channels x samples format.
    cov : NDArray[np.floating], optional
        The covariance matrix that defines the distribution.
        If none is provided, it is computed from the data object.

    Returns
    -------
    NDArray[np.floating]

    """
    cov_: NDArray[np.floating] = check_cov(cov, data)

    joint: tuple[int, ...] = idxs_x + idxs_y

    h_x_z = local_conditional_entropy(idxs_x, idxs_z, data=data, cov=cov_)
    h_y_z = local_conditional_entropy(idxs_y, idxs_z, data=data, cov=cov_)
    h_xy_z = local_conditional_entropy(joint, idxs_z, data=data, cov=cov_)

    return h_x_z + h_y_z - h_xy_z


def kullback_leibler_divergence(
    cov_posterior: NDArray[np.floating], cov_prior: NDArray[np.floating]
) -> float:
    """
    Computes the Gaussian Kullback-Leibler divergence between two :math:`k`-dimensional multivariate Gaussians parameterized by covariance matrices.

    .. math::

        D_{KL}(\\mathcal{N}_0 || \\mathcal{N}_1) = \\frac{1}{2}[ \\operatorname{tr}(\\Sigma_{1}^{-1}\\Sigma_{0}) - k + (\\mu_1 - \\mu_0)^\\mathsf{T} \\Sigma_{1}^{-1}(\\mu_1 - \\mu_0) + \\log\\frac{|\\Sigma_{1}|}{|\\Sigma_{0}|}]

    Parameters
    ----------
    cov_posterior : NDArray[np.floating]
        The covariance matrix that defines the posterior distribution.
    cov_prior : NDArray[np.floating]
        The covariance matrix that defines the prior distribution .

    Returns
    -------
    float

    """
    N: int = cov_prior.shape[0]  # Dimensionality

    assert N == cov_posterior.shape[0], "The covariance matrices must be the same size"

    inv_prior = np.linalg.inv(cov_prior)  # Inverse of Sigma2
    trace_term = np.trace(inv_prior @ cov_posterior)  # tr(Sigma2^{-1} Sigma1)
    log_det_term = (
        np.linalg.slogdet(cov_prior)[1] - np.linalg.slogdet(cov_posterior)[1]
    )  # log(det(Sigma2)/det(Sigma1))

    return 0.5 * (trace_term - N + log_det_term)


def local_kullback_leibler_divergence(
    cov_posterior: NDArray[np.floating],
    cov_prior: NDArray[np.floating],
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Computes the local Kullback-Leibler divergence between the
    posterior and the prior for every sample in the data.
    The local KL divergence is a rarely used measure.
    
    .. math::

        d_{kl}^{P||Q}(x) = h^{Q}(x) - h^{P}(x)
    
    Parameters
    ----------
    cov_posterior : NDArray[np.floating]
        The covariance matrix that defines the posterior distribution.
    cov_prior : NDArray[np.floating]
        The covariance matrix that defines the prior distribution .
    data : NDArray[np.floating]
        The data, assumed to be in channels x samples format.

    Returns
    -------
    NDArray[np.floating]
        The local Kullback-Leibler divergence.

    References
    ----------
    Varley, T. F. (2024).
    Generalized decomposition of multivariate information.
    PLOS ONE, 19(2), e0297128.
    https://doi.org/10.1371/journal.pone.0297128

    """

    h_posterior: NDArray[np.floating] = local_differential_entropy(data, cov_posterior)
    h_prior: NDArray[np.floating] = local_differential_entropy(data, cov_prior)

    return h_prior - h_posterior

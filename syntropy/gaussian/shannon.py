import numpy as np
import scipy.stats as stats

H_SINGLE: float = np.log(np.sqrt(2.0 * np.pi * np.e))
LN_TWO_PI_E: float = np.log(2.0 * np.pi * np.e)
TWO_PI: float = 2.0 * np.pi
SQRT_TWO_PI: float = np.sqrt(TWO_PI)

COV_NULL = np.array([[-1]])


def differential_entropy(cov: np.ndarray, inputs: tuple = (-1,)) -> float:
    """
    Computes the expected differential entropy of a multivariate
    distribution parameterized by a covariance matrix using a
    Gaussian estimator.

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

    """

    if inputs[0] == -1:
        return stats.multivariate_normal(cov=cov).entropy()
    else:
        if len(inputs) == 1:
            return H_SINGLE
        else:
            return stats.multivariate_normal(cov=cov[inputs, :][:, inputs]).entropy()


def local_differential_entropy(
    data: np.ndarray, cov: np.ndarray = COV_NULL
) -> np.ndarray:
    """
    Computes the framewise differential entropy for a set of variables.

    Parameters
    ----------
    data : np.ndarray
        The data in channels x samples format.
    cov : np.ndarray, optional
        The covariance matrix that defines the distribution.
        If none is provided, it is computed from the data object.

    Returns
    -------
    np.ndarray
        The series of pointwise entropies.

    """
    N: int = data.shape[0]

    if cov[0][0] == -1:
        cov = 1 * np.cov(data, ddof=0.0)
    else:
        cov = 1 * cov

    if N == 1:
        return -np.log((1.0 / (SQRT_TWO_PI)) * np.exp(-0.5 * (data) ** 2.0))
    else:
        return -(
            stats.multivariate_normal.logpdf(x=data.T, mean=data.mean(axis=-1), cov=cov)
        )


def conditional_entropy(
    idxs_x: tuple, idxs_y: tuple, cov: np.ndarray = COV_NULL
) -> float:
    """
    Computes the conditional entropy of X given Y.

    Parameters
    ----------
    idxs_x : tuple
        The indices of the variables to compute the conditional entropy on.
    idxs_y : tuple
        The indices of the conditioning set.
    cov : np.ndarray, optional
        The covariance matrix that defines the distribution.
        If none is provided, it is computed from the data object.

    Returns
    -------
    float

    """

    joint = idxs_x + idxs_y

    h_joint = differential_entropy(cov, joint)
    h_y = differential_entropy(cov, idxs_y)

    return h_joint - h_y


def local_conditional_entropy(
    idxs_x: tuple, idxs_y: tuple, data: np.ndarray, cov: np.ndarray = COV_NULL
) -> np.ndarray:
    """
    Computes the local condition entropy for every sample in data.

    Parameters
    ----------
    idxs_x : tuple
        The indices of the variables to compute the conditional entropy on.
    idxs_y : tuple
        The indices of the conditioning set.
    data : np.ndarray
        The data in channels x samples format.
    cov : np.ndarray, optional
        The covariance matrix that defines the distribution.
        If none is provided, it is computed from the data object.

    Returns
    -------
    np.ndarray
    """

    joint = idxs_x + idxs_y

    h_y = local_differential_entropy(data[idxs_y, :], cov[idxs_y, :][:, idxs_y])
    h_joint = local_differential_entropy(data[joint, :], cov[joint, :][:, joint])

    return h_joint - h_y


def mutual_information(idxs_x: tuple, idxs_y: tuple, cov: np.ndarray) -> float:
    """
    Computes the mutual information between the idxs_x and the idxs_y.
    Note that the mutual information is symmetric in its arguments.

    Parameters
    ----------
    idxs_x : tuple
        The indices of the source variables. Can be multivariate.
    idxs_y : tuple
        The indices of the idxs_y variable. Can be multivariate.
    cov : np.ndarray
        The covariance matrix that defines the distribution.

    Returns
    -------
    float

    """
    joint = idxs_x + idxs_y

    cov_idxs_x = cov[idxs_x, :][:, idxs_x]
    cov_idxs_y = cov[idxs_y, :][:, idxs_y]
    cov_joint = cov[joint, :][:, joint]

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
    idxs_x: tuple, idxs_y: tuple, data: np.ndarray, cov: np.ndarray = COV_NULL
) -> np.ndarray:
    """
    Computes the local mutual information between X and Y for every sample in data.
    Note that the local mutual information can be negative.

    Parameters
    ----------
    idxs_x : tuple
        The indices of the source variables. Can be multivariate.
    idxs_y : tuple
        The indices of the idxs_y variable. Can be multivariate.
    data : np.ndarray
        The data in channels x samples format.
    cov : np.ndarray, optional
        The covariance matrix that defines the distribution.
        If none is provided, it is computed from the data object.

    Returns
    -------
     np.ndarray

    """

    if cov[0][0] == -1:
        cov = np.cov(data, ddof=0.0)

    joint = idxs_x + idxs_y

    h_x = local_differential_entropy(data[idxs_x, :], cov[idxs_x, :][:, idxs_x])
    h_y = local_differential_entropy(data[idxs_y, :], cov[idxs_y, :][:, idxs_y])
    h_joint = local_differential_entropy(data[joint, :], cov[joint, :][:, joint])

    return h_x + h_y - h_joint


def conditional_mutual_information(
    idxs_x: tuple, idxs_y: tuple, idxs_z: tuple, cov: np.ndarray
) -> float:
    """
    Computes the expected mutual information between a set of variables X
    and Y, conditional on a third set Z.

    Parameters
    ----------
    idxs_x : tuple
        The indices of the X variables. Can be multivariate.
    idxs_y : tuple
        The indices of the Y variable. Can be multivariate.
    idxs_z : tuple
        The indices of the conditioning set. Can be multivariate.
    cov : np.ndarray
        The covariance matrix that defines the distribution.

    Returns
    -------
    float

    """

    joint: tuple = idxs_x + idxs_y

    h_x_z = conditional_entropy(idxs_x, idxs_z, cov)
    h_y_z = conditional_entropy(idxs_y, idxs_z, cov)
    h_xy_z = conditional_entropy(joint, idxs_z, cov)

    return h_x_z + h_y_z - h_xy_z


def local_conditional_mutual_information(
    idxs_x: tuple,
    idxs_y: tuple,
    idxs_z: tuple,
    data: np.ndarray,
    cov: np.ndarray = COV_NULL,
):
    """
    Computes the local conditional mutual information between
    two sets of variables X and Y, conditional on another set Z.

    Returns a numpy array with one value for every sample.

    Parameters
    ----------
    idxs_x : tuple
        The indices of the X variables. Can be multivariate.
    idxs_y : tuple
        The indices of the Y variable. Can be multivariate.
    idxs_z : tuple
        The indices of the conditioning set. Can be multivariate.
    data : np.ndarray
         The data in channels x samples format.
    cov : np.ndarray, optional
        The covariance matrix that defines the distribution.
        If none is provided, it is computed from the data object.

    Returns
    -------
    np.ndarray
    """
    if cov[0][0] == -1:
        cov = np.cov(data, ddof=0.0)

    joint = idxs_x + idxs_y

    h_x_z = local_conditional_entropy(idxs_x, idxs_z, data, cov)
    h_y_z = local_conditional_entropy(idxs_y, idxs_z, data, cov)
    h_xy_z = local_conditional_entropy(joint, idxs_z, data, cov)

    return h_x_z + h_y_z - h_xy_z


def kullback_leibler_divergence(
    cov_posterior: np.ndarray, cov_prior: np.ndarray
) -> float:
    """
    Computes the Gaussian Kullback-Leibler divergence between
    two multivariate Gaussians parameterized by covariance matrices.

    Parameters
    ----------
    cov_posterior : np.ndarray
        The covariance matrix that defines the posterior distribution.
    cov_prior : np.ndarray
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
    cov_posterior: np.ndarray, cov_prior: np.ndarray, data: np.ndarray
) -> np.ndarray:
    """
    Computes the local Kullback-Leibler divergence between the
    posterior and the prior for every sample in the data.

    The local KL divergence is a rarely used measure, for details, see:

    Varley, T. F. (2024).
    Generalized decomposition of multivariate information.
    PLOS ONE, 19(2), e0297128.
    https://doi.org/10.1371/journal.pone.0297128

    Parameters
    ----------
    cov_posterior : np.ndarray
        The covariance matrix that defines the posterior distribution.
    cov_prior : np.ndarray
        The covariance matrix that defines the prior distribution .
    data : np.ndarray
        The data, assumed to be in channels x samples format.

    Returns
    -------
    np.ndarray.
        The local Kullback-Leibler divergence.

    """

    h_posterior = local_differential_entropy(data, cov_posterior)
    h_prior = local_differential_entropy(data, cov_prior)

    return h_prior - h_posterior

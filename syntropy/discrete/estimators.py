import numpy as np

from scipy.special import digamma
from numpy.typing import NDArray
from typing import Any, Sequence, Callable

from .shannon import shannon_entropy, mutual_information
from .utils import get_marginal_distribution

DiscreteDist = dict[tuple[Any, ...], float]


def dirichlet_probabilities(
    data: NDArray[Any],
    prior: str | Sequence[float],
    alphabet: Sequence[Any] | None = None,
) -> DiscreteDist:
    """
    Estimate probabilities using Dirichlet-Multinomial Bayesian model.

    Parameters
    ----------
    data : NDArray[Any]
        2D array of shape (channels, samples) containing discrete observations.
    prior : str | Sequence[float]
        Either a named prior ('mle', 'laplace', 'jeffreys', 'perks') or a 
        custom vector of pseudocounts with length matching the alphabet size.
    alphabet : Sequence[Any] | None
        Optional full alphabet of possible states. If None, uses only observed states.

    Returns
    -------
    DiscreteDist
        Dictionary mapping states to their estimated probabilities. Only includes
        states with non-zero probability.

    References
    ----------
    Perks, W. (1947). Some observations on inverse probability including a new indifference rule. Journal of the Institute of Actuaries, 73, 285-334.
    
    Hausser, J., & Strimmer, K. (2009). Entropy inference and the James-Stein estimator, with application to nonlinear gene association networks. 
    Journal of Machine Learning Research, 10, 1469-1484. 
    https://jmlr.org/papers/v10/hausser09a.html
    """
    assert len(data.shape) == 2, "The data must be two-dimensional."

    # Extract observed symbols (columns are observations)
    unq, counts = np.unique(data, axis=1, return_counts=True)
    observed_states = [tuple(unq[:, i]) for i in range(unq.shape[1])]
    observed_counts = dict(zip(observed_states, counts))

    N = data.shape[1]

    states: Sequence[Any]
    counts_aligned: NDArray[np.integer]
    # Determine full support and aligned counts
    if alphabet is None:
        states = observed_states
        counts_aligned = np.array([observed_counts[s] for s in states])
    else:
        states = alphabet  # fixed, deterministic order
        counts_aligned = np.array([observed_counts.get(s, 0) for s in states])

    k: int = len(states)
    alpha: float
    alpha_vec: NDArray[np.floating]
    if isinstance(prior, str):
        assert prior in {"mle", "laplace", "jeffreys", "perks"}

        if prior == "mle":
            alpha = 0.0
        elif prior == "laplace":
            alpha = 1.0
        elif prior == "jeffreys":
            alpha = 0.5
        elif prior == "perks":
            alpha = 1.0 / k

        alpha_vec = np.full(k, alpha)
    else:
        alpha_vec = np.array(prior, dtype=float)
        if len(alpha_vec) != k:
            raise ValueError(
                f"Length of prior ({len(alpha_vec)}) must match number of states ({k})"
            )

    probs = (counts_aligned + alpha_vec) / (N + alpha_vec.sum())

    return {states[i]: probs[i] for i in range(k) if probs[i] > 0}


def plugin_probabilities(data: NDArray[Any]) -> DiscreteDist:
    """
    Maximum likelihood (plugin) probability estimator.

    Estimates probabilities using empirical frequencies with no smoothing.
    Equivalent to Dirichlet prior with alpha=0.

    Parameters
    ----------
    data : NDArray[Any]
        2D array of shape (channels, samples) containing discrete observations.

    Returns
    -------
    DiscreteDist
        Dictionary mapping observed states to their empirical frequencies.

    References
    ----------
    Paninski, L. (2003). Estimation of entropy and mutual information. 
        Neural Computation, 15(6), 1191-1253. 
        https://doi.org/10.1162/089976603321780272
    """
    return dirichlet_probabilities(data, prior="mle")


def miller_madow_entropy(data: NDArray[Any]) -> float:
    """
    Miller-Madow bias-corrected entropy estimator.

    Applies a first-order analytical correction to the maximum likelihood 
    entropy estimate: H_MM = H_ML + (K-1)/(2N), where K is the number of 
    observed bins and N is the sample size.

    Parameters
    ----------
    data : NDArray[Any]
        2D array of shape (channels, samples) containing discrete observations.

    Returns
    -------
    float
        Bias-corrected entropy estimate in nats.

    References
    ----------
    Miller, G. A. (1955). Note on the bias of information estimates. In H. Quastler (Ed.), Information Theory in Psychology: Problems and Methods (pp. 95-100). Free Press.
    """
    N: int = data.shape[1]

    joint_distribution: DiscreteDist
    ent: float

    joint_distribution = plugin_probabilities(data)

    _, ent = shannon_entropy(joint_distribution)

    k: int = len(joint_distribution)

    return ent + ((k - 1) / (2 * N))


def grassberger_entropy(data: NDArray[Any]) -> float:
    """
    Grassberger bias-corrected entropy estimator.

    Uses digamma function corrections to provide improved entropy estimates,
    particularly effective for moderately undersampled distributions.

    Parameters
    ----------
    data : NDArray[Any]
        2D array of shape (channels, samples) containing discrete observations.

    Returns
    -------
    float
        Bias-corrected entropy estimate in nats.

    References
    ----------
    Grassberger, P. (2003). Entropy estimates from insufficient samplings. 
    https://arxiv.org/abs/physics/0307138
    """

    assert len(data.shape) == 2, (
        "The data array must be two-dimensional, in channels x time format."
    )

    N: int = data.shape[1]
    digamma_N: float = digamma(N)

    counts: NDArray[np.integer]
    _, counts = np.unique(data, axis=1, return_counts=True)

    ent: float = 0.0
    for i in range(counts.shape[0]):
        ent += (counts[i] / N) * (digamma(counts[i]) - digamma_N)

    return -ent


def chao_shen_entropy(data: NDArray[Any]) -> float:
    """
    Chao-Shen coverage-adjusted entropy estimator.

    Estimates entropy by adjusting for unseen species using Good-Turing 
    coverage estimation. Particularly effective for severely undersampled 
    distributions where many states may be unobserved.

    Parameters
    ----------
    data : NDArray[Any]
        2D array of shape (channels, samples) containing discrete observations.

    Returns
    -------
    float
        Coverage-adjusted entropy estimate in nats.

    References
    ----------
    Chao, A., & Shen, T.-J. (2003). Nonparametric estimation of Shannon's index of diversity when there are unseen species in sample. Environmental and Ecological Statistics, 10(4), 429-443.
    https://doi.org/10.1023/A:1026096204727
    """
    unq, counts = np.unique(data, axis=1, return_counts=True)
    N = counts.sum()

    # Coverage: proportion NOT on singletons
    f1 = np.sum(counts == 1)  # number of singletons
    C = 1 - f1 / N if N > 0 else 0

    if C == 0:
        return 0.0

    # Adjusted probabilities
    probs = counts / N
    probs_adjusted = C * probs

    # Only sum over positive adjusted probs
    mask = probs_adjusted > 0
    return -np.sum(probs_adjusted[mask] * np.log(probs_adjusted[mask]))


def dirichlet_entropy(
    data: NDArray[Any],
    alphabet: Sequence[Any] | None = None,
    alpha: float | None = None,
) -> float:
    """
    Bayesian entropy estimator using Dirichlet prior.

    Computes the posterior mean of entropy under a Dirichlet-Multinomial 
    Bayesian model. Default prior is alpha = 1/K (Perks prior).

    Parameters
    ----------
    data : NDArray[Any]
        2D array of shape (channels, samples) containing discrete observations.
    alphabet : Sequence[Any] | None
        Optional full alphabet of possible states. If None, uses only observed states.
    alpha : float | None
        Symmetric Dirichlet prior pseudocount parameter. If None, defaults to 
        1/K (Perks prior).

    Returns
    -------
    float
        Bayesian entropy estimate in nats.

    References
    ----------
    Wolpert, D. H., & Wolf, D. R. (1995). Estimating functions of probability 
        distributions from a finite set of samples. Physical Review E, 52(6), 
        6841-6851. https://doi.org/10.1103/PhysRevE.52.6841
    
    Hausser, J., & Strimmer, K. (2009). Entropy inference and the James-Stein 
        estimator, with application to nonlinear gene association networks. 
        Journal of Machine Learning Research, 10, 1469-1484. 
        https://jmlr.org/papers/v10/hausser09a.html
    """
    assert len(data.shape) == 2, "Data must be 2D: channels x samples"

    # Count occurrences of each observed column (joint state)
    unq, counts = np.unique(data, axis=1, return_counts=True)
    observed_states = [tuple(unq[:, i]) for i in range(unq.shape[1])]
    observed_counts = dict(zip(observed_states, counts))

    # Determine full alphabet / state support
    if alphabet is None:
        states = observed_states
    else:
        states = list(alphabet)

    counts_aligned = np.array([observed_counts.get(s, 0) for s in states], dtype=float)

    k = len(states)
    N = counts_aligned.sum()

    # Default prior: weak uniform if not provided
    if alpha is None:
        alpha = 1.0 / k

    # Posterior Dirichlet parameters
    alpha_vec = counts_aligned + alpha
    denom = N + k * alpha

    # SG formula
    ent = float(digamma(denom + 1) - np.sum(alpha_vec * digamma(alpha_vec + 1)) / denom)

    return ent


def panzeri_treves_mutual_information(
    idxs_x: tuple[int, ...], idxs_y: tuple[int, ...], data: NDArray[Any]
) -> float:
    """
    Panzeri-Treves bias-corrected mutual information estimator.

    Applies an analytical bias correction to the plugin MI estimate based on 
    the number of bins in the marginal distributions.

    Parameters
    ----------
    idxs_x : tuple[int, ...]
        Channel indices for variable X.
    idxs_y : tuple[int, ...]
        Channel indices for variable Y.
    data : NDArray[Any]
        2D array of shape (channels, samples) containing discrete observations.

    Returns
    -------
    float
        Bias-corrected mutual information estimate in nats.

    References
    ----------
    Panzeri, S., & Treves, A. (1996). Analytical estimates of limited sampling biases in different information measures. Network: Computation in Neural Systems, 7(1), 87-107. 
    10.1080/0954898X.1996.11978656 
    """
    joint: tuple[int, ...] = idxs_x + idxs_y
    N: int = len(joint)

    joint_distribution: DiscreteDist
    joint_distribution = plugin_probabilities(data[joint, :])

    _idxs_x = tuple(i for i in range(len(idxs_x)))
    _idxs_y = tuple(i for i in range(len(idxs_x), N))

    dist_x: DiscreteDist = get_marginal_distribution(_idxs_x, joint_distribution)
    dist_y: DiscreteDist = get_marginal_distribution(_idxs_y, joint_distribution)

    r: int = len(dist_x)
    c: int = len(dist_y)

    _, mi = mutual_information(idxs_x, idxs_y, joint_distribution)
    return mi - ((r - 1) * (c - 1)) / (2 * N)


def mutual_information(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    data: NDArray[Any],
    estimator: Callable[..., float],
    **entropy_kwargs,
) -> float:
    """
    Mutual information I(X;Y) using specified entropy estimator.

    Computes mutual information via the entropy decomposition:
    I(X;Y) = H(X) + H(Y) - H(X,Y)

    Parameters
    ----------
    idxs_x : tuple[int, ...]
        Channel indices for variable X.
    idxs_y : tuple[int, ...]
        Channel indices for variable Y.
    data : NDArray[Any]
        2D array of shape (channels, samples) containing discrete observations.
    estimator : Callable[..., float]
        Entropy estimation function (e.g., grassberger_entropy, chao_shen_entropy).
    **entropy_kwargs
        Additional keyword arguments passed to the entropy estimator.

    Returns
    -------
    float
        Mutual information estimate in nats.

    References
    ----------
    Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory 
        (2nd ed.). Wiley-Interscience.
    """
    joint: tuple[int, ...] = idxs_x + idxs_y

    h_x: float = estimator(data[idxs_x, :], **entropy_kwargs)
    h_y: float = estimator(data[idxs_y, :], **entropy_kwargs)
    h_joint: float = estimator(data[joint, :], **entropy_kwargs)

    return h_x + h_y - h_joint


def conditional_mutual_information(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    idxs_z: tuple[int, ...],
    data: NDArray[Any],
    estimator: Callable[..., float],
    **entropy_kwargs,
) -> float:
    """
    Conditional mutual information I(X;Y|Z) using specified entropy estimator.

    Computes conditional mutual information via the entropy decomposition:
    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)

    Parameters
    ----------
    idxs_x : tuple[int, ...]
        Channel indices for variable X.
    idxs_y : tuple[int, ...]
        Channel indices for variable Y.
    idxs_z : tuple[int, ...]
        Channel indices for conditioning variable Z.
    data : NDArray[Any]
        2D array of shape (channels, samples) containing discrete observations.
    estimator : Callable[..., float]
        Entropy estimation function (e.g., grassberger_entropy, chao_shen_entropy).
    **entropy_kwargs
        Additional keyword arguments passed to the entropy estimator.

    Returns
    -------
    float
        Conditional mutual information estimate in nats.

    References
    ----------
    Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory 
        (2nd ed.). Wiley-Interscience.
    """
    xz: tuple[int, ...] = idxs_x + idxs_z
    yz: tuple[int, ...] = idxs_y + idxs_z
    xyz: tuple[int, ...] = idxs_x + idxs_y + idxs_z

    h_xz: float = estimator(data[xz, :], **entropy_kwargs)
    h_yz: float = estimator(data[yz, :], **entropy_kwargs)
    h_z: float = estimator(data[idxs_z, :], **entropy_kwargs)
    h_xyz: float = estimator(data[xyz, :], **entropy_kwargs)

    return h_xz + h_yz - h_z - h_xyz


def total_correlation(
    idxs: tuple[int, ...],
    data: NDArray[Any],
    estimator: Callable[..., float],
    **entropy_kwargs,
) -> float:
    """
    Total correlation TC(X₁, X₂, ..., Xₙ) using specified entropy estimator.

    Also known as multi-information or integration. Computes total correlation
    via the entropy decomposition: TC = Σᵢ H(Xᵢ) - H(X₁, X₂, ..., Xₙ)

    Parameters
    ----------
    idxs : tuple[int, ...]
        Channel indices for all variables.
    data : NDArray[Any]
        2D array of shape (channels, samples) containing discrete observations.
    estimator : Callable[..., float]
        Entropy estimation function (e.g., grassberger_entropy, chao_shen_entropy).
    **entropy_kwargs
        Additional keyword arguments passed to the entropy estimator.

    Returns
    -------
    float
        Total correlation estimate in nats.

    References
    ----------
    Watanabe, S. (1960). Information theoretical analysis of multivariate 
        correlation. IBM Journal of Research and Development, 4(1), 66-82. 
        https://doi.org/10.1147/rd.41.0066
    """
    # Sum of marginal entropies
    h_marginals: float = sum(
        estimator(data[i : i + 1, :], **entropy_kwargs) for i in idxs
    )

    # Joint entropy
    h_joint: float = estimator(data[idxs, :], **entropy_kwargs)

    return h_marginals - h_joint

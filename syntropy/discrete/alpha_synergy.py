import itertools
import numpy as np

from syntropy.discrete.shannon import conditional_entropy
from syntropy.discrete.optimization import constrained_maximum_entropy_distributions
from syntropy.discrete.utils import get_marginal_distribution, reduce_state

from typing import Any

DiscreteDist = dict[tuple[Any, ...], float]
AlphaSynDist = dict[tuple[Any, ...], float]
PartialSpectra = dict[tuple[Any, ...], list[float]]


def alpha_synergistic_entropy(
    joint_distribution: DiscreteDist,
    alpha: int,
    num_samples: int = -1,
    definition: str = "min",
) -> AlphaSynDist:
    """
    Computes the :math:`\\alpha`-synergistic entropy for a joint distribution
    for a given value of :math:`\\alpha`. See:

        Varley, T. F. (2024).
        A scalable synergy-first backbone decomposition of
        higher-order structures in complex systems.
        Npj Complexity, 1(1), 1–11.
        https://doi.org/10.1038/s44260-024-00011-1


    Parameters
    ----------
    joint_distribution : dict[tuple, float]
        The joint distribution dictionary object.
    alpha : int
        The scale to consider.
    num_samples : int, optional
        The number of samples to trial. The default is -1, in which case, all permutations are trialed.
    definition : str, optional
        How to define the loss of information. Can be "min", "max", or "avg". The default is "min".

    Returns
    -------
    dict[tuple, float]
        The local alpha-synergy for each state.

    """
    assert definition in {
        "min",
        "max",
        "avg",
    }, "The optional definitions are 'min', 'max', or 'avg'."

    N: int = len(next(iter(joint_distribution)))

    val: float = 0.0

    if definition == "min":
        val = np.inf
    elif definition == "max":
        val = -np.inf

    alpha_syns: AlphaSynDist = {key: val for key in joint_distribution.keys()}

    sources: list[tuple[Any, ...]] = []
    if num_samples == -1:  # Get all combinations of elements of size alpha.
        sources += list(itertools.combinations(tuple(i for i in range(N)), r=alpha))
    else:  # Randomly sample elements of size alpha.
        sources += list(
            {
                tuple(sorted(np.random.choice(N, size=alpha, replace=False).tolist()))
                for i in range(num_samples)
            }
        )

    num_sources: int = len(sources)

    residuals: list[
        tuple[Any, ...]
    ] = [  # The residual indices for each source in sources.
        tuple(i for i in range(N) if i not in source) for source in sources
    ]

    for i in range(num_sources):
        source: tuple[Any, ...] = sources[i]  # The indices of the source variables.
        complement: tuple[Any, ...] = residuals[
            i
        ]  # The indices of the complenetary variables.

        idxs: tuple[Any, ...] = source + complement

        ptw, _ = conditional_entropy(source, complement, joint_distribution)

        for key in ptw.keys():
            # Undoing the unpacking of state into source and
            # conditional
            flatten: tuple[Any, ...] = sum(key, ())  # ((0,1),(2,)) -> (0,1,2)

            temp: list[int] = [0] * N
            for i in range(N):  # For each element of flatten:
                temp[idxs[i]] = flatten[i]  # Mapping back to the original order.
            state: tuple[Any, ...] = tuple(temp)

            # Accounting for the various ways synergy can be defined.
            if definition == "min":
                if alpha_syns[state] > ptw[key]:
                    alpha_syns[state] = ptw[key]
            elif definition == "max":
                if alpha_syns[state] < ptw[key]:
                    alpha_syns[state] = ptw[key]
            elif definition == "avg":
                alpha_syns[state] += ptw[key] / float(num_sources)

    return alpha_syns


def partial_entropy_spectra(
    joint_distribution: DiscreteDist,
    num_samples: int = -1,
    definition: str = "min",
) -> PartialSpectra:
    """
    Computes the partial entropy spectrum for each state.

    Parameters
    ----------
    joint_distribution : dict[tuple,float]
        The joint probability dictionary object.
    num_samples : int, optional
        The number of samples to trial. The default is -1, in which case, all permutations are trialed.
    definition : str, optional
        How to define the loss of information. Can be "min", "max", or "avg". The default is "min".

    Returns
    -------
    dict[tuple,list]
        The alpha-synergistic entropy spectrum for each state.

    """

    N: int = len(list(joint_distribution.keys())[0])
    spectra: dict[tuple[Any, ...], list[float]] = {
        key: [] for key in joint_distribution.keys()
    }

    for alpha in range(1, N + 1):
        alpha_syns: AlphaSynDist = alpha_synergistic_entropy(
            joint_distribution,
            alpha=alpha,
            num_samples=num_samples,
            definition=definition,
        )

        for key in alpha_syns.keys():
            spectra[key].append(alpha_syns[key] - sum(spectra[key]))

    return spectra


def partial_kullback_leibler_spectra(
    posterior: DiscreteDist,
    prior: DiscreteDist,
    num_samples: int = -1,
    definition: str = "min",
) -> PartialSpectra:
    """
    Computes the local Kullback-Leibler spectrum for each state

    Parameters
    ----------
    posterior : dict[tuple,float]
        The distribution that describes the posterior beliefs.
    prior : dict[tuple,float]
        The distribution that describes the prior beliefs.
    num_samples : int, optional
        The number of samples to trial.
        The default is -1, in which case, all permutations are trialed.
    definition : str, optional
        How to define the loss of information.
        Can be "min", "max", or "avg". The default is "min".

    Returns
    -------
    dict[tuple,list]
        The alpha-synergistic DKL spectrum for each state.

    """

    dkl_spectra: PartialSpectra = {key: [] for key in posterior.keys()}
    prior_spectra: PartialSpectra = partial_entropy_spectra(
        prior, num_samples=num_samples, definition=definition
    )
    posterior_spectra: PartialSpectra = partial_entropy_spectra(
        posterior, num_samples=num_samples, definition=definition
    )

    for key in dkl_spectra.keys():
        dkl_spectra[key] = list(
            map(lambda x, y: x - y, prior_spectra[key], posterior_spectra[key])
        )

    return dkl_spectra


def partial_total_correlation_spectra(
    joint_distribution: DiscreteDist,
    num_samples: int = -1,
    definition: str = "min",
) -> PartialSpectra:
    """
    Computes the local total correlation spectrum for each state.

    Parameters
    ----------
    joint_distribution : dict[tuple,float]
        The joint probability dictionary object.
    num_samples : int, optional
        The number of samples to trial. The default is -1, in which case, all permutations are trialed.
    definition : str, optional
        How to define the loss of information. Can be "min", "max", or "avg". The default is "min".

    Returns
    -------
    dict[tuple,list]
        The alpha-synergistic total correlation spectrum for each state.


    """

    prior: DiscreteDist = constrained_maximum_entropy_distributions(
        joint_distribution, order=1
    )

    return partial_kullback_leibler_spectra(
        joint_distribution, prior, num_samples=num_samples, definition=definition
    )


def partial_information_spectra(
    inputs: tuple[Any, ...],
    target: tuple[Any, ...],
    joint_distribution: DiscreteDist,
    num_samples: int = -1,
    definition: str = "min",
) -> list[float]:
    """
    Computes the local mutual information spectrum for each state.

    Parameters
    ----------
    inputs : tuple
        The indices of the input variables.
    target : tuple
        The indices of the target variables.
    joint_distribution : dict[tuple,float]
        The joint probability dictionary object.
    num_samples : int, optional
        The number of samples to trial. The default is -1, in which case, all permutations are trialed.
    definition : str, optional
        How to define the loss of information. Can be "min", "max", or "avg". The default is "min".

    Returns
    -------
    list[float]
    """

    N: int = len(inputs)

    # The marginal distribution on the inputs
    # and target.
    marginal_inputs: DiscreteDist = get_marginal_distribution(
        inputs, joint_distribution
    )
    marginal_targets: DiscreteDist = get_marginal_distribution(
        target, joint_distribution
    )

    # The alpha-synergistic entropy decomposition of h(x)
    input_entropy_spectra: PartialSpectra = partial_entropy_spectra(
        marginal_inputs, num_samples=num_samples, definition=definition
    )

    avg: list[float] = [0.0 for _ in range(N)]
    for target_key in marginal_targets.keys():
        p_y: float = marginal_targets[target_key]

        target_conditional_distribution: DiscreteDist = {
            reduce_state(key, inputs): joint_distribution[key] / p_y
            for key in joint_distribution.keys()
            if reduce_state(key, target) == target_key
        }

        conditional_entropy_spectra: PartialSpectra = partial_entropy_spectra(
            target_conditional_distribution,
            num_samples=num_samples,
            definition=definition,
        )

        ptw: list[float] = [0 for _ in range(N)]
        for input_key in conditional_entropy_spectra.keys():
            
            input_spectra: list[float] = input_entropy_spectra[input_key]
            conditional_spectra: list[float] = conditional_entropy_spectra[input_key]
            target_conditional_entropy: float = target_conditional_distribution[input_key]

            for i in range(N):
                ptw[i] += target_conditional_entropy * (
                    input_spectra[i] - conditional_spectra[i]
                )
        
        for i in range(N):
            avg[i] += ptw[i] * p_y
        
    return avg

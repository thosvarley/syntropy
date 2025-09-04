import itertools
import numpy as np

from syntropy.discrete.shannon import conditional_entropy
from syntropy.discrete.optimization import constrained_maximum_entropy_distributions
from syntropy.discrete.utils import get_marginal_distribution


def alpha_synergistic_entropy(
    joint_distribution: dict[tuple, float],
    alpha: int,
    num_samples: int = -1,
    definition: str = "min",
) -> dict[tuple, float]:
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

    N: int = len(list(joint_distribution.keys())[0])

    if definition == "min":
        val: float = np.inf
    elif definition == "max":
        val: float = -np.inf
    elif val == "average":
        val: float = 0.0

    alpha_syns: dict[tuple, float] = {key: val for key in joint_distribution.keys()}

    sources: list[tuple] = []
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

    residuals: list = [
        tuple(i for i in range(N) if i not in source) for source in sources
    ]

    for i in range(num_sources):

        source: tuple = sources[i]
        complement: tuple = residuals[i]

        idxs: tuple = source + complement

        ptw, _ = conditional_entropy(source, complement, joint_distribution)

        for key in ptw.keys():

            # Undoing the unpacking of state into source and
            # conditional
            flatten: tuple = sum(key, ())  # ((0,1),(2,)) -> (0,1,2)

            state = [0] * N
            for i in range(N):
                state[idxs[i]] = flatten[i]
            state = tuple(state)

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
    joint_distribution: dict[tuple, float],
    num_samples: int = -1,
    definition: str = "min",
) -> dict[tuple, list]:
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
    spectra: dict[tuple, list] = {key: [] for key in joint_distribution.keys()}

    for alpha in range(1, N + 1):
        alpha_syns = alpha_synergistic_entropy(
            joint_distribution,
            alpha=alpha,
            num_samples=num_samples,
            definition=definition,
        )

        for key in alpha_syns.keys():
            spectra[key].append(alpha_syns[key] - sum(spectra[key]))

    return spectra


def partial_kullback_leibler_spectra(
    posterior: dict[tuple, float],
    prior: dict[tuple, float],
    num_samples: int = -1,
    definition: str = "min",
) -> dict[tuple, list]:
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

    dkl_spectra: dict[tuple, list] = {key: [] for key in posterior.keys()}
    prior_spectra: dict[tuple, list] = partial_entropy_spectra(
        prior, num_samples=num_samples, definition=definition
    )
    posterior_spectra: dict[tuple, list] = partial_entropy_spectra(
        posterior, num_samples=num_samples, definition=definition
    )

    for key in dkl_spectra.keys():

        dkl_spectra[key] = list(
            map(lambda x, y: x - y, prior_spectra[key], posterior_spectra[key])
        )

    return dkl_spectra


def partial_total_correlation_spectra(
    joint_distribution: dict[tuple, float],
    num_samples: int = -1,
    definition: str = "min",
) -> dict[tuple, list]:
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

    prior: dict[tuple, float] = constrained_maximum_entropy_distributions(
        joint_distribution, order=1
    )

    return partial_kullback_leibler_spectra(
        joint_distribution, prior, num_samples=num_samples, definition=definition
    )


def partial_information_spectra(
    inputs: tuple,
    target: tuple,
    joint_distribution: dict[tuple, float],
    num_samples: int = -1,
    definition: str = "min",
) -> dict[tuple, float]:
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
    dict[tuple,list]
        The alpha-synergistic total correlation spectrum for each state.

    """

    N = len(inputs)

    # The marginal distribution on the inputs
    # and target.
    marginal_inputs = get_marginal_distribution(inputs, joint_distribution)
    marginal_targets = get_marginal_distribution(target, joint_distribution)

    # The alpha-synergistic entropy decomposition of h(x)
    input_entropy_spectra = partial_entropy_spectra(
        marginal_inputs, num_samples=num_samples, definition=definition
    )

    conditional_distribution = conditional_entropy(inputs, target, joint_distribution)[
        0
    ]

    # Converting the pointwise conditional entropies back into a
    # probability distribution.
    conditional_distribution = {
        key: 2 ** (-conditional_distribution[key])
        for key in conditional_distribution.keys()
    }

    avg = [0 for i in range(N)]
    for target_key in marginal_targets.keys():

        target_conditional_distribution = {
            key[0]: conditional_distribution[key]
            for key in conditional_distribution.keys()
            if key[1] == target_key
        }

        conditional_entropy_spectra = partial_entropy_spectra(
            target_conditional_distribution,
            num_samples=num_samples,
            definition=definition,
        )

        ptw = [0 for i in range(N)]
        for input_key in conditional_entropy_spectra.keys():

            input_spectra = input_entropy_spectra[input_key]
            conditional_spectra = conditional_entropy_spectra[input_key]

            for i in range(N):
                ptw[i] += target_conditional_distribution[input_key] * (
                    input_spectra[i] - conditional_spectra[i]
                )

        for i in range(N):

            avg[i] += ptw[i] * marginal_targets[target_key]

    return avg

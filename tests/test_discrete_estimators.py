import pytest
import numpy as np
from scipy.special import digamma

from syntropy.discrete.estimators import (
    dirichlet_probabilities,
    plugin_probabilities,
    miller_madow_entropy,
    grassberger_entropy,
    chao_shen_entropy,
    dirichlet_entropy,
    panzeri_treves_mutual_information,
)
from syntropy.discrete.shannon import (
    shannon_entropy,
    mutual_information as shannon_mutual_information,
)

pytest_abs = 1e-9

# The estimators in this module all return entropy / mutual information in
# NATS (natural log), matching their docstrings, whereas the plugin
# distributions and the syntropy.discrete.shannon helpers work in BITS
# (log2). A couple of tests below pin down that convention explicitly.
LN2 = np.log(2.0)


def _plugin_entropy_nats(joint_distribution) -> float:
    _, ent_bits = shannon_entropy(joint_distribution)
    return ent_bits * LN2


def test_plugin_probabilities():
    data = np.array([[0, 0, 0, 1]])
    result = plugin_probabilities(data)

    assert result == {(0,): pytest.approx(0.75), (1,): pytest.approx(0.25)}
    assert result == dirichlet_probabilities(data, prior="mle")

    rng = np.random.default_rng(0)
    random_data = rng.integers(0, 4, size=(2, 500))
    assert sum(plugin_probabilities(random_data).values()) == pytest.approx(
        1.0, abs=pytest_abs
    )


def test_dirichlet_probabilities_priors():
    data = np.array([[0, 0, 0, 1]])  # counts (0,):3, (1,):1, so N=4, K=2

    # (count + alpha) / (N + K*alpha)
    laplace = dirichlet_probabilities(data, prior="laplace")  # alpha=1
    assert laplace[(0,)] == pytest.approx(4 / 6, abs=pytest_abs)
    assert laplace[(1,)] == pytest.approx(2 / 6, abs=pytest_abs)

    jeffreys = dirichlet_probabilities(data, prior="jeffreys")  # alpha=0.5
    assert jeffreys[(0,)] == pytest.approx(3.5 / 5, abs=pytest_abs)
    assert jeffreys[(1,)] == pytest.approx(1.5 / 5, abs=pytest_abs)

    # With Laplace smoothing an unobserved state (2,) still receives mass:
    # counts are 3, 1, 0; N=4, K=3 -> (count+1)/(4+3)
    with_alphabet = dirichlet_probabilities(
        data, prior="laplace", alphabet=[(0,), (1,), (2,)]
    )
    assert with_alphabet[(0,)] == pytest.approx(4 / 7, abs=pytest_abs)
    assert with_alphabet[(1,)] == pytest.approx(2 / 7, abs=pytest_abs)
    assert with_alphabet[(2,)] == pytest.approx(1 / 7, abs=pytest_abs)

    rng = np.random.default_rng(1)
    random_data = rng.integers(0, 3, size=(2, 200))
    for prior in ("laplace", "jeffreys", "perks"):
        result = dirichlet_probabilities(random_data, prior=prior)
        assert sum(result.values()) == pytest.approx(1.0, abs=pytest_abs)


def test_dirichlet_probabilities_custom_prior_length_mismatch():
    data = np.array([[0, 0, 0, 1]])  # 2 observed states
    with pytest.raises(ValueError):
        dirichlet_probabilities(data, prior=[1.0])


def test_miller_madow_entropy():
    # Four equally likely states over N=4 samples: plugin entropy is ln(4)
    # nats, and the Miller-Madow correction is (k-1)/(2N) = 3/8.
    uniform = np.array([[0, 1, 2, 3]])
    assert miller_madow_entropy(uniform) == pytest.approx(
        np.log(4) + 3 / (2 * 4), abs=pytest_abs
    )

    rng = np.random.default_rng(2)
    data = rng.integers(0, 5, size=(1, 300))
    joint = plugin_probabilities(data)
    k, N = len(joint), data.shape[1]
    assert miller_madow_entropy(data) == pytest.approx(
        _plugin_entropy_nats(joint) + (k - 1) / (2 * N), abs=pytest_abs
    )

    # The correction is strictly positive, so Miller-Madow always lies above
    # the (downward-biased) plugin estimate.
    assert miller_madow_entropy(data) > _plugin_entropy_nats(joint)


def test_grassberger_entropy():
    # Two states each observed twice: -Σ (n_i/N)(ψ(n_i) - ψ(N)).
    data = np.array([[0, 0, 1, 1]])
    expected = -(2 * (2 / 4) * (digamma(2) - digamma(4)))
    assert grassberger_entropy(data) == pytest.approx(expected, abs=pytest_abs)
    assert grassberger_entropy(data) >= 0.0

    # Uniform over 4 states -> true entropy ln(4) nats.
    rng = np.random.default_rng(5)
    large = rng.integers(0, 4, size=(1, 50_000))
    assert grassberger_entropy(large) == pytest.approx(np.log(4), abs=1e-2)


def test_grassberger_rejects_non_2d():
    with pytest.raises(AssertionError):
        grassberger_entropy(np.array([0, 0, 1, 1]))


def test_chao_shen_entropy():
    # No singleton counts -> coverage C = 1, so the estimator collapses to
    # the plain plugin entropy (in nats).
    no_singletons = np.array([[0, 0, 1, 1]])
    assert chao_shen_entropy(no_singletons) == pytest.approx(np.log(2), abs=pytest_abs)

    rng = np.random.default_rng(6)
    assert chao_shen_entropy(rng.integers(0, 8, size=(1, 60))) >= 0.0

    large = rng.integers(0, 4, size=(1, 50_000))
    assert chao_shen_entropy(large) == pytest.approx(np.log(4), abs=1e-2)


def test_dirichlet_entropy():
    # Independent recomputation of the posterior-mean (Schürmann-Grassberger)
    # formula from digamma primitives, as a cross-check on the implementation.
    data = np.array([[0, 0, 1, 1, 2]])
    _, counts = np.unique(data, axis=1, return_counts=True)
    counts = counts.astype(float)
    k = len(counts)
    N = counts.sum()
    alpha = 1.0 / k
    alpha_vec = counts + alpha
    denom = N + k * alpha
    expected = float(
        digamma(denom + 1) - np.sum(alpha_vec * digamma(alpha_vec + 1)) / denom
    )
    assert dirichlet_entropy(data) == pytest.approx(expected, abs=pytest_abs)
    assert dirichlet_entropy(data) > 0.0

    rng = np.random.default_rng(9)
    large = rng.integers(0, 4, size=(1, 50_000))
    assert dirichlet_entropy(large) == pytest.approx(np.log(4), abs=1e-2)


def test_panzeri_treves_mutual_information():
    rng = np.random.default_rng(10)
    data = np.vstack([rng.integers(0, 3, size=500), rng.integers(0, 2, size=500)])
    joint = plugin_probabilities(data)
    _, mi_bits = shannon_mutual_information((0,), (1,), joint)
    mi_nats = mi_bits * LN2

    r, c = len(set(data[0])), len(set(data[1]))
    num_samples = data.shape[1]
    expected = mi_nats - (r - 1) * (c - 1) / (2 * num_samples)
    assert panzeri_treves_mutual_information((0,), (1,), data) == pytest.approx(
        expected, abs=pytest_abs
    )

    # For truly independent variables and a large sample, the corrected MI
    # should sit close to zero.
    independent = np.vstack(
        [rng.integers(0, 3, size=20_000), rng.integers(0, 3, size=20_000)]
    )
    assert panzeri_treves_mutual_information(
        (0,), (1,), independent
    ) == pytest.approx(0.0, abs=1e-2)


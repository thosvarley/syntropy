import pytest
import numpy as np

from syntropy.gaussian.optimization import (
    neg_o_information,
    simulated_annealing,
    irreducible_synergy,
)
from syntropy.gaussian.multivariate_mi import o_information
from helpers import equicorr_matrix

pytest_abs = 1e-9


def pos_o_information(x):
    """The un-negated counterpart to neg_o_information, for maximizing
    (rather than minimizing) O-information -- i.e. searching for the most
    redundancy-dominated subset instead of the most synergy-dominated one.
    """
    return o_information(*x)


def test_neg_o_information_is_negation():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    cov = A @ A.T + 4 * np.eye(4)
    idxs = (0, 1, 2, 3)

    assert neg_o_information((cov, idxs)) == pytest.approx(
        -o_information(cov, idxs), abs=pytest_abs
    )


# simulated_annealing fixtures: embed a strongly synergistic (or redundant)
# block of BLOCK_SIZE variables inside a larger covariance matrix of
# otherwise-independent variables, so the embedded block is the unique
# global optimum of O-information over every size-BLOCK_SIZE subset. The
# annealer should recover exactly that block regardless of where it's seeded
# from. Deterministic (no rng involved in building them), so unlike
# test_gaussian_decompositions.py these are fine to share across tests.
N_TOTAL = 8
BLOCK_SIZE = 3

COV_SYNERGY = np.eye(N_TOTAL)
COV_SYNERGY[:BLOCK_SIZE, :BLOCK_SIZE] = equicorr_matrix(BLOCK_SIZE, -0.45)

COV_REDUNDANT = np.eye(N_TOTAL)
COV_REDUNDANT[:BLOCK_SIZE, :BLOCK_SIZE] = equicorr_matrix(BLOCK_SIZE, 0.9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_simulated_annealing_recovers_known_synergistic_subset(seed):
    best_set, best_value, _ = simulated_annealing(
        cov=COV_SYNERGY, function=neg_o_information, size=BLOCK_SIZE, seed=seed
    )

    assert set(best_set) == {0, 1, 2}
    assert best_value == pytest.approx(-o_information(COV_SYNERGY, (0, 1, 2)), abs=1e-4)


def test_simulated_annealing_recovers_known_redundant_subset():
    """Same search, run in the opposite direction: maximizing the raw (not
    negated) O-information should recover the most redundancy-dominated
    subset instead of the most synergy-dominated one."""
    best_set, best_value, _ = simulated_annealing(
        cov=COV_REDUNDANT, function=pos_o_information, size=BLOCK_SIZE, seed=0
    )

    assert set(best_set) == {0, 1, 2}
    assert best_value == pytest.approx(o_information(COV_REDUNDANT, (0, 1, 2)), abs=1e-4)


def test_simulated_annealing_best_value_and_trace_invariants():
    best_set, best_value, values = simulated_annealing(
        cov=COV_SYNERGY, function=neg_o_information, size=BLOCK_SIZE, seed=1
    )

    recomputed = neg_o_information((COV_SYNERGY, tuple(sorted(best_set))))
    assert best_value == pytest.approx(recomputed, abs=pytest_abs)

    # `values` records the currently-accepted objective value at each
    # temperature step (which can dip below the best-ever value due to
    # Metropolis acceptance of worse moves), so it should never exceed
    # best_value.
    assert np.all(values <= best_value + 1e-9)


def test_simulated_annealing_reproducible_with_fixed_seed():
    result_a = simulated_annealing(
        cov=COV_SYNERGY, function=neg_o_information, size=BLOCK_SIZE, seed=42
    )
    result_b = simulated_annealing(
        cov=COV_SYNERGY, function=neg_o_information, size=BLOCK_SIZE, seed=42
    )

    assert result_a[0] == result_b[0]
    assert result_a[1] == result_b[1]
    assert np.array_equal(result_a[2], result_b[2])


def test_simulated_annealing_rejects_nonpositive_min_temperature():
    with pytest.raises(AssertionError):
        simulated_annealing(
            cov=COV_SYNERGY,
            function=neg_o_information,
            size=BLOCK_SIZE,
            min_temperature=0.0,
        )


def test_irreducible_synergy_true_for_pure_synergy_triplet():
    """A synergistic triplet where every leave-one-out pair has O = 0 (an
    O-information is undefined/neutral for N=2), which is never less than
    the triplet's negative O, so no removal can deepen the synergy."""
    cov = equicorr_matrix(3, -0.45)
    assert o_information(cov, (0, 1, 2)) < 0
    assert irreducible_synergy(cov, (0, 1, 2)) is True


def test_irreducible_synergy_false_for_reducible_quadruplet():
    """A concrete counterexample found by random search: a synergy-dominated
    quadruplet where removing variable 0 makes the O-information *more*
    negative (deepens the synergy), so the full set is reducible."""
    cov = np.array(
        [
            [0.6972889103563518, -0.09306668586768825, 0.44837132926085244, -0.47841733920359486],
            [-0.09306668586768825, 1.9614701992833765, 0.04817243578182886, -0.5939956127728808],
            [0.44837132926085244, 0.04817243578182886, 1.514698335058918, -0.44359009328925797],
            [-0.47841733920359486, -0.5939956127728808, -0.44359009328925797, 1.2724975003797734],
        ]
    )
    idxs = (0, 1, 2, 3)
    o_full = o_information(cov, idxs)
    o_remove_0 = o_information(cov, (1, 2, 3))

    assert o_full < 0
    assert o_remove_0 < o_full
    assert irreducible_synergy(cov, idxs) is False


def test_irreducible_synergy_rejects_redundancy_dominated_input():
    cov = equicorr_matrix(3, 0.5)
    assert o_information(cov, (0, 1, 2)) > 0
    with pytest.raises(AssertionError):
        irreducible_synergy(cov, (0, 1, 2))

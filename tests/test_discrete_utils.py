import pytest
import numpy as np

from syntropy.discrete.utils import (
    make_powerset,
    flatten_nested_tuple,
    clean_distribution,
    reduce_state,
    construct_joint_distribution,
    get_marginal_distribution,
    marginalize_out,
    get_all_marginal_distributions,
    product_distribution,
    generate_closed_distribution,
)
from syntropy.discrete.distributions import XOR_DIST, GIANT_BIT

pytest_abs = 1e-9


def test_make_powerset():
    assert set(make_powerset([1, 2, 3])) == {
        (),
        (1,),
        (2,),
        (3,),
        (1, 2),
        (1, 3),
        (2, 3),
        (1, 2, 3),
    }
    assert set(make_powerset([])) == {()}


def test_flatten_nested_tuple():
    assert flatten_nested_tuple(((1, 2), (3,), (4, 5, 6))) == (1, 2, 3, 4, 5, 6)
    assert flatten_nested_tuple(()) == ()
    assert flatten_nested_tuple(((), (1,), ())) == (1,)


def test_clean_distribution():
    dist = {(0, 0): 0.5, (0, 1): 0.0, (1, 0): 0.5, (1, 1): 0.0}
    assert clean_distribution(dist) == {(0, 0): 0.5, (1, 0): 0.5}
    assert clean_distribution(XOR_DIST) == XOR_DIST  # nothing to remove


def test_reduce_state():
    assert reduce_state((5, 6, 7), (0, 2)) == (5, 7)
    assert reduce_state((5, 6, 7), (1,)) == (6,)
    assert reduce_state((5, 6, 7), ()) == ()
    assert reduce_state((5, 6, 7), (2, 0)) == (7, 5)  # preserves source order


def test_construct_joint_distribution():
    uniform = construct_joint_distribution(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]))
    assert uniform == {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.25, (1, 1): 0.25}

    nonuniform = construct_joint_distribution(np.array([[0, 0, 0, 1], [0, 0, 0, 1]]))
    assert nonuniform[(0, 0)] == pytest.approx(0.75, abs=pytest_abs)
    assert nonuniform[(1, 1)] == pytest.approx(0.25, abs=pytest_abs)
    assert sum(nonuniform.values()) == pytest.approx(1.0, abs=pytest_abs)


def test_construct_joint_distribution_rejects_float_data():
    with pytest.raises(AssertionError):
        construct_joint_distribution(np.array([[0.0, 1.0], [0.0, 1.0]]))


def test_get_marginal_distribution():
    assert get_marginal_distribution((0,), XOR_DIST) == {(0,): 0.5, (1,): 0.5}
    assert get_marginal_distribution((1,), GIANT_BIT) == {(0,): 0.5, (1,): 0.5}
    # Marginalizing over every variable recovers the original distribution.
    assert get_marginal_distribution((0, 1, 2), XOR_DIST) == XOR_DIST


def test_marginalize_out():
    # Marginalizing out {0} is the complement of keeping {1, 2}.
    assert marginalize_out((0,), XOR_DIST) == get_marginal_distribution(
        (1, 2), XOR_DIST
    )
    assert marginalize_out((), XOR_DIST) == XOR_DIST
    assert marginalize_out((0, 1, 2), XOR_DIST) == {(): 1.0}


def test_get_all_marginal_distributions():
    marginals = get_all_marginal_distributions(XOR_DIST)

    assert set(marginals.keys()) == {
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    }
    assert marginals[(0, 1, 2)] == XOR_DIST  # the full-source marginal is the original
    for idxs in marginals:
        assert marginals[idxs] == get_marginal_distribution(idxs, XOR_DIST)


def test_product_distribution():
    result = product_distribution({(0,): 0.5, (1,): 0.5}, {(0,): 0.3, (1,): 0.7})
    assert result == {
        (0, 0): pytest.approx(0.15, abs=pytest_abs),
        (0, 1): pytest.approx(0.35, abs=pytest_abs),
        (1, 0): pytest.approx(0.15, abs=pytest_abs),
        (1, 1): pytest.approx(0.35, abs=pytest_abs),
    }
    assert sum(result.values()) == pytest.approx(1.0, abs=pytest_abs)

    # States concatenate rather than merge.
    assert product_distribution({(0,): 1.0}, {(1, 2): 1.0}) == {(0, 1, 2): 1.0}


def test_generate_closed_distribution_structure():
    dist = generate_closed_distribution(6, seed=0)

    assert sum(dist.values()) == pytest.approx(1.0, abs=pytest_abs)
    for state in dist:
        assert len(state) == 6
        assert set(state) <= {0, 1}

    # A closed distribution requires every pair of states in the support to
    # differ by a Hamming distance of at least 2, otherwise a single bit
    # flip would leave H(X_i | X^{-i}) > 0 for some i.
    states = list(dist.keys())
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            hamming = sum(a != b for a, b in zip(states[i], states[j]))
            assert hamming >= 2


def test_generate_closed_distribution_seed_behavior():
    assert generate_closed_distribution(4, seed=42) == generate_closed_distribution(
        4, seed=42
    )
    assert generate_closed_distribution(4, seed=0) != generate_closed_distribution(
        4, seed=1
    )

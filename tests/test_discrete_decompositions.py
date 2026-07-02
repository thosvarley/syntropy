import pytest
import numpy as np

from syntropy.discrete.decompositions import (
    local_precompute_sources,
    hmin_discrete_redundancy,
    hsx_discrete_redundancy,
    mmi_discrete_redundancy,
    ipm_discrete_redundancy,
    isx_discrete_redundancy,
    representational_complexity,
)
from syntropy.discrete.distributions import XOR_DIST, GIANT_BIT
from syntropy.discrete.shannon import mutual_information

pytest_abs = 1e-9

# Same Phi-ID example distribution used in tests/test_discrete.py::test_phiiid,
# taken from Varley (2023) https://doi.org/10.1371/journal.pone.0282950
DISINTEGRATED_SYSTEM = {
    (0, 0, 1, 1): 1 / 4,
    (1, 1, 0, 0): 1 / 4,
    (0, 1, 1, 0): 1 / 4,
    (1, 0, 0, 1): 1 / 4,
}


def test_local_precompute_sources():
    # For XOR_DIST every singleton marginal is uniform (P=0.5, h=1 bit) and
    # every pairwise/triple marginal is uniform over 4 equally likely
    # combinations (P=0.25, h=2 bits) -- and the local entropy of the full
    # joint source is always -log2(P(state)), regardless of structure.
    xor_sources = local_precompute_sources(XOR_DIST)
    assert xor_sources[(0, 0, 0)] == pytest.approx(
        {
            (0,): 1.0,
            (1,): 1.0,
            (2,): 1.0,
            (0, 1): 2.0,
            (0, 2): 2.0,
            (1, 2): 2.0,
            (0, 1, 2): 2.0,
        },
        abs=pytest_abs,
    )
    for state, prob in XOR_DIST.items():
        assert xor_sources[state][(0, 1, 2)] == pytest.approx(
            -np.log2(prob), abs=pytest_abs
        )

    # In a fully redundant (giant bit) system every subset of variables is
    # as informative as any other, so every source has the same local
    # entropy in a given state.
    giant_bit_sources = local_precompute_sources(GIANT_BIT)
    for state in GIANT_BIT:
        assert set(giant_bit_sources[state].values()) == {1.0}


def test_hmin_discrete_redundancy():
    sources = local_precompute_sources(XOR_DIST)
    state = (0, 0, 0)

    assert hmin_discrete_redundancy(
        atom=((0,),), state=state, sources=sources, joint_distribution=XOR_DIST
    ) == pytest.approx(1.0, abs=pytest_abs)

    # h((0,)) = 1.0, h((0,1)) = 2.0 -> min is 1.0
    assert hmin_discrete_redundancy(
        atom=((0,), (0, 1)), state=state, sources=sources, joint_distribution=XOR_DIST
    ) == pytest.approx(1.0, abs=pytest_abs)

    # Every source is equally informative in a giant bit.
    gb_sources = local_precompute_sources(GIANT_BIT)
    assert hmin_discrete_redundancy(
        atom=((0,), (1,)), state=state, sources=gb_sources, joint_distribution=GIANT_BIT
    ) == pytest.approx(1.0, abs=pytest_abs)


def test_hmin_discrete_redundancy_zero_probability_state():
    sources = local_precompute_sources(XOR_DIST)
    dist_with_zero = dict(XOR_DIST)
    dist_with_zero[(0, 0, 1)] = 0.0

    result = hmin_discrete_redundancy(
        atom=((0,),), state=(0, 0, 1), sources=sources, joint_distribution=dist_with_zero
    )
    assert result == 0


def test_hsx_discrete_redundancy():
    state = (0, 0, 0)

    # P(X0=0) = 0.5 -> -log2(0.5) = 1.0
    assert hsx_discrete_redundancy(
        atom=((0,),), state=state, joint_distribution=XOR_DIST
    ) == pytest.approx(1.0, abs=pytest_abs)

    # P(X0=0, X1=0) = 0.25 -> -log2(0.25) = 2.0
    assert hsx_discrete_redundancy(
        atom=((0, 1),), state=state, joint_distribution=XOR_DIST
    ) == pytest.approx(2.0, abs=pytest_abs)

    # States matching X0=0 are (0,0,0),(0,1,1); states matching X1=0 are
    # (0,0,0),(1,1,0). Union has probability mass 3/4.
    result = hsx_discrete_redundancy(
        atom=((0,), (1,)), state=state, joint_distribution=XOR_DIST
    )
    assert result == pytest.approx(-np.log2(3 / 4), abs=pytest_abs)


def test_hsx_discrete_redundancy_zero_probability_state():
    dist_with_zero = dict(XOR_DIST)
    dist_with_zero[(0, 0, 1)] = 0.0

    result = hsx_discrete_redundancy(
        atom=((0,),), state=(0, 0, 1), joint_distribution=dist_with_zero
    )
    assert result == 0.0


def test_mmi_discrete_redundancy_pid():
    # For XOR, X0 and X1 are each marginally independent of the target X2.
    mmi_xor = mmi_discrete_redundancy(
        atom=((0,), (1,)), inputs=(0, 1), target=(2,), joint_distribution=XOR_DIST
    )
    mi0 = mutual_information((0,), (2,), XOR_DIST)[1]
    mi1 = mutual_information((1,), (2,), XOR_DIST)[1]
    assert mmi_xor == pytest.approx(min(mi0, mi1), abs=pytest_abs)
    assert mmi_xor == pytest.approx(0.0, abs=pytest_abs)

    # Every input is fully informative about the target in a giant bit.
    mmi_gb = mmi_discrete_redundancy(
        atom=((0,), (1,)), inputs=(0, 1), target=(2,), joint_distribution=GIANT_BIT
    )
    assert mmi_gb == pytest.approx(1.0, abs=pytest_abs)


def test_mmi_discrete_redundancy_phiid_takes_minimum_over_all_pairs():
    inputs = (0, 1)
    target = (2, 3)
    atom = (((0,), (1,)), ((0,), (1,)))

    mmi = mmi_discrete_redundancy(
        atom=atom,
        inputs=inputs,
        target=target,
        joint_distribution=DISINTEGRATED_SYSTEM,
        single_target_flag=False,
    )

    manual_min = min(
        mutual_information(ix, iy, DISINTEGRATED_SYSTEM)[1]
        for ix in [(0,), (1,)]
        for iy in [(2,), (3,)]
    )
    assert mmi == pytest.approx(manual_min, abs=pytest_abs)


def test_ipm_discrete_redundancy_matches_manual_formula_pid():
    sources = local_precompute_sources(XOR_DIST)
    state = (0, 0, 0)
    atom = ((0,), (1,))
    inputs = (0, 1)
    target = (2,)

    mn_inputs = hmin_discrete_redundancy(
        atom=atom, state=state, sources=sources, joint_distribution=XOR_DIST
    )
    h_target = sources[state][target]
    mn_joint = hmin_discrete_redundancy(
        atom=((0, 2), (1, 2)), state=state, sources=sources, joint_distribution=XOR_DIST
    )
    expected = mn_inputs + h_target - mn_joint

    result = ipm_discrete_redundancy(
        atom=atom,
        inputs=inputs,
        target=target,
        state=state,
        sources=sources,
        joint_distribution=XOR_DIST,
    )
    assert result == pytest.approx(expected, abs=pytest_abs)


def test_ipm_discrete_redundancy_matches_manual_formula_phiid():
    inputs = (0, 1)
    target = (2, 3)
    atom = (((0,), (1,)), ((0,), (1,)))
    state = (0, 0, 1, 1)
    sources = local_precompute_sources(DISINTEGRATED_SYSTEM)

    atom_inputs = ((0,), (1,))
    atom_target = ((2,), (3,))
    mn_inputs = hmin_discrete_redundancy(
        atom=atom_inputs,
        state=state,
        sources=sources,
        joint_distribution=DISINTEGRATED_SYSTEM,
    )
    mn_target = hmin_discrete_redundancy(
        atom=atom_target,
        state=state,
        sources=sources,
        joint_distribution=DISINTEGRATED_SYSTEM,
    )
    mn_joint = min(
        sources[state][tuple(set(s1 + s2))]
        for s1 in atom_inputs
        for s2 in atom_target
    )
    expected = mn_inputs + mn_target - mn_joint

    result = ipm_discrete_redundancy(
        atom=atom,
        inputs=inputs,
        target=target,
        state=state,
        sources=sources,
        joint_distribution=DISINTEGRATED_SYSTEM,
        single_target_flag=False,
    )
    assert result == pytest.approx(expected, abs=pytest_abs)


def test_isx_discrete_redundancy_matches_manual_formula_pid():
    sources = local_precompute_sources(XOR_DIST)
    state = (0, 0, 0)
    atom = ((0,), (1,))
    inputs = (0, 1)
    target = (2,)

    sx_inputs = hsx_discrete_redundancy(
        atom=atom, state=state, joint_distribution=XOR_DIST
    )
    h_target = sources[state][target]
    sx_joint = hsx_discrete_redundancy(
        atom=((0, 2), (1, 2)), state=state, joint_distribution=XOR_DIST
    )
    expected = sx_inputs + h_target - sx_joint

    result = isx_discrete_redundancy(
        atom=atom,
        inputs=inputs,
        target=target,
        state=state,
        sources=sources,
        joint_distribution=XOR_DIST,
    )
    assert result == pytest.approx(expected, abs=pytest_abs)


def test_isx_discrete_redundancy_matches_manual_formula_phiid():
    inputs = (0, 1)
    target = (2, 3)
    atom = (((0,), (1,)), ((0,), (1,)))
    state = (0, 0, 1, 1)
    sources = local_precompute_sources(DISINTEGRATED_SYSTEM)

    atom_inputs = ((0,), (1,))
    atom_target = ((2,), (3,))
    sx_inputs = hsx_discrete_redundancy(
        atom=atom_inputs, state=state, joint_distribution=DISINTEGRATED_SYSTEM
    )
    sx_target = hsx_discrete_redundancy(
        atom=atom_target, state=state, joint_distribution=DISINTEGRATED_SYSTEM
    )
    atom_joint = tuple(
        tuple(set(s1 + s2)) for s1 in atom_inputs for s2 in atom_target
    )
    sx_joint = hsx_discrete_redundancy(
        atom=atom_joint, state=state, joint_distribution=DISINTEGRATED_SYSTEM
    )
    expected = sx_inputs + sx_target - sx_joint

    result = isx_discrete_redundancy(
        atom=atom,
        inputs=inputs,
        target=target,
        state=state,
        sources=sources,
        joint_distribution=DISINTEGRATED_SYSTEM,
        single_target_flag=False,
    )
    assert result == pytest.approx(expected, abs=pytest_abs)


def test_representational_complexity():
    # min(len) for ((0,),) is 1, for ((0,1),) is 2.
    # rc = (0.5*1 + 0.5*2) / (0.5 + 0.5) = 1.5
    avg = {((0,),): 0.5, ((0, 1),): 0.5}
    assert representational_complexity(avg) == pytest.approx(1.5, abs=pytest_abs)

    # A single atom mixing a unary and a binary source, so min and max
    # comparators disagree -- and np.min/np.max must agree with the builtins.
    mixed = {((0,), (1, 2)): 1.0}
    assert representational_complexity(mixed, comparator=min) == pytest.approx(
        1.0, abs=pytest_abs
    )
    assert representational_complexity(mixed, comparator=max) == pytest.approx(
        2.0, abs=pytest_abs
    )
    assert representational_complexity(
        mixed, comparator=np.min
    ) == pytest.approx(representational_complexity(mixed, comparator=min), abs=pytest_abs)
    assert representational_complexity(
        mixed, comparator=np.max
    ) == pytest.approx(representational_complexity(mixed, comparator=max), abs=pytest_abs)


def test_representational_complexity_rejects_invalid_comparator():
    with pytest.raises(AssertionError):
        representational_complexity({((0,),): 1.0}, comparator=sum)

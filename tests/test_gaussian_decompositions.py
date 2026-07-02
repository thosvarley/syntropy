import pytest
import numpy as np

from syntropy.gaussian.decompositions import (
    unpack_atom,
    local_precompute_sources,
    hmin_differential_redundancy,
    mmi_differential_redundancy,
    ipm_differential_redundancy,
    representational_complexity,
    partial_information_decomposition as pid,
)
from syntropy.gaussian.shannon import local_differential_entropy, mutual_information

pytest_abs = 1e-6

rng = np.random.default_rng(0)
_N = 6
_A = rng.normal(size=(_N, _N))
COV = _A @ _A.T + _N * np.eye(_N)
DATA = rng.multivariate_normal(mean=np.zeros(_N), cov=COV, size=3000).T


def test_unpack_atom():
    assert unpack_atom(((0,), (1,))) == {0, 1}
    assert unpack_atom(((0,), (1, 2))) == {0, 1, 2}
    assert unpack_atom(((0, 1), (1, 2))) == {0, 1, 2}  # deduplicates


def test_local_precompute_sources():
    N = 3
    data, cov = DATA[:N, :], COV[:N, :N]

    sources = local_precompute_sources(data, cov=cov)

    assert set(sources.keys()) == {(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)}
    for source, values in sources.items():
        direct = local_differential_entropy(
            data[list(source), :], cov[np.ix_(source, source)]
        )
        assert np.allclose(values, direct, atol=pytest_abs)


def test_hmin_differential_redundancy():
    N = 3
    sources = local_precompute_sources(DATA[:N, :], cov=COV[:N, :N])

    result = hmin_differential_redundancy(atom=((0,), (1,)), sources=sources)
    assert np.allclose(result, np.minimum(sources[(0,)], sources[(1,)]), atol=pytest_abs)

    # A single-source atom is the identity.
    single = hmin_differential_redundancy(atom=((0, 1),), sources=sources)
    assert np.allclose(single, sources[(0, 1)], atol=pytest_abs)


def test_mmi_differential_redundancy_pid():
    inputs, target = (0, 1), (2,)

    result = mmi_differential_redundancy(
        atom=((0,), (1,)), inputs=inputs, target=target, cov=COV
    )
    mi0 = mutual_information(idxs_x=(0,), idxs_y=target, cov=COV)
    mi1 = mutual_information(idxs_x=(1,), idxs_y=target, cov=COV)
    assert result == pytest.approx(min(mi0, mi1), abs=pytest_abs)

    # Regression test: mmi_differential_redundancy must resolve atom-local
    # lattice indices through the `inputs` tuple before indexing into cov,
    # exactly like the discrete analogue (mmi_discrete_redundancy) does. It
    # previously used the atom's local indices directly against the full
    # covariance matrix, silently computing the MI of the wrong variables
    # whenever `inputs` was not a zero-based contiguous prefix.
    remapped_inputs, remapped_target = (2, 3), (4,)
    remapped = mmi_differential_redundancy(
        atom=((0,),), inputs=remapped_inputs, target=remapped_target, cov=COV
    )
    expected = mutual_information(idxs_x=(2,), idxs_y=remapped_target, cov=COV)
    wrong = mutual_information(idxs_x=(0,), idxs_y=remapped_target, cov=COV)
    assert remapped == pytest.approx(expected, abs=pytest_abs)
    assert remapped != pytest.approx(wrong, abs=pytest_abs)


def test_mmi_differential_redundancy_phiid():
    inputs, target = (1, 2), (4, 5)
    atom = (((0,), (1,)), ((0,), (1,)))

    result = mmi_differential_redundancy(
        atom=atom, inputs=inputs, target=target, cov=COV, single_target_flag=False
    )
    manual = min(
        mutual_information(idxs_x=ix, idxs_y=iy, cov=COV)
        for ix in [(1,), (2,)]
        for iy in [(4,), (5,)]
    )
    assert result == pytest.approx(manual, abs=pytest_abs)

    # Same remapping regression as the PID case, in the Phi-ID branch.
    remapped_inputs, remapped_target = (2, 3), (4, 5)
    remapped_atom = (((0,),), ((0,),))
    remapped = mmi_differential_redundancy(
        atom=remapped_atom,
        inputs=remapped_inputs,
        target=remapped_target,
        cov=COV,
        single_target_flag=False,
    )
    expected = mutual_information(idxs_x=(2,), idxs_y=(4,), cov=COV)
    assert remapped == pytest.approx(expected, abs=pytest_abs)


def test_mmi_differential_redundancy_invariant_to_variable_permutation():
    # End-to-end check via the public API: the PID result for a given set
    # of (input, target) variables should not depend on where those
    # variables happen to sit in the covariance matrix.
    avg_a = pid(inputs=(2, 3), target=(4,), data=DATA, cov=COV, redundancy_function="mmi")

    perm = [2, 3, 4, 0, 1, 5]
    cov_p = COV[np.ix_(perm, perm)]
    data_p = DATA[perm, :]
    avg_b = pid(inputs=(0, 1), target=(2,), data=data_p, cov=cov_p, redundancy_function="mmi")

    for atom in avg_a:
        assert avg_a[atom] == pytest.approx(avg_b[atom], abs=pytest_abs)


def test_ipm_differential_redundancy_matches_manual_formula_pid():
    inputs = (0, 1)
    target = (2,)
    atom = ((0,), (1,))
    sources = local_precompute_sources(DATA[:3, :], cov=COV[:3, :3])

    result = ipm_differential_redundancy(
        atom=atom, inputs=inputs, target=target, sources=sources
    )

    target_ = (2,)
    mn_inputs = hmin_differential_redundancy(atom=atom, sources=sources)
    mn_target = sources[target_]
    mn_joint = hmin_differential_redundancy(
        tuple(x + target_ for x in atom), sources=sources
    )
    expected = mn_inputs + mn_target - mn_joint

    assert np.allclose(result, expected, atol=pytest_abs)


def test_ipm_differential_redundancy_matches_manual_formula_phiid():
    inputs = (1, 2)
    target = (4, 5)
    joint = inputs + target
    sources = local_precompute_sources(DATA[list(joint), :], COV[np.ix_(joint, joint)])

    num_inputs, num_target = len(inputs), len(target)
    target_ = tuple(range(num_inputs, num_inputs + num_target))
    atom_inputs = ((0,), (1,))
    atom_targets = tuple(tuple(target_[i] for i in x) for x in ((0,), (1,)))
    atom = (atom_inputs, ((0,), (1,)))

    result = ipm_differential_redundancy(
        atom=atom,
        inputs=inputs,
        target=target,
        sources=sources,
        single_target_flag=False,
    )

    mn_inputs = hmin_differential_redundancy(atom=atom_inputs, sources=sources)
    mn_target = hmin_differential_redundancy(atom=atom_targets, sources=sources)
    mn_joint = np.repeat(np.inf, mn_inputs.shape[1])
    for s1 in atom_inputs:
        for s2 in atom_targets:
            mn_joint = np.minimum(mn_joint, sources[s1 + s2])
    expected = mn_inputs + mn_target - mn_joint

    assert np.allclose(result, expected, atol=pytest_abs)


def test_representational_complexity():
    avg = {((0,),): 0.5, ((0, 1),): 0.5}
    assert representational_complexity(avg) == pytest.approx(1.5, abs=pytest_abs)

    # A single atom mixing a unary and a binary source, so min and max
    # comparators disagree -- and np.min/np.max must agree with the builtins
    # (previously they received a bare generator expression, which they
    # don't consume the way the builtins do, causing a silent no-op
    # followed by a TypeError on multiplication).
    mixed = {((0,), (1, 2)): 1.0}
    for comparator, expected in ((min, 1.0), (np.min, 1.0), (max, 2.0), (np.max, 2.0)):
        assert representational_complexity(mixed, comparator=comparator) == pytest.approx(
            expected, abs=pytest_abs
        )


def test_representational_complexity_rejects_invalid_comparator():
    with pytest.raises(AssertionError):
        representational_complexity({((0,),): 1.0}, comparator=sum)

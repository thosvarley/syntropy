import pytest
import numpy as np
from scipy.spatial import cKDTree

from syntropy.knn.utils import build_tree_and_get_distances, get_counts_from_tree

pytest_abs = 1e-9


def test_build_tree_and_get_distances_hand_computed():
    # Points 0, 1, 2, 4 on a line. Nearest-neighbor distances (excluding
    # self) are 1, 1, 1, 2 respectively.
    data = np.array([[0.0, 1.0, 2.0, 4.0]])

    tree, distances, indices = build_tree_and_get_distances(data, k=1)

    assert isinstance(tree, cKDTree)
    assert distances.shape == (4, 2)
    assert indices.shape == (4, 2)

    # First column is always the point itself, at distance 0.
    assert np.allclose(distances[:, 0], 0.0)
    assert np.array_equal(indices[:, 0], np.arange(4))

    # Second column is the true nearest-neighbor distance.
    assert np.allclose(distances[:, -1], [1.0, 1.0, 1.0, 2.0])

    # Unambiguous (non-tied) nearest neighbors.
    assert indices[0, -1] == 1
    assert indices[3, -1] == 2


def test_build_tree_and_get_distances_matches_scipy_directly():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3, 50))
    k = 4

    _, distances, indices = build_tree_and_get_distances(data, k=k, p=2)

    ref_tree = cKDTree(data.T)
    ref_distances, ref_indices = ref_tree.query(data.T, k=k + 1, p=2)

    assert np.allclose(distances, ref_distances, atol=pytest_abs)
    assert np.array_equal(indices, ref_indices)


def test_norm_order_affects_both_functions():
    # (0, 0) and (3, 4): Euclidean distance is 5, Chebyshev distance is 4.
    data = np.array([[0.0, 3.0], [0.0, 4.0]])

    _, dist_euclidean, _ = build_tree_and_get_distances(data, k=1, p=2)
    _, dist_chebyshev, _ = build_tree_and_get_distances(data, k=1, p=np.inf)
    assert dist_euclidean[0, -1] == pytest.approx(5.0, abs=pytest_abs)
    assert dist_chebyshev[0, -1] == pytest.approx(4.0, abs=pytest_abs)

    tree = cKDTree(data.T)
    eps = np.full(2, 4.5)
    counts_euclidean = get_counts_from_tree(tree, data, eps, p=2)
    counts_chebyshev = get_counts_from_tree(tree, data, eps, p=np.inf)
    assert counts_euclidean[0] == 0  # 5.0 > 4.5, excluded
    assert counts_chebyshev[0] == 1  # 4.0 < 4.5, included


def test_get_counts_from_tree_hand_computed():
    # Points 0, 1, 2, 5 on a line.
    data = np.array([[0.0, 1.0, 2.0, 5.0]])
    tree = cKDTree(data.T)

    # Radius 1.5 excludes ties at distance exactly 1 or 2 from the
    # boundary, so strict vs. non-strict does not matter here.
    counts = get_counts_from_tree(tree, data, eps=np.full(4, 1.5))
    assert np.array_equal(counts, [1, 2, 1, 0])

    # A single point can never have any neighbors, however large the radius.
    single = np.array([[0.0]])
    single_tree = cKDTree(single.T)
    assert get_counts_from_tree(single_tree, single, eps=np.array([100.0]))[0] == 0


def test_get_counts_from_tree_strict_excludes_boundary_points():
    data = np.array([[0.0, 1.0, 2.0, 5.0]])
    tree = cKDTree(data.T)

    # Radius exactly 1.0 matches real inter-point distances, so this
    # isolates the strict "<" vs. non-strict "<=" boundary behavior.
    eps = np.full(4, 1.0)

    strict_counts = get_counts_from_tree(tree, data, eps, strict=True)
    nonstrict_counts = get_counts_from_tree(tree, data, eps, strict=False)

    assert np.array_equal(strict_counts, [0, 0, 0, 0])
    assert np.array_equal(nonstrict_counts, [1, 2, 1, 0])

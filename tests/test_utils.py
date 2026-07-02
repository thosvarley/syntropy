import numpy as np

from syntropy.utils import check_idxs, make_powerset


def test_check_idxs():
    data = np.zeros((4, 100))
    assert check_idxs(None, data) == (0, 1, 2, 3)
    assert check_idxs((1, 3), data) == (1, 3)


def test_make_powerset():
    result = set(make_powerset([1, 2, 3]))
    expected = {(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)}
    assert result == expected
    assert set(make_powerset([])) == {()}

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Iterable
import itertools as it


def check_idxs(
    idxs: tuple[int, ...] | None, data: NDArray[np.floating]
) -> tuple[int, ...]:
    """
    Normalizes an optional channel-index selection against a data array.

    Parameters
    ----------
    idxs : tuple[int, ...] | None
        Indices of channels to use. If None, all channels are used.
    data : NDArray[np.floating]
        Data array of shape (n_variables, n_samples), used only to
        determine the total number of channels when idxs is None.

    Returns
    -------
    tuple[int, ...]
        idxs itself if given, otherwise a tuple of every channel index in
        data (0, 1, ..., n_variables - 1).

    """
    if idxs is None:
        idxs_ = tuple(i for i in range(data.shape[0]))
    else:
        idxs_ = idxs

    return idxs_


def make_powerset(iterable: Iterable) -> it.chain:
    """
    Computes the powerset of a collection of elements.

    .. math::
        \\mathcal{P}(\\{X_1,X_2,X_3\\}) \\to (\\{\\}, \\{X_1\\}, \\{X_2\\}, \\{X_3\\}, \\{X_1,X_2\\}, \\{X_1,X_3\\}, \\{X_1,X_2,X_3\\} )

    Parameters
    ----------
    iterable : Iterable
        Any iterable collection of elements.

    Returns
    -------
    itertools.chain
        An iterator over every subset of the input, from the empty tuple
        up to the full set of elements.

    """
    xs: list = list(iterable)
    # note we return an iterator rather than a list
    return it.chain.from_iterable(it.combinations(xs, n) for n in range(len(xs) + 1))

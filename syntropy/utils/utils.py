import numpy as np 
from numpy.typing import NDArray
from typing import Iterable
import itertools as it 
from __future__ import annotations

def check_idxs(idxs: tuple[int, ...] | None, data: NDArray[np.floating]) -> tuple[int, ...]:
    
    if idxs is None:
        idxs_ = tuple(i for i in range(data.shape[0]))
    else:
        idxs_ = idxs 

    return idxs_

def make_powerset(iterable: Iterable):
    """
    Computes the powerset of a collection of elements.

    .. math::
        \\mathcal{P}(\\{X_1,X_2,X_3\\}) \\to (\\{\\}, \\{X_1\\}, \\{X_2\\}, \\{X_3\\}, \\{X_1,X_2\\}, \\{X_1,X_3\\}, \\{X_1,X_2,X_3\\} )

    """
    xs: list = list(iterable)
    # note we return an iterator rather than a list
    return it.chain.from_iterable(it.combinations(xs, n) for n in range(len(xs) + 1))

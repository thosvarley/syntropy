import numpy as np 
from numpy.typing import NDArray

def check_idxs(idxs: tuple[int, ...] | None, data: NDArray[np.floating]) -> tuple[int, ...]:
    
    if idxs is None:
        idxs_ = tuple(i for i in range(data.shape[0]))
    else:
        idxs_ = idxs 

    return idxs_

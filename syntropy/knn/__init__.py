from .shannon import (
    differential_entropy,
    mutual_information,
)

from .multivariate_mi import (
    total_correlation,
    dual_total_correlation,
    s_information,
    o_information
)

__all__ = [
    "differential_entropy",
    "mutual_information",
    "total_correlation",
    "dual_total_correlation",
    "s_information",
    "o_information",
]

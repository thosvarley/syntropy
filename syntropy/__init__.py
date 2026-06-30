"""
Syntropy: A package for multivariate information theory on discrete and continuous random variables.

This package provides tools for computing information-theoretic quantities
including entropy, mutual information, and various multivariate decompositions
for discrete, Gaussian, and general continuous distributions.
"""

__version__ = "0.0.1"
__author__ = "Thomas F. Varley"
__email__ = "tfvarley@uvm.edu"

from . import discrete
from . import gaussian
from . import knn
from . import mixed
from . import lattices


# `neural` depends on torch + nflows (heavy, optional). Import it lazily so the
# core library works without them; `syntropy.neural` triggers the import only
# when actually accessed.
def __getattr__(name):
    if name == "neural":
        import importlib

        return importlib.import_module(".neural", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "discrete",
    "gaussian",
    "knn",
    "neural",
    "mixed",
    "lattices",
    "__version__",
    "__author__",
    "__email__",
]

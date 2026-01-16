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
from . import neural
from . import mixed
from . import lattices

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

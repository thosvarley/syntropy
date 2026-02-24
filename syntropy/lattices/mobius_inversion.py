import networkx as nx
import numpy as np
from typing import Callable, Any
from copy import copy

def mobius_inversion(
    redundancy_func: Callable, lattice: nx.DiGraph, kwargs: dict[str, Any]
) -> nx.DiGraph:
    
    """
    Computes the Mobius inversion on an arbitrary lattice, given an arbitrary redundancy function

    Parameters
    ----------
    redundancy_func : Callable
        The redundancy function.
        
    lattice : nx.DiGraph
        The partial information lattice (can be single-target or multi-target).
        
    kwargs : dict[str, Any]
        Whatever aguments the redundancy function needs. 
        

    Returns
    -------
    nx.DiGraph
        A copy of the lattice, populated with all the right partial information atoms. 
        

    """
    lattice = lattice.copy()
    layers = list({lattice.nodes[node]["distance_from_top"] for node in lattice.nodes})

    for layer in layers[::-1]: # Why did I do distance from top? 
        atoms = [
            node
            for node in lattice.nodes
            if lattice.nodes[node]["distance_from_top"] == layer
        ]

        for atom in atoms:
            lattice.nodes[atom]["total_information"] = redundancy_func(
                atom=atom, **kwargs
            )
            lattice.nodes[atom]["pi"] = copy(lattice.nodes[atom]["total_information"])

            for d in lattice.nodes[atom]["descendants"]:
                lattice.nodes[atom]["pi"] -= lattice.nodes[d]["pi"]

    return lattice

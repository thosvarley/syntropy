import pytest
import networkx as nx

from syntropy.lattices import load_lattice, mobius_inversion

pytest_abs = 1e-6


def test_load_lattice_two_inputs_structure():
    g = load_lattice(num_inputs=2)

    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == 4

    # The redundancy atom (both sources) is the unique top.
    assert g.nodes[((0, 1),)]["top"] is True
    assert g.nodes[((0, 1),)]["distance_from_top"] == 0

    # The synergy atom (each source alone) is the unique bottom, with no
    # descendants.
    assert g.nodes[((0,), (1,))]["bottom"] is True
    assert g.nodes[((0,), (1,))]["descendants"] == set()

    # The two unique-information atoms sit in between.
    assert g.nodes[((0,),)]["distance_from_top"] == 1
    assert g.nodes[((1,),)]["distance_from_top"] == 1


def test_mobius_inversion_hand_computed():
    # A hand-derivable example: assign each atom a known "total_information"
    # value via a lookup table, and verify the Mobius-inverted partial
    # information (pi) matches what you get by manually subtracting
    # descendant pi values.
    lattice = load_lattice(num_inputs=2)

    total_info = {
        ((0, 1),): 3.0,
        ((0,),): 2.0,
        ((1,),): 2.0,
        ((0,), (1,)): 1.0,
    }

    def redundancy_func(atom, lookup):
        return lookup[atom]

    result = mobius_inversion(
        redundancy_func=redundancy_func, lattice=lattice, kwargs={"lookup": total_info}
    )

    # Bottom atom has no descendants, so pi == total_information.
    assert result.nodes[((0,), (1,))]["pi"] == pytest.approx(1.0, abs=pytest_abs)

    # Middle atoms: pi = total_information - pi(bottom).
    assert result.nodes[((0,),)]["pi"] == pytest.approx(2.0 - 1.0, abs=pytest_abs)
    assert result.nodes[((1,),)]["pi"] == pytest.approx(2.0 - 1.0, abs=pytest_abs)

    # Top atom: pi = total_information - sum(pi of all descendants).
    assert result.nodes[((0, 1),)]["pi"] == pytest.approx(
        3.0 - 1.0 - 1.0 - 1.0, abs=pytest_abs
    )

    # General Mobius-inversion invariant: the sum of all pi values equals
    # the total_information at the top of the lattice.
    total_pi = sum(result.nodes[node]["pi"] for node in result.nodes)
    assert total_pi == pytest.approx(
        result.nodes[((0, 1),)]["total_information"], abs=pytest_abs
    )

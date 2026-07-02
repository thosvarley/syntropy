import networkx as nx


Atom = tuple[tuple[int, ...], ...]

ATTRS = {
    "top": False,
    "bottom": False,
    "descendants": set(),
    "pi": 0.0,
    "total_information": 0.0,
    "distance_from_top": None,
}
# %%


def construct_product_lattice(
    lattice_1: nx.DiGraph, lattice_2: nx.DiGraph
) -> nx.DiGraph:
    """
    Constructs the product lattice of two single-target partial information lattices. 

    Parameters
    ----------
    lattice_1 : nx.DiGraph()
        The first single-target redundancy lattice.
    lattice_2 : nx.DiGraph()
        The second single-target redundancy lattice.

    Returns
    -------
    nx.DiGraph()
        The product lattice. 

    References
    ----------
    Mediano, P. A. M., Rosas, F., Carhart-Harris, R. L., Seth, A. K., & Barrett, A. B. (2019).
    Beyond Integrated Information: A Taxonomy of Information Dynamics Phenomena.
    arXiv:1909.02297 [physics, q-bio].
    https://arxiv.org/abs/1909.02297

    """
    nodes: tuple[Atom, Atom] = [
        (x, y) for x in lattice_1.nodes for y in lattice_2.nodes
    ]

    G = nx.DiGraph()
    G.add_nodes_from(nodes, **ATTRS)

    for node in G.nodes:
        x, y = node

        if (lattice_1.nodes[x]["bottom"] is True) and (
            lattice_2.nodes[y]["bottom"] is True
        ):
            G.nodes[node]["bottom"] = True
        if (lattice_1.nodes[x]["top"] is True) and (lattice_2.nodes[y]["top"] is True):
            G.nodes[node]["top"] = True
            G.nodes[node]["distance_from_top"] = 0
            top = node

        desc_1 = lattice_1.nodes[x]["descendants"].union({x})
        desc_2 = lattice_2.nodes[y]["descendants"].union({y})

        # a -> b \preceq a' -> b' iff a \preceq a' and b \preceq b'
        G.nodes[node]["descendants"] = {
            p for p in G.nodes if (p[0] in desc_1) and (p[1] in desc_2)
        }.difference({node})

    # Only store those edges that connect nodes to their immediate predecessor/successor
    for p in G.nodes:
        D_p = G.nodes[p]["descendants"]
        # Start with all descendants as candidates for immediate successors
        children = set(D_p)

        # Remove any node that is a descendant of another descendant
        # (i.e., keep only the "maximal" elements under p)
        for d in D_p:
            children.difference_update(G.nodes[d]["descendants"])

        for c in children:
            G.add_edge(p, c)

    # Get distances from top for each node (importan for Mobius inversion)
    shortest_paths = nx.shortest_path_length(source=top, G=G)

    for node in G.nodes:
        G.nodes[node]["distance_from_top"] = shortest_paths[node]

    return G


# %%

# lattices = [LATTICE_2, LATTICE_3, LATTICE_4]
# sizes = [2, 3, 4]
#
# for i in range(3):
#     for j in range(3):
#         size_i = str(sizes[i])
#         size_j = str(sizes[j])
#
#         size_str = "".join((size_i, size_j))
#
#         G = construct_product_lattice(lattices[i], lattices[j])
#
#         # Save graph object to a file
#         with open(f'pi_lattice_{size_str}.pickle', 'wb') as f:
#             pickle.dump(G, f)

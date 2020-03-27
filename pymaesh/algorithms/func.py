import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as gs
from ..utils.numpy import cantor_pairing
from ..backend import Graph

def nonexisting_edges(graph, seed=None):
    """
    Returns a generator iterating randomly over the non-existing edges of the graph. All returned
    edges will be of the format that the index of the first node is smaller than the index of the
    second node. No self-loops are returned. Note that this generator expects that the user
    samples much fewer edges than the number of nonexisting edges. Otherwise, do not use this
    function.

    Parameters
    ----------
    graph: pymaesh.Graph
        The graph to yield nonexisting edges from.
    seed: int, default: None
        The seed to use for random sampling.

    Returns
    -------
    generator of tuple of (int, int)
        The edges which do not exist in the graph.
    """
    # pylint: disable=no-member
    randomizer = np.random.RandomState(seed)

    sampled = set()
    count = 0
    total = (graph.num_nodes ** 2 - graph.num_nodes) // 2 - graph.num_edges

    while count < total:
        # Choose node
        n = randomizer.choice(graph.num_nodes)
        m = randomizer.choice(graph.num_nodes)

        # No self-loops and no edges that already exist
        if n == m or graph.adjacency[n, m] == 1:
            continue

        # Order node and check if already sampled
        edge = (min(n, m), max(n, m))
        digest = cantor_pairing(*edge)
        if digest in sampled:
            continue

        # Set already sampled
        sampled.add(digest)
        count += 1

        yield edge


def random_subgraph(graph, size, seed=None):
    """
    Computes a random subgraph of the given graph with the specified fraction of edges. The
    resulting subgraph is guaranteed to be connected.

    Note
    ----
    Implementation modified from https://bit.ly/2wdQsrz.

    Parameters
    ----------
    graph: pymaesh.Graph
        The graph to sample a subgrah from.
    size: float
        The size of the graphs. The number of edges for the subgraph is given as
        `size * graph.num_edges`.
    seed: int, default: None
        The seed to use for random sampling.

    Returns
    -------
    pymaesh.Graph
        The subgraph with number of edges proportional to `size`.
    numpy.ndarray [N, 2]
        The edges which are not in the subgraph but in the original graph. The edges are always
        given with the node having the smaller index first. Use `graph.utils.symmetric_edge_list`
        to obtain 'directed' edges.
    """
    assert 0 < size < 1, "Invalid size. Must be in the range (0, 1)."
    assert gs.connected_components(graph.adjacency)[0] == 1, \
        "Given graph has more than one connected component."

    # pylint: disable=no-member
    randomizer = np.random.RandomState(seed)

    # 0) Setup
    num_edges = int(graph.num_edges * size)

    # 1) Ensure that resulting graph is connected
    mst = gs.minimum_spanning_tree(graph.adjacency)
    mst = mst + mst.T
    mst[mst > 1] = 1
    ensure_edges = np.array(sp.triu(mst).nonzero()).T

    if ensure_edges.shape[0] > num_edges:
        raise ValueError(
            f"Size {size} is too small. Subgraph cannot be fully connected."
        )

    # 2) Sample from the remaining edges
    A_prime = graph.adjacency - mst
    remaining_edges = np.array(sp.triu(A_prime).nonzero()).T
    edge_perturbation = randomizer.permutation(remaining_edges.shape[0])
    edge_choices = edge_perturbation[:num_edges-ensure_edges.shape[0]]

    # 3) Build adjacency
    add_edges = remaining_edges[edge_choices]
    A = sp.csr_matrix(
        (np.ones(num_edges),
         (np.concatenate([ensure_edges[:, 0], add_edges[:, 0]]),
          np.concatenate([ensure_edges[:, 1], add_edges[:, 1]]))),
        shape=(graph.num_nodes, graph.num_nodes)
    )
    A = A + A.T
    A[A > 1] = 1
    subgraph = Graph(A, graph.features, graph.labels)

    # 4) Find all other edges
    A_diff = graph.adjacency - subgraph.adjacency
    excluded_edges = np.array(sp.triu(A_diff).nonzero()).T

    return subgraph, excluded_edges

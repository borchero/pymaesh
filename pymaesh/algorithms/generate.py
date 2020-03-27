import numpy as np
import scipy.sparse as sp

def erdos_renyi(num_nodes, num_edges, seed=None):
    """
    Generates a totally random graph with the specified number of nodes and edges.

    Parameters
    ----------
    num_nodes: int
        The number of nodes in the graph.
    num_edges: int
        The number of edges in the graph.
    seed: int, default: None
        The seed to use for sampling random edges. If it is None, no particular seed is used.

    Returns
    -------
    scipy.sparse.csr_matrix
        The constructed (symmetric) adjacency matrix.
    """
    # pylint: disable=no-member
    randomizer = np.random.RandomState(seed)

    A = sp.dok_matrix((num_nodes, num_nodes))
    i = 0
    while i < num_edges:
        n, m = (randomizer.choice(num_nodes), randomizer.choice(num_nodes))
        if n == m:
            continue
        n, m = (min(n, m), max(n, m))
        if A[n, m] == 1:
            continue
        A[n, m] = 1
        i += 1

    A = A.tocsr()
    A = A + A.T
    A[A > 1] = 1

    return A

def sbm(communities, edge_probabilities, seed=None):
    """
    Generates a random graph using a stochastic block model.

    Parameters
    ----------
    - communities: list of int [N]
        A list of integers describing the size of each community.
    - edge_probabilities: np.array [N, N]
        Matrix $P$ where index $P_{ij}$ describes the probability of edges from communities $i$ and
        $j$ to be connected. This matrix is supposed to be symmetric.
    - seed: int, default: None
        The seed to use for sampling random edges. If it is None, no particular seed is used.

    Returns
    -------
    scipy.sparse.csr_matrix
        The constructed (symmetric) adjacency matrix.
    """
    assert (edge_probabilities == edge_probabilities.T).all(), \
        "Edge probability matrix is not symmetric."

    # pylint: disable=no-member
    randomizer = np.random.RandomState(seed)

    num_nodes = np.sum(communities)
    community_idx = np.cumsum(communities)

    adjacency = sp.dok_matrix((num_nodes, num_nodes))

    community_i = 0
    for i in range(num_nodes):
        if i == community_idx[community_i]:
            community_i += 1

        community_j = 0
        for j in range(num_nodes):
            if j == community_idx[community_j]:
                community_j += 1

            p = edge_probabilities[community_i, community_j]
            adjacency[i, j] = randomizer.choice(2, p=(1 - p, p))

    A = adjacency.tocsr()
    A = A + A.T
    A[A > 1] = 1

    return A

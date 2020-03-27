import numpy as np
import scipy.sparse as sp

def compute_transition_counts(random_walks, num_nodes, ensure_no_loops=False):
    """
    Computes the transition counts between all pairs of nodes based on the
    provided random walks.

    Parameters
    ----------
    random_walks: numpy.ndarray [W, L] or [W, L, N]
        A set of random walks where W is the number of walks and L is the walks' lengths. Random
        walks must either be given as integers representing node indices or one-hot vectors
        describing the node.
    num_nodes: int
        The number of nodes in the resulting graph. Must be equal to the third dimension of the
        provided random walks if present.
    ensure_no_loops: bool, default: False
        Whether to assert that the transition matrix's diagonal is all zero. If the matrix's
        diagonal contains values, the random walker did not move in every step.

    Returns
    -------
    scipy.sparse.csr_matrix [N, N]
        The symmetric matrix containing the transition counts for all node pairs.
    """
    assert len(random_walks.shape) == 2 or random_walks.shape[-1] == num_nodes, \
        "The dimensionality of the random walks does not match the number " + \
        "of expected nodes."

    # 1) Prepare walks
    # 1.1) Make sure random walks provide node indices
    if len(random_walks.shape) == 3:
        random_walks = np.argmax(random_walks, axis=2)

    # 1.2) Get transitions
    random_walk_transitions = np.array([
        list(zip(random_walks[:, :-1], random_walks[:, 1:]))
    ])[0].transpose(0, 2, 1).reshape(-1, 2)

    # 1.3) Ensure symmetric transitions
    random_walk_transitions = np.vstack([
        random_walk_transitions,
        np.roll(random_walk_transitions, shift=1, axis=1)
    ])

    # 2) Compute transition matrix
    transition_matrix = sp.csr_matrix(
        (np.ones(len(random_walk_transitions)),
         (random_walk_transitions[:, 0], random_walk_transitions[:, 1])),
        shape=(num_nodes, num_nodes)
    )

    if ensure_no_loops:
        assert (np.diag(transition_matrix) == 0).all(), \
            "Diagonal of transition matrix is not all zero."

    return transition_matrix.tocsr()


def assemble_adjacency_matrix(transition_counts, num_edges, inplace=True, seed=None):
    """
    Computes an adjacency matrix for a graph based on the given transition counts and the desired
    number of edges. The resulting adjacency matrix will represent a graph with no singleton nodes
    (however, possibly with multiple connected components).

    Note
    ----
    The strategy is described in *NetGAN: Generating Graphs via Random Walks* (Bojchevski, Shchur,
    Zügner, Günnemann, 2018).

    Parameters
    ----------
    transition_counts: scipy.sparse.csr_matrix [N, N]
        The transition counts (e.g. obtained from random walks) for all pairs of nodes. Must be
        symmetric.
    num_edges: int
        The number of edges the output adjacency matrix should contain.
    inplace: bool, default: True
        Whether the transition_counts matrix may be modified. Otherwise, a copy is performed.
    seed: int, default: None
        The seed to use for generating random values.

    Returns
    -------
    scipy.sparse.csr_matrix
        A binary adjacency matrix containing the desired number of edges. The function tries to
        assemble a matrix with `2 * num_edges` entries. However, if
        `num_edges < transitions_count.shape[0]`, then this cannot be guaranteed.  The diagonal of
        the adjacency matrix is always zero.
    """
    # 1) Setup
    # pylint: disable=no-member
    randomizer = np.random.RandomState(seed)

    # 1.1) Copy if needed
    if not inplace:
        transition_counts = transition_counts.copy()

    # 1.2) Set diagonal to zero
    transition_counts = transition_counts.tolil()
    transition_counts.setdiag(0)

    # 2) Check if the transition matrix can be converted easily
    if len(transition_counts.nonzero()[0]) // 2 <= num_edges:
        transition_counts[transition_counts.nonzero()] = 1
        transition_counts += transition_counts.T
        transition_counts[transition_counts > 1] = 1
        return transition_counts

    # 3) Assemble the adjacency matrix according to paper
    N = transition_counts.shape[0]
    result = sp.dok_matrix((N, N))
    # transition probabilities
    div = transition_counts.sum(axis=0)
    div[div <= 0] = 1
    P = (transition_counts / div).T

    # 3.1) Iterate over nodes in random order to sample one neighbor
    for node in randomizer.permutation(N)[:min(num_edges, N)]:
        # 3.1.1) Skip if no neighbor for the node is present
        if P[node].sum() == 0:
            continue

        # 3.1.2) Sample neighbor according to probabilities
        neighbor = randomizer.choice(N, p=P[node].A1)
        result[node, neighbor] = result[neighbor, node] = 1

    # 3.2) Sample remaining edges
    # 3.2.1) Compute probabilities for drawing
    num_remaining_edges = int(num_edges - result.sum() / 2)
    if num_remaining_edges > 0:
        # equals size of the upper triangular matrix
        num_choices = (N * N + N) // 2
        transition_counts[result.nonzero()] = 0
        P_triu = sp.triu(transition_counts).tocsr()
        P_triu_indices = np.triu_indices_from(transition_counts)
        probabilities = (P_triu / P_triu.sum())[P_triu_indices]

        # 3.2.2) Choose edges
        edges = randomizer.choice(
            num_choices, replace=False, p=probabilities.A1, size=num_remaining_edges
        )

        # 3.2.3) Add edge choices to result
        rows = P_triu_indices[0][edges]
        cols = P_triu_indices[1][edges]
        result[rows, cols] = result[cols, rows] = 1

    return result.tocsr()

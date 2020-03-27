import unittest
import time
import numpy as np
import scipy.sparse as sp
from pymaesh import Graph, BiasedSecondOrderRandomWalker
from pymaesh.algorithms.build import \
    compute_transition_counts, assemble_adjacency_matrix

class TestGraphBuilding(unittest.TestCase):

    def test_inverse(self):
        graph = Graph.erdos_renyi(100, 500)
        walker = BiasedSecondOrderRandomWalker(graph)
        random_walks = walker.sample_walks(100_000, 16, p=1, q=1)
        transition_counts = compute_transition_counts(random_walks, 100)
        assembled_graph = assemble_adjacency_matrix(
            transition_counts, graph.num_edges
        )
        new_graph = Graph(assembled_graph, None, None)
        self.assertEqual(graph, new_graph)

    def test_compute_transition_counts(self):
        for _ in range(3):
            n = np.random.randint(1500, 5000)
            walks = np.random.randint(n, size=(1_000_000, 16))

            expected = _reference_compute_transition_counts(walks, n)
            ours = compute_transition_counts(walks, n)

            self.assertTrue((expected == ours).todense().all())

    def test_assemble_adjacency_matrix(self):
        for _ in range(3):
            n = np.random.randint(1500, 5000)
            walks = np.random.randint(n, size=(1_000_000, 16))
            counts = compute_transition_counts(walks, n)

            m = int(np.random.uniform(1.5, 5.0) * n)

            seed = np.random.choice(1000)
            np.random.seed(seed)
            expected = _reference_assemble_adjacency_matrix(counts, m * 2)
            ours = assemble_adjacency_matrix(counts, m, seed=seed)

            self.assertTrue((expected == ours).all())

    def test_assemble_sparse_adjacency_martrix(self):
        for _ in range(3):
            n = np.random.randint(1500, 5000)
            walks = np.random.randint(n, size=(5000, 16))
            counts = compute_transition_counts(walks, n)

            m = int(np.random.uniform(1.5, 5.0) * n)

            seed = np.random.choice(1000)
            np.random.seed(seed)
            expected = _reference_assemble_adjacency_matrix(counts, m * 2)
            ours = assemble_adjacency_matrix(counts, m, seed=seed)

            self.assertTrue((expected == ours).all())


def _reference_compute_transition_counts(random_walks, N, symmetric=True):
    """
    Copied from https://github.com/danielzuegner/netgan.
    """
    random_walks = np.array(random_walks)
    bigrams = np.array(list(zip(random_walks[:, :-1], random_walks[:, 1:])))
    bigrams = np.transpose(bigrams, [0, 2, 1])
    bigrams = bigrams.reshape([-1, 2])
    if symmetric:
        bigrams = np.row_stack((bigrams, bigrams[:, ::-1]))

    mat = sp.coo_matrix((np.ones(bigrams.shape[0]), (bigrams[:, 0], bigrams[:, 1])),
                        shape=[N, N])
    return mat


def _reference_assemble_adjacency_matrix(scores, n_edges):
    """
    Copied from https://github.com/danielzuegner/netgan.
    """
    def symmetric(directed_adjacency, clip_to_one=True):
        A_symmetric = directed_adjacency + directed_adjacency.T
        if clip_to_one:
            A_symmetric[A_symmetric > 1] = 1
        return A_symmetric

    if  len(scores.nonzero()[0]) < n_edges:
        return symmetric(scores) > 0

    target_g = np.zeros(scores.shape) # initialize target graph
    scores_int = scores.toarray().copy() # internal copy of the scores matrix
    scores_int[np.diag_indices_from(scores_int)] = 0  # set diagonal to zero
    degrees_int = scores_int.sum(0)   # The row sum over the scores.

    N = scores.shape[0]

    for n in np.random.choice(N, replace=False, size=N):
        row = scores_int[n,:].copy()
        if row.sum() == 0:
            continue

        probs = row / row.sum()

        target = np.random.choice(N, p=probs)
        target_g[n, target] = 1
        target_g[target, n] = 1

    diff = np.round((n_edges - target_g.sum())/2)
    if diff > 0:

        triu = np.triu(scores_int)
        triu[target_g > 0] = 0
        triu[np.diag_indices_from(scores_int)] = 0
        triu = triu / triu.sum()

        triu_ixs = np.triu_indices_from(scores_int)
        extra_edges = np.random.choice(triu_ixs[0].shape[0], replace=False, p=triu[triu_ixs], size=int(diff))

        target_g[(triu_ixs[0][extra_edges], triu_ixs[1][extra_edges])] = 1
        target_g[(triu_ixs[1][extra_edges], triu_ixs[0][extra_edges])] = 1

    target_g = symmetric(target_g)
    return target_g


if __name__ == '__main__':
    unittest.main()

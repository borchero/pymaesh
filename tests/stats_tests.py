import copy
import unittest
import time
import numpy as np
from pymaesh import stats, Graph

class TestGraphStats(unittest.TestCase):

    def test_edge_overlap(self):
        lhs = Graph.erdos_renyi(100, 500)
        rhs = copy.deepcopy(lhs)
        
        lhs_nnz = lhs.adjacency.nonzero()
        rhs_zeros_upper = (lhs_nnz[0][:50], lhs_nnz[1][:50])
        rhs_zeros_lower = (lhs_nnz[1][:50], lhs_nnz[0][:50])

        rhs.adjacency[rhs_zeros_upper] = 0
        rhs.adjacency[rhs_zeros_lower] = 0

        print(stats.edge_overlap(lhs, rhs))


if __name__ == '__main__':
    unittest.main()

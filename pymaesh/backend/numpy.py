import functools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as gs
from .base import BaseGraph
from ..algorithms import stats, func
from ..algorithms import generate

class Graph(BaseGraph):
    """
    The graph class represents undirected graphs, optionally with node features and labels.

    The following datatypes are expected:
    * `adjacency_matrix`: numpy.ndarray or scipy.sparse.csr_matrix
    * `feature_matrix`: numpy.ndarray or scipy.sparse.csr_matrix
    * `labels`: numpy.ndarray
    """

    @staticmethod
    def load(file):
        """
        Loads a graph from the specified file.

        Note
        ----
        Implementation adapted from https://github.com/danielzuegner/netgan.

        Parameters
        ----------
        file: str
            The file to load the graph from. The initialization depends on the
            extension of the file name.
        """
        if file.endswith('.npz'):
            with np.load(file, allow_pickle=True) as loader:
                loader = dict(loader)
                if 'arr_0' in loader:
                    loader = dict(loader)['arr_0'].item()
                A = sp.csr_matrix(
                    (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                    shape=loader['adj_shape']
                )

                if 'attr_data' in loader:
                    X = sp.csr_matrix(
                        (loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                        shape=loader['attr_shape']
                    )
                else:
                    X = loader.get('attr')

                Z = loader.get('labels')
                L = loader.get('graph_label')

                A = A + A.T
                A[A > 1] = 1

            return Graph(A, X, Z, L)
        raise ValueError(
            'Graph cannot be loaded if no filename extension is specified. '
            'Currently, valid extensions are [.npz].'
        )

    @staticmethod
    def erdos_renyi(num_nodes, num_edges, seed=None):
        """
        Generates a random Erdos-Renyi graph with the given number of nodes and edges. Refer to
        `pymaesh.algorithms.generate.erdos_renyi` for more information on the generation of the
        adjacency matrix.

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
        pymaesh.Graph
            The graph with the generated adjacency matrix.
        """
        A = generate.erdos_renyi(num_nodes, num_edges, seed=seed)
        return Graph(A)

    @staticmethod
    def sbm(communities, edge_probabilities, seed=None):
        """
        Generates a random SBM graph with the given community sizes and edge probabilities. Refer to
        `pymaesh.algorithms.generate.sbm` for more information on the generation of the adjacency
        matrix.

        Parameters
        ----------
        communities: list of int [N]
            A list of integers describing the size of each community.
        edge_probabilities: np.array [N, N]
            Matrix $P$ where index $P_{ij}$ describes the probability of edges from communities $i$
            and $j$ to be connected. This matrix is supposed to be symmetric.
        seed: int, default: None
            The seed to use for sampling random edges. If it is None, no particular seed is used.

        Returns
        -------
        pymaesh.Graph
            The graph with the generated adjacency matrix.
        """
        A = generate.sbm(communities, edge_probabilities, seed=seed)
        return Graph(A)

    @property
    def num_edges(self):
        return int(self.adjacency.sum()) // 2

    @property
    def num_classes(self):
        return np.max(self.labels) + 1

    @property
    def node_degrees(self):
        """
        Returns the node degrees of all nodes.

        Returns
        -------
        numpy.ndarray [N]
            The node degrees for all nodes (as int64).
        """
        return self.adjacency.sum(axis=1).A1.astype(np.int64)

    @property
    def edges(self):
        """
        Returns the edges of the graph as tuples of node indices.

        Returns
        -------
        numpy.ndarray [2, M]
            The edges as tuples of size 2.
        """
        return np.array(self.adjacency.nonzero())

    def largest_connected_component(self):
        """
        Returns a new graph containing only the nodes from the largest connected component. In case
        the graph has a label, the label is removed.

        Returns
        -------
        pymaesh.Graph
            A new, potentially smaller, graph.
        """
        _, components = gs.connected_components(self.adjacency)
        largest_component = np.bincount(components).argmax()  # pylint: disable=no-member
        mask = components == largest_component
        return Graph(
            self.adjacency[mask][:, mask],
            self.features[mask] if self.features is not None else None,
            self.labels[mask] if self.labels is not None else None
        )

    def save(self, file):
        """
        Saves the graph to the specified file. The file should have no extension such that it can
        be saved with the .npz extension.

        Parameters
        ----------
        file: str
            The file to save the graph to.
        """
        data = {
            'adj_data': self.adjacency.data,
            'adj_indices': self.adjacency.indices,
            'adj_indptr': self.adjacency.indptr,
            'adj_shape': self.adjacency.shape
        }
        if self.features is not None:
            if isinstance(self.features, np.ndarray):
                data['attr'] = self.features
            else:
                data['attr_data'] = self.features.data
                data['attr_indices'] = self.features.indices
                data['attr_indptr'] = self.features.indptr
                data['attr_shape'] = self.features.shape
        if self.labels is not None:
            data['labels'] = self.labels
        if self.graph_label is not None:
            data['graph_label'] = self.graph_label
        np.savez(file, data)

    def __getattr__(self, name):
        # Check if there's a stats function or a func function
        if hasattr(stats, name):
            f = getattr(stats, name)
        elif hasattr(func, name):
            f = getattr(func, name)
        else:
            classname = self.__class__.__name__
            raise AttributeError(
                f'Cannot get attribute {name} of {classname}.'
            )
        return functools.partial(f, self)

    def __eq__(self, other):
        if not isinstance(other, Graph):
            return False
        if other.num_nodes != self.num_nodes:
            return False
        return not (self.adjacency != other.adjacency).todense().any()

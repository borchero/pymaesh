from abc import ABC, abstractmethod

class BaseGraph(ABC):
    """
    Abstract base class for graph implementations using matrices.
    """

    def __init__(self, adjacency, features=None, labels=None, graph_label=None):
        """
        Initializes a new graph. The types of the passed inputs depend on the
        particular implementation of the subclass.

        Parameters
        ----------
        adjacency: matrix [N, N]
            The graph's symmetric adjacency matrix. Unless the graph is small or
            dense, it should be sparse.
        features: matrix [N, D]
            The graph's feature matrix. Should only be sparse if the features
            are sparse. The parameter may be None if the graph does not have
            node features.
        labels: vector [N]
            The labels for all nodes. May be None if the graph does not have
            node labels.
        graph_label: object
            A label to assign to the graph.
        """
        assert adjacency.shape[0] == adjacency.shape[1], \
            "Adjacency matrix is not square."
        assert features is None or adjacency.shape[0] == features.shape[0], \
            "Dimensions of adjacency matrix and feature matrix are invalid."
        assert labels is None or adjacency.shape[0] == labels.shape[0], \
            "Dimension of label matrix is invalid."

        self.adjacency = adjacency
        self.features = features
        self.labels = labels
        self.graph_label = graph_label

    @property
    def num_nodes(self):
        """
        Computes the number of nodes in the graph.

        Returns
        -------
        int
            The number of nodes.
        """
        return self.adjacency.shape[0]

    @property
    @abstractmethod
    def num_edges(self):
        """
        Computes the number of edges in the graph.

        Returns
        -------
        int
            The number of edges.
        """

    @property
    @abstractmethod
    def num_classes(self):
        """
        Computes the number of classes assuming the node labels are present.

        Returns
        -------
        int
            The number of classes
        """

    def __repr__(self):
        name = self.__class__.__name__
        result = f'{name}(<nodes: {self.num_nodes}, edges: {self.num_edges}'
        if self.features is not None:
            result += f', features: {self.features.shape[1]}'
        if self.labels is not None:
            result += f', classes: {self.num_classes}'
        if self.graph_label is not None:
            result += f', label: {self.graph_label}'
        return result + '>)'

    def __len__(self):
        return self.adjacency.shape[0]

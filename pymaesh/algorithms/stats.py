import numpy as np
import numpy.linalg as la
import scipy.sparse.csgraph as gs
import scipy.stats as st
import scipy.sparse as sp
from ..utils.numpy import intersection_mask_pairs

def edge_overlap(graph_lhs, graph_rhs):
    """
    Computes the edge overlap between two graphs.

    Parameters
    ----------
    graph_lhs: pymaesh.Graph
        The first graph.
    graph_rhs: pymaesh.Graph
        The second graph.

    Returns
    -------
    int
        The number of overlapping edges.
    """

    A = graph_lhs.adjacency
    B = graph_rhs.adjacency

    A_nnz = np.array(sp.triu(A).nonzero()).T
    B_nnz = np.array(sp.triu(B).nonzero()).T

    return int(intersection_mask_pairs(A_nnz, B_nnz).sum())


def wedge_count(graph):
    """
    Computes the number of wedges (2-stars) in the graph.

    Note
    ----
    The number of wedges s is given by the following formula:

    .. math::

        s = \\sum_{u \\in V}{\\binom{d(u)}{2}} =
            \\sum_{u \\in V}{1/2 d(u) \\cdot (d(u) - 1)}

    Parameters
    ----------
    graph: pymaesh.Graph
        The graph to compute the wedge count for.

    Returns
    -------
    int
        The number of wedges in the graph.
    """
    degs = graph.node_degrees
    return np.sum(degs * (degs - 1) / 2)


def claw_count(graph):
    """
    Computes the number of claws (3-stars) in the graph.

    Note
    ----
    The number of claws z is given by the following formula:

    .. math::

        z = \\sum_{u \\in V}{\\binom{d(u)}{3}} =
            \\sum_{u \\in V}{1/6 d(u) \\cdot (d(u) - 1) \\cdot (d(u) - 2)}

    Parameters
    ----------
    graph: pymaesh.Graph
        The graph to compute the wedge count for.

    Returns
    -------
    int
        The number of wedges in the graph.
    """
    degs = graph.node_degrees
    return np.sum(degs * (degs - 1) * (degs - 2) / 6)


def triangle_count(graph):
    """
    Computes the number of triangles in the graph.

    Parameters
    ----------
    graph: pymaesh.Graph
        The graph to compute the triangle count for.

    Returns
    -------
    int
        The number of triangles in the graph.
    """
    A3 = la.matrix_power(graph.adjacency, 3)
    return np.trace(A3) / 6


def square_count(graph):
    """
    Computes the number of squares in the graph.

    Note
    ----
    Formula is taken from https://bit.ly/2YwBWr2.

    Parameters
    ----------
    graph: pymaesh.Graph
        The graph to compute the square count for.

    Returns
    -------
    int
        The square count.
    """
    A4 = la.matrix_power(graph.adjacency, 4)
    denom = np.sum(graph.node_degrees ** 2)
    return np.trace(A4) / denom


def edge_distribution_entropy(graph):
    """
    Computes the edge distribution entropy of the graph.

    Parameters
    ----------
    graph: pymaesh.Graph
        The graph to compute the edge distribution entropy for.

    Returns
    -------
    float
        The edge distribution entropy in [0, 1].
    """
    degs = graph.node_degrees / (2 * graph.num_edges)
    return -1 / np.log(graph.num_nodes) * np.sum(degs * np.log(degs))


def gini_coefficient(graph):
    """
    Computes the Gini coefficient of the graph.

    Parameters
    ----------
    graph: pymaesh.Graph
        The graph to compute the Gini coefficient for.

    Returns
    -------
    float
        The Gini coefficient in [0, 1].
    """
    degs = graph.node_degrees
    np.sort(degs)
    num = 2 * np.sum(np.arange(1, graph.num_nodes + 1) * degs)
    denom = graph.num_nodes * np.sum(degs)
    return num / denom - (graph.num_nodes + 1) / graph.num_nodes


def average_diameter(graph):
    """
    Computes the average diameter of the graph.

    Parameters
    ----------
    graph: pymaesh.Graph
        The graph to compute the average diameter for.

    Returns
    -------
    float
        The average diameter.
    """
    shortest_paths = gs.shortest_path(graph.adjacency)
    mask = (~np.isinf(shortest_paths)) * (1 - np.eye(graph.num_nodes))
    return np.mean(shortest_paths[mask])


def power_law_exponent(graph):
    """
    Estimates the power law exponent of the graph.

    Note
    ----
    The power law exponent gamma can be estimated as follows:

    .. math::

        \\gamma = 1 + \\frac{n}{\\sum_{u \\in V}{\\ln{\\frac{d(u)}{d_{\\min}}}}}

    Parameters
    ----------
    graph: pymaesh.Graph
        The graph to estimate the power law exponent for.

    Returns
    -------
    float
        The power law exponent.
    """
    degs = graph.node_degrees
    min_deg = max(np.min(degs), 1)
    denom = np.sum(np.log(degs / min_deg))
    return 1 + graph.num_nodes / denom


def clustering_coefficient(graph):
    """
    Computes the clustering coefficient of the graph.

    Parameters
    ----------
    graph: pymaesh.Graph
        The graph to compute the clustering coefficient for.

    Returns
    --------
    float
        The clustering coefficient.
    """
    return 3 * triangle_count(graph) / claw_count(graph)


def degree_assortativity(graph):
    """
    Computes the degree assortativity of the graph.

    Parameters
    ----------
    - graph: pymaesh.Graph
        The graph to compute the clustering coefficient for.

    Returns
    -------
    float
        The degree assortativity in [-1, 1].
    """
    edges = graph.edges
    degs = graph.node_degrees
    return st.pearsonr(degs[edges[0]], degs[edges[1]])[0]


def adjacency_eigenvectors(graph, count=1, iterations=None):
    """
    Computes eigenvectors corresponding to the largest eigenvalues of the graph's adjacency matrix
    via power iteration.

    Parameters
    ----------
    count: int, default: 1
        The number of eigenvectors to compute.
    iterations: int, default: None
        The number of iterations to use for power iteration.

    Returns
    -------
    np.array [C, N]
        The eigenvectors corresponding to the C largest eigenvalues (count C, number of nodes in
        the graph N).
    """
    _, eigvecs = sp.linalg.eigs(graph.adjacency, k=count, which='LM', maxiter=iterations)
    return eigvecs

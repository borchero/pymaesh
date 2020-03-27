import numpy as np
import numba

def symmetric_edge_list(edges):
    """
    Given a list of edges, this function computes a set which contains the edges "in both
    directions".

    Parameters
    ----------
    edges: numpy.ndarray [N, 2]
        The edges where edges are considered undirected.

    Returns
    -------
    numpy.ndarray [N * 2, 2]
        The edges where edges may be considered directed.
    """
    reverse_edges = np.column_stack([edges[:, 1], edges[:, 0]])
    return np.concatenate([edges, reverse_edges])


@numba.njit
def intersection_mask(x, y):
    """
    Computes the intersection of the given arrays.

    The arrays are expected to be sorted in ascending order and elements must be unique. Otherwise,
    the result is undefined.

    Parameters
    ----------
    x: numpy.ndarray [N]
        Array number one.
    y: numpy.ndarray [M]
        Array number two.

    Returns
    -------
    numpy.ndarray [N]
        A mask for the array passed as first parameter. When applied to the first array, it returns
        the values contained in both arrays.
    """
    mask = np.zeros_like(x)

    N, M = (x.shape[0], y.shape[0])
    xp, yp = (0, 0)

    while xp < N and yp < M:
        if x[xp] == y[yp]:
            mask[xp] = 1
            xp += 1
            yp += 1
        elif x[xp] < y[yp]:
            xp += 1
        else:
            yp += 1

    return mask


@numba.njit
def intersection_mask_pairs(x, y):
    """
    Computes the intersection of the given pair arrays.

    The arrays are expected to be sorted alphabetically (i.e. (1, 0) comes after (0, 9)) in
    ascending order (first in terms of the first dimension, then in terms of the second). Otherwise,
    the result is undefined.

    Parameters
    ----------
    x: numpy.ndarray [N, 2]
        Array of pairs number one.
    y: numpy.ndarray [N, 2]
        Array of pairs number two.

    Returns
    -------
    numpy.ndarray [N]
        A mask for the array passed as first parameter. When applied to the first array, it returns
        the pairs contained in both arrays.
    """
    N, M = (x.shape[0], y.shape[0])
    mask = np.zeros(N)

    xp, yp = (0, 0)

    while xp < N and yp < M:
        if x[xp][0] == y[yp][0] and x[xp][1] == y[yp][1]:
            mask[xp] = 1
            xp += 1
            yp += 1
        elif x[xp][0] < y[yp][0] or \
                (x[xp][0] == y[yp][0] and x[xp][1] < y[yp][1]):
            xp += 1
        else:
            yp += 1

    return mask


@numba.njit
def cantor_pairing(x, y):
    """
    Computes the cantor pairing function for two integer, mapping two integers bijectively to a
    single one.

    The function is given as follows:

    .. math::

        p(x, y) = \\frac{((x + y) \\cdot (x + y + 1))}{2} + y

    Parameters
    ----------
    x: int
        The first integer.
    y: int
        The second integer.

    Returns
    -------
    int
        The result of the pairing function.
    """
    return ((x + y) * (x + y + 1)) // 2 + y

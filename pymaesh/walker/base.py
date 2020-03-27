from abc import ABC, abstractmethod
import numpy as np

class RandomWalker(ABC):
    """
    Base random walker class for different random walkers on graphs.
    """

    def __init__(self, graph):
        """
        Initializes a new random walker on the specified graph.

        Parameters
        ----------
        graph: pymaesh.Graph
            The graph to generate random walks for.
        """
        self.graph = graph

    def sample_walks(self, num_walks, steps, **kwargs):
        """
        Function to sample a batch of walks.

        Parameters
        ----------
        num_walks: int
            The number of walks to generate.
        steps: int
            The number of steps for the random walker. Must be positive.
        kwargs: keyword arguments
            Arguments dependent on the subclass.

        Returns
        -------
        numpy.ndarray [num_walks, steps]
            The indices of the nodes sampled in the random walks.
        """
        assert steps > 0, \
            "Random walker must at least visit one node."
        assert num_walks > 0, \
            "Random walker must at least make one walk."

        result = np.zeros((num_walks, steps), dtype=np.int64)
        self._sample_walks(result, **kwargs)
        return result

    def walk_generator(self, steps, **kwargs):
        """
        Generator function which yields random walks infinitely often.

        Parameters
        ----------
        steps: int
            The number of steps for the random walker. Must be positive.
        kwargs: keyword arguments
            Arguments dependent on the subclass.

        Returns
        -------
        numpy.ndarray [steps]
            The indices of the nodes sampled in the random walk.
        """
        assert steps > 0, \
            "Random walker must at least visit one node."

        while True:
            out = np.zeros(steps, dtype=np.int64)
            yield self._generate_random_walk(out, **kwargs)

    @abstractmethod
    def _generate_random_walk(self, out, **kwargs):
        pass

    def _sample_walks(self, out, **kwargs):
        for i in range(out.shape[0]):
            self._generate_random_walk(out[i], **kwargs)

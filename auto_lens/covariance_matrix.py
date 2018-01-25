import sys

# TODO: fix environment so we don't need this! *vomits*

if sys.version[0] == '2':
    # noinspection PyPep8Naming
    import Queue as queue
else:
    # noinspection PyUnresolvedReferences
    import queue as queue


class CovarianceMatrixGenerator(object):
    """Class for efficient calculation of big F from little f"""

    def __init__(self, pixel_maps, noise_vector):
        """
        Parameters
        ----------
        pixel_maps: [{int: float}]
            List of dictionaries. Each dictionary describes the contribution that a source pixel makes to image pixels.
        noise_vector: [float]
            A list of noise values of length image pixels
        """
        self.pixel_maps = pixel_maps
        self.noise_vector = noise_vector

    def calculate_covariance(self, source_index_a, source_index_b):
        """
        Calculates Fab

        Parameters
        ----------
        source_index_a: int
            The source pixel index a
        source_index_b: int
            The source pixel index b

        Returns
        -------
            covariance: Float
                The covariance between a and b
        """
        mapping_dict_1 = self.pixel_maps[source_index_a]
        mapping_dict_2 = self.pixel_maps[source_index_b]
        return sum([mapping_dict_1[i] * mapping_dict_2[i] / self.noise_vector[i] for i in mapping_dict_1.keys() if
                    i in mapping_dict_2])


class BreadthFirstSearch(object):
    """Performs a breadth first graph search on the condition that neighbours are added"""

    def __init__(self, graph):
        """
        Parameters
        ----------
        graph: [[int]]
            A list of lists description of a graph such as neighbours in the source plane.
        """
        self.graph = graph
        self.queue = queue.Queue()
        self.visited = set()

    def neighbours(self):
        """
        Returns
        -------
            A generator that yields indices of previously unvisited neighbours
        """
        while not self.queue.empty():
            yield self.queue.get()

    def add_neighbours_of(self, index):
        """
        Enqueue neighbours of a particular node that have not been visited. Note that once a node has been added in this
        way it will never be yielded as a neighbour.

        Parameters
        ----------
        index: int
            The index of the node

        """
        self.visited.add(index)
        for neighbour in self.graph[index]:
            if neighbour not in self.visited:
                self.visited.add(neighbour)
                self.queue.put(neighbour)


class TestBreadthFirstSearch(object):
    def test_simple_search(self):
        """Does the search yield neighbours?"""
        graph = [[1, 2]]

        bfs = BreadthFirstSearch(graph)

        bfs.add_neighbours_of(0)

        assert 2 == len(list(bfs.neighbours()))

    def test_neighbours_in_loop(self):
        """Does the search recursively yield neighbours?"""
        graph = [[1, 2], [3], [], []]

        bfs = BreadthFirstSearch(graph)

        bfs.add_neighbours_of(0)

        count = 0

        for neighbour in bfs.neighbours():
            bfs.add_neighbours_of(neighbour)
            count += 1

        assert count == 3

    def test_ignore_visited(self):
        """Does the search ignore previously visited nodes?"""
        graph = [[1, 2], [3, 0], [1], []]

        bfs = BreadthFirstSearch(graph)

        bfs.add_neighbours_of(0)

        count = 0

        for neighbour in bfs.neighbours():
            bfs.add_neighbours_of(neighbour)
            count += 1

        assert count == 3


class TestCalculateCovariance(object):
    def test_calculate_covariance(self):
        """Is covariance correct in a simple case?"""
        generator = CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1}], [1, 1])

        assert generator.calculate_covariance(0, 1) == 2

    def test_no_covariance(self):
        """Is covariance zero when two source pixels share no image pixels?"""
        generator = CovarianceMatrixGenerator([{1: 3}, {0: 1}], [1, 1])

        assert generator.calculate_covariance(0, 1) == 0

    def test_variable_noise(self):
        """Is the result correct when noise is taken into account?"""
        generator = CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1, 1: 1}], [2, 3])

        assert generator.calculate_covariance(0, 1) == 2

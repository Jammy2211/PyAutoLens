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

    def __init__(self, pixel_maps, noise_vector, graph):
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
        self.graph = graph
        # dictionary mapping coordinate tuples to values {(a, b): covariance}
        self.calculated_covariances = {}

    def find_contiguous_covariances(self, source_index):
        self.add_covariance_for_indices(source_index, source_index)
        bfs = BreadthFirstSearch(self.graph)
        bfs.add_neighbours_of(source_index)

        for index in bfs.neighbours():
            if self.add_covariance_for_indices(source_index, index) > 0:
                bfs.add_neighbours_of(index)

    def add_covariance_for_indices(self, source_index_a, source_index_b):
        """
        Checks if a covariance value has been found for a pair of indices.

        If a value has been found, returns that value.

        If a value can be determined by symmetry, the value is added for this index pair and returned.

        Otherwise, the value is calculated, added and returned.

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
        tup = (source_index_a, source_index_b)
        if tup in self.calculated_covariances:
            return self.calculated_covariances[tup]
        if (source_index_b, source_index_a) in self.calculated_covariances:
            value = self.calculated_covariances[(source_index_b, source_index_a)]
        else:
            value = self.calculate_covariance(source_index_a, source_index_b)
        self.calculated_covariances[tup] = value
        return value

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


class TestContiguousCovariances(object):
    def test_simple_example(self):
        graph = [[1], [2], [3], [4], []]
        generator = CovarianceMatrixGenerator([{0: 1}, {0: 1}, {0: 1}, {1: 1}, {0: 1}], [1, 1, 1, 1, 1], graph)

        generator.find_contiguous_covariances(0)

        assert {(0, 0): 1, (0, 1): 1, (0, 2): 1, (0, 3): 0} == generator.calculated_covariances


class TestReflexiveCovariances(object):
    def test_reflexive_calculation(self):
        """Does Fab == Fba?"""
        generator = CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1}], [1, 1], None)

        assert generator.calculate_covariance(0, 1) == generator.calculate_covariance(1, 0)

    def test_reflexive_recall(self):
        """Does Fab == Fba because of recall from memory?"""
        generator = CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1}], [1, 1], None)
        generator.calculated_covariances[(0, 1)] = 7

        assert generator.add_covariance_for_indices(0, 1) == 7
        assert generator.add_covariance_for_indices(1, 0) == 7


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
        generator = CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1}], [1, 1], None)

        assert generator.calculate_covariance(0, 1) == 2

    def test_no_covariance(self):
        """Is covariance zero when two source pixels share no image pixels?"""
        generator = CovarianceMatrixGenerator([{1: 3}, {0: 1}], [1, 1], None)

        assert generator.calculate_covariance(0, 1) == 0

    def test_variable_noise(self):
        """Is the result correct when noise is taken into account?"""
        generator = CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1, 1: 1}], [2, 3], None)

        assert generator.calculate_covariance(0, 1) == 2

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

    def find_all_contiguous_covariances(self):
        """
        Finds the local contiguous patch in the source plane of non-zero covariances for each source pixel.
        """

        # noinspection PyTypeChecker
        for index in len(self.pixel_maps):
            self.find_contiguous_covariances(index)

    def find_contiguous_covariances(self, source_index):
        """
        Performs a breadth first search starting at the source pixel and calculating covariance with the source pixel
        and each found pixels until no further neighbours that have non-zero covariance with the source pixel are found.

        Parameters
        ----------
        source_index: int
            The index of the pixel for which covariances should be found

        """
        self.add_covariance_for_indices(source_index, source_index)
        bfs = BreadthFirstSearch(self.graph)
        bfs.add_neighbours_of(source_index)

        for index in bfs.neighbours():
            if self.add_covariance_for_indices(source_index, index) > 0:  # TODO: this limit could be some low value
                # TODO: eliminating the need to search all the way to zero covariance pixels
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
        self.calculated_covariances[tup] = value  # Warning: this is a side effect.
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

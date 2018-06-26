import queue as queue

import numpy as np

"""
Find an F matrix from an f matrix efficiently.

It is assumed that the f matrix is sparse_grid. It is also assumed that covariance of source data_to_pixel will 
principally be found between a pixel and data_to_pixel in a contiguous patch about that pixel in the source plane, 
except for the case that overlap occurs between convolution kernels. See https://github.com/Jammy2211/AutoLens/issues/6 
for a thorough discussion.

Given a convolved list of pixel maps pixel_maps of type [{int: float}], a signal_to_noise_ratio vector noise_vector of 
type [float] and a graph describing neighbours on the source plane [[int]] this module will return a covariance matrix 
of type {(int, int): float} where the tuple is a pair of indices of the matrix and only non-zero entries are included.

matrix = covariance_matrix.create_covariance_matrix(pixel_maps, noise_vector, graph)

The optional key word argument neighbour_search_limit can be specified to change the sensitivity of the model_mapper to 
smaller covariance values. The initial step of the process is to find a contiguous neighbour in the source plane of 
non-zero covariance with some pixel. If a zero covariance is found for some pixel then the search does not include 
further neighbours of that pixel. Setting neighbour_search_limit causes the adding of neighbours to stop at the set 
covariance limit.

"""


def compute_covariance_matrix_exact(blurred_mapping_matrix, noise_vector):
    """ Compute the covariance matrix directly - used to integration test that our covariance matrix generator approach
    truly works."""
    covariance_matrix = np.zeros((blurred_mapping_matrix.shape[1], blurred_mapping_matrix.shape[1]))

    for i in range(blurred_mapping_matrix.shape[0]):
        for jx in range(blurred_mapping_matrix.shape[1]):
            for jy in range(blurred_mapping_matrix.shape[1]):
                covariance_matrix[jx, jy] += blurred_mapping_matrix[i, jx] * blurred_mapping_matrix[i, jy] \
                                             / (noise_vector[i] ** 2.0)

    return covariance_matrix


def compute_d_vector_exact(blurred_mapping_matrix, image_vector, noise_vector):
    """ Compute the covariance matrix directly - used to integration test that our covariance matrix generator approach
    truly works."""
    d_matrix = np.zeros((blurred_mapping_matrix.shape[1],))

    for i in range(blurred_mapping_matrix.shape[0]):
        for j in range(blurred_mapping_matrix.shape[1]):
            d_matrix[j] += image_vector[i] * blurred_mapping_matrix[i, j] / (noise_vector[i] ** 2.0)

    return d_matrix


def create_d_matrix(pixel_maps, noise_vector, image_vector):
    """
    Creates a D column matrix

    Parameters
    ----------
    pixel_maps: [{int: float}]
        List of dictionaries. Each dictionary describes the contribution that a source pixel makes to image_grid
        data_to_pixel.
    noise_vector: [float]
        A list of signal_to_noise_ratio values of length image_grid data_to_pixel
    image_vector: [float]
        A vector describing the image_grid

    Returns
    -------
    d_matrix: [float]
        A column matrix
    """

    def value_for_pixel_map(pixel_map):
        value = 0
        for index in pixel_map.keys():
            value += image_vector[index] * pixel_map[index] // noise_vector[index]
        return value

    return list(map(value_for_pixel_map, pixel_maps))


def create_covariance_matrix(pixel_maps, noise_vector, graph, neighbour_search_limit=0.):
    """
    Creates the F matrix from an f matrix

    Parameters
    ----------
    pixel_maps: [{int: float}]
        List of dictionaries. Each dictionary describes the contribution that a source pixel makes to image_grid
        data_to_pixel.
    noise_vector: [float]
        A list of signal_to_noise_ratio values of length image_grid data_to_pixel
    graph: [[int]]
        A graph representing source pixel neighbouring
    neighbour_search_limit: float
        The limit of covariance below which neighbours of a pixel will not be added to the neighbour search queue

    Returns
    -------
    covariance_matrix: {(int, int): float}
        A dictionary mapping indices of non-zero F matrix elements to their values.
    """
    generator = CovarianceMatrixGenerator(pixel_maps, noise_vector, graph, neighbour_search_limit)

    generator.find_all_contiguous_covariances()
    generator.find_all_non_contiguous_covariances()

    return generator.non_zero_covariances


class CovarianceMatrixGenerator(object):
    """Class for efficient calculation of big F from little f"""

    def __init__(self, pixel_maps, noise_vector, graph, neighbour_search_limit=0.):
        """
        Parameters
        ----------
        pixel_maps: [{int: float}]
            List of dictionaries. Each dictionary describes the contribution that a source pixel makes to image_grid
            data_to_pixel.
        noise_vector: [float]
            A list of signal_to_noise_ratio values of length image_grid data_to_pixel
        graph: [[int]]
            A graph representing source pixel neighbouring
        neighbour_search_limit: float
            The limit of covariance below which neighbours of a pixel will not be added to the neighbour search queue
        """
        self.pixel_maps = pixel_maps
        self.noise_vector = noise_vector
        self.graph = graph
        # dictionary mapping coordinate tuples to values {(a, b): covariance}
        self.calculated_covariances = {}

        self.non_zero_covariances = {}

        self.neighbour_search_limit = neighbour_search_limit

        self.no_source_pixels = len(pixel_maps)

        self.neighbour_lists = [[] for _ in range(self.no_source_pixels)]

    def find_all_non_contiguous_covariances(self):
        for l in self.neighbour_lists:
            for a in l:
                for b in l:
                    if a != b:
                        result = self.add_covariance_for_indices(a, b)
                        if result > 0:
                            self.non_zero_covariances[(a, b)] = result

    def find_all_contiguous_covariances(self):
        """
        Finds the local contiguous patch in the source plane of non-zero covariances for each source pixel.
        """

        # noinspection PyTypeChecker
        for index in range(self.no_source_pixels):
            self.find_contiguous_covariances(index)

    def find_contiguous_covariances(self, source_index):
        """
        Performs a breadth first search starting at the source pixel and calculating covariance with the source pixel
        and each found data_to_pixel until no further neighbours that have non-zero covariance with the source pixel
        are found.

        Parameters
        ----------
        source_index: int
            The index of the pixel for which covariances should be found

        """
        self.non_zero_covariances[(source_index, source_index)] = self.add_covariance_for_indices(source_index,
                                                                                                  source_index)
        bfs = BreadthFirstSearch(self.graph)
        bfs.add_neighbours_of(source_index)

        for index in bfs.neighbours():
            result = self.add_covariance_for_indices(source_index, index)
            if result > self.neighbour_search_limit:
                self.neighbour_lists[source_index].append(index)
                self.non_zero_covariances[(source_index, index)] = result
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
        return sum([mapping_dict_1[i] * mapping_dict_2[i] // self.noise_vector[i] for i in mapping_dict_1.keys() if
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

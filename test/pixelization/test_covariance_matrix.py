from src.pixelization import covariance_matrix
import pytest

import numpy as np


@pytest.fixture(name="trivial_pixel_maps")
def make_trivial_pixel_maps():
    return [{0: 1}, {0: 1}, {0: 1}, {1: 1}, {0: 1}]


@pytest.fixture(name="counting_pixel_maps")
def make_counting_pixel_maps():
    return [{0: 1}, {0: 1, 1: 1}, {0: 1, 1: 1, 2: 1}]


class TestDMatrix(object):
    def test_simple_example(self, trivial_pixel_maps):
        noise_vector = [1, 1]
        image_vector = [1, 2]

        assert [1, 1, 1, 2, 1] == covariance_matrix.create_d_matrix(trivial_pixel_maps, noise_vector, image_vector)

    def test_variable_no_pixels_mapped(self, counting_pixel_maps):
        noise_vector = [1, 1, 1]
        image_vector = [1, 1, 1]

        assert [1, 2, 3] == covariance_matrix.create_d_matrix(counting_pixel_maps, noise_vector, image_vector)

    def test_variable_noise(self, counting_pixel_maps):
        noise_vector = [1, 2, 3]
        image_vector = [1, 1, 1]

        assert [1, 1, 1] == covariance_matrix.create_d_matrix(counting_pixel_maps, noise_vector, image_vector)

    def test_variable_image(self):
        noise_vector = [1, 1, 1]
        image_vector = [3, 2, 1]

        pixel_maps = [{0: 1}, {1: 1}, {2: 1}]

        assert [3, 2, 1] == covariance_matrix.create_d_matrix(pixel_maps, noise_vector, image_vector)


@pytest.fixture(name="line_generator")
def make_line_generator(trivial_pixel_maps):
    graph = [[1], [2], [3], [4], []]
    noise_vector = [1 for _ in range(2)]
    return covariance_matrix.CovarianceMatrixGenerator(trivial_pixel_maps, noise_vector, graph)


@pytest.fixture(name="generator")
def make_generator():
    graph = [[1, 2], [0, 3, 4], [0, 4, 5], [1, 4], [1, 2, 3, 6], [2, 6], [2, 4, 5, 7], [6]]
    pixel_maps = []
    for l in [[0, 1, 2, 8], [0, 1, 2, 3], [1, 2, 4], [3, 5], [0, 1, 2, 3, 5, 6], [4, 7], [6, 7], [6, 8]]:
        pixel_maps.append({i: 1 for i in l})
    noise_vector = [1 for _ in range(9)]
    return covariance_matrix.CovarianceMatrixGenerator(pixel_maps, noise_vector, graph)

class TestComputeCovarianceMatrixExactly(object):

    def test__simple_blurred_mapping_matrix__correct_covariance_matrix(self):

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        noise_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        cov = covariance_matrix.compute_covariance_matrix_exact(blurred_mapping_matrix, noise_vector)

        assert (cov == np.array([[2.0, 1.0, 0.0],
                                 [1.0, 3.0, 1.0],
                                 [0.0, 1.0, 1.0]])).all()

    def test__simple_blurred_mapping_matrix__change_noise_values__correct_covariance_matrix(self):

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        noise_vector = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        cov = covariance_matrix.compute_covariance_matrix_exact(blurred_mapping_matrix, noise_vector)

        assert (cov == np.array([[1.25, 0.25, 0.0],
                                 [0.25, 2.25, 1.0],
                                 [0.0, 1.0, 1.0]])).all()

class TestComputeDMatrixExactly(object):

    def test__simple_blurred_mapping_matrix__correct_d_matrix(self):

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        image_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        noise_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        d = covariance_matrix.compute_d_vector_exact(blurred_mapping_matrix, image_vector, noise_vector)

        assert (d == np.array([2.0, 3.0, 1.0])).all()

    def test__simple_blurred_mapping_matrix__change_image_values__correct_d_matrix(self):

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        image_vector = np.array([3.0, 1.0, 1.0, 10.0, 1.0, 1.0])
        noise_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        d = covariance_matrix.compute_d_vector_exact(blurred_mapping_matrix, image_vector, noise_vector)

        assert (d == np.array([4.0, 14.0, 10.0])).all()

    def test__simple_blurred_mapping_matrix__change_noise_values__correct_d_matrix(self):

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        image_vector = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
        noise_vector = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

        d = covariance_matrix.compute_d_vector_exact(blurred_mapping_matrix, image_vector, noise_vector)

        assert (d == np.array([2.0, 3.0, 1.0])).all()


class TestMissingCovariances(object):
    def test_line_neighbour_lists(self, line_generator):
        line_generator.find_contiguous_covariances(0)

        assert line_generator.neighbour_lists == [[1, 2], [], [], [], []]

    def test_group_neighbour_lists(self, generator):
        generator.find_all_contiguous_covariances()

        assert generator.neighbour_lists[0] == [1, 2, 4]
        assert generator.neighbour_lists[7] == [6, 4]

    def test_include_missing_covariances(self, generator):
        generator.find_all_contiguous_covariances()

        non_zero_covariances = generator.non_zero_covariances.copy()

        generator.find_all_non_contiguous_covariances()

        assert len(non_zero_covariances) + 2 == len(generator.non_zero_covariances)

        # TODO: should be iter()
        # noinspection PyCompatibility
        assert {(0, 7), (7, 0)} == {t for t, v in generator.non_zero_covariances.items() if
                                    t not in non_zero_covariances}


class TestContiguousCovariances(object):
    def test_calculated_covariances(self, line_generator):
        line_generator.find_contiguous_covariances(0)

        assert {(0, 0): 1, (0, 1): 1, (0, 2): 1, (0, 3): 0} == line_generator.calculated_covariances

    def test_non_zero_covariances(self, generator):
        generator.find_contiguous_covariances(0)

        assert {(0, 0): 4, (0, 1): 3, (0, 2): 2, (0, 4): 3} == generator.non_zero_covariances


class TestReflexiveCovariances(object):
    def test_reflexive_calculation(self):
        """Does Fab == Fba?"""
        generator = covariance_matrix.CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1}], [1, 1], None)

        assert generator.calculate_covariance(0, 1) == generator.calculate_covariance(1, 0)

    def test_reflexive_recall(self):
        """Does Fab == Fba because of recall from memory?"""
        generator = covariance_matrix.CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1}], [1, 1], None)
        generator.calculated_covariances[(0, 1)] = 7

        assert generator.add_covariance_for_indices(0, 1) == 7
        assert generator.add_covariance_for_indices(1, 0) == 7


class TestBreadthFirstSearch(object):
    def test_simple_search(self):
        """Does the search yield neighbours?"""
        graph = [[1, 2]]

        bfs = covariance_matrix.BreadthFirstSearch(graph)

        bfs.add_neighbours_of(0)

        assert 2 == len(list(bfs.neighbours()))

    def test_neighbours_in_loop(self):
        """Does the search recursively yield neighbours?"""
        graph = [[1, 2], [3], [], []]

        bfs = covariance_matrix.BreadthFirstSearch(graph)

        bfs.add_neighbours_of(0)

        count = 0

        for neighbour in bfs.neighbours():
            bfs.add_neighbours_of(neighbour)
            count += 1

        assert count == 3

    def test_ignore_visited(self):
        """Does the search ignore previously visited nodes?"""
        graph = [[1, 2], [3, 0], [1], []]

        bfs = covariance_matrix.BreadthFirstSearch(graph)

        bfs.add_neighbours_of(0)

        count = 0

        for neighbour in bfs.neighbours():
            bfs.add_neighbours_of(neighbour)
            count += 1

        assert count == 3


class TestCalculateCovariance(object):
    def test_calculate_covariance(self):
        """Is covariance correct in a simple case?"""
        generator = covariance_matrix.CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1}], [1, 1], None)

        assert generator.calculate_covariance(0, 1) == 2

    def test_no_covariance(self):
        """Is covariance zero when two source data_to_image share no image_grid data_to_image?"""
        generator = covariance_matrix.CovarianceMatrixGenerator([{1: 3}, {0: 1}], [1, 1], None)

        assert generator.calculate_covariance(0, 1) == 0

    def test_variable_noise(self):
        """Is the result correct when signal_to_noise_ratio is taken into account?"""
        generator = covariance_matrix.CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1, 1: 1}], [2, 3], None)

        assert generator.calculate_covariance(0, 1) == 2

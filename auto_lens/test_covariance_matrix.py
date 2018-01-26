import covariance_matrix
import pytest


@pytest.fixture(name="line_generator")
def make_line_generator():
    graph = [[1], [2], [3], [4], []]
    pixel_maps = [{0: 1}, {0: 1}, {0: 1}, {1: 1}, {0: 1}]
    noise = [1 for _ in range(2)]
    return covariance_matrix.CovarianceMatrixGenerator(pixel_maps, noise, graph)


@pytest.fixture(name="generator")
def make_generator():
    graph = [[1, 2], [0, 3, 4], [0, 4, 5], [1, 4], [1, 2, 3, 6], [2, 6], [2, 4, 5, 7], [6]]
    pixel_maps = []
    for l in [[0, 1, 2, 8], [0, 1, 2, 3], [1, 2, 4], [3, 5], [0, 1, 2, 3, 5, 6], [4, 7], [6, 7], [6, 8]]:
        pixel_maps.append({i: 1 for i in l})
    noise = [1 for _ in range(9)]
    return covariance_matrix.CovarianceMatrixGenerator(pixel_maps, noise, graph)


class TestMissingCovariances(object):
    def test_line_neighbour_lists(self, line_generator):
        line_generator.find_contiguous_covariances(0)

        assert line_generator.neighbour_lists == [[1, 2], [], [], [], []]

    def test_group_neighbour_lists(self, generator):
        generator.find_all_contiguous_covariances()

        assert generator.neighbour_lists[0] == [1, 2, 4]
        assert generator.neighbour_lists[7] == [6, 4]


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
        """Is covariance zero when two source pixels share no image pixels?"""
        generator = covariance_matrix.CovarianceMatrixGenerator([{1: 3}, {0: 1}], [1, 1], None)

        assert generator.calculate_covariance(0, 1) == 0

    def test_variable_noise(self):
        """Is the result correct when noise is taken into account?"""
        generator = covariance_matrix.CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1, 1: 1}], [2, 3], None)

        assert generator.calculate_covariance(0, 1) == 2

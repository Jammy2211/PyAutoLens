from analysis import covariance_matrix
import pytest


class TestMappingMatrix:

    def test__coordinates_to_source_pixel_index__3x6_sub_grid_size_1(self):
        source_pixel_total = 3
        image_pixel_total = 6
        sub_grid_size = 1

        sub_image_pixel_to_image_pixel_index = [0, 1, 2, 3, 4, 5]  # For no sub grid, image pixels map to sub-pixels.
        sub_image_pixel_to_source_pixel_index = [0, 1, 2, 0, 1, 2]

        mapping_matrix = covariance_matrix.create_mapping_matrix(source_pixel_total, image_pixel_total, sub_grid_size,
                                          sub_image_pixel_to_source_pixel_index,
                                          sub_image_pixel_to_image_pixel_index)

    #    assert (mapping_matrix == np.array([[1, 0, 0, 1, 0, 0],  # Image pixels 0 and 3 map to source pixel 0.
    #                                        [0, 1, 0, 0, 1, 0],  # Image pixels 1 and 4 map to source pixel 1.
    #                                        [0, 0, 1, 0, 0, 1]])).all()  # Image pixels 2 and 5 map to source pixel 2

        assert (mapping_matrix == [{0:1, 3:1}, {1:1, 4:1}, {2:1.0, 5:1}])

    def test__coordinates_to_source_pixel_index__5x11_grid_size_1(self):
        source_pixel_total = 5
        image_pixel_total = 11
        sub_grid_size = 1

        sub_image_pixel_to_image_pixel_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                10]  # For no sub grid, image pixels map to sub-pixels.
        sub_image_pixel_to_source_pixel_index = [0, 1, 2, 0, 1, 2, 4, 3, 2, 4, 3]

        mapping_matrix = covariance_matrix.create_mapping_matrix(source_pixel_total, image_pixel_total, sub_grid_size,
                                          sub_image_pixel_to_source_pixel_index,
                                          sub_image_pixel_to_image_pixel_index)

     #   assert (mapping_matrix == np.array(
     #       [[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Image pixels 0 and 3 map to source pixel 0.
     #        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Image pixels 1 and 4 map to source pixel 1.
     #        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
     #        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
     #        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])).all()  # Image pixels 2 and 5 map to source pixel 2

        assert (mapping_matrix == [{0:1, 3:1}, {1:1, 4:1}, {2:1, 5:1, 8:1},
                                   {7:1, 10:1}, {6:1, 9:1}])

    def test__coordinates_to_source_pixel_index__3x6_grid_size_2_but_fully_overlaps_image_pixels(self):
        source_pixel_total = 3
        image_pixel_total = 6
        sub_grid_size = 2

        # all sub-pixels to pixel / source_pixel mappings below have been set up such that all sub-pixels in an image pixel
        # map to the same source pixel. This means the same mapping matrix as above will be computed with no fractional
        # values in the final matrix.

        sub_image_pixel_to_image_pixel_index = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                                3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]

        sub_image_pixel_to_source_pixel_index = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                                 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

        mapping_matrix = covariance_matrix.create_mapping_matrix(source_pixel_total, image_pixel_total, sub_grid_size,
                                          sub_image_pixel_to_source_pixel_index,
                                          sub_image_pixel_to_image_pixel_index)

    #    assert (mapping_matrix == np.array([[1, 0, 0, 1, 0, 0],  # Image pixels 0 and 3 map to source pixel 0.
    #                                        [0, 1, 0, 0, 1, 0],  # Image pixels 1 and 4 map to source pixel 1.
    #                                        [0, 0, 1, 0, 0, 1]])).all()  # Image pixels 2 and 5 map to source pixel 2

        assert (mapping_matrix == [{0:1, 3:1}, {1:1, 4:1}, {2:1, 5:1}])

    def test__coordinates_to_source_pixel_index__5x11_grid_size_2_but_fully_overlaps_image_pixels(self):
        source_pixel_total = 5
        image_pixel_total = 11
        sub_grid_size = 2

        # all sub-pixels to pixel / source_pixel mappings below have been set up such that all sub-pixels in an image pixel
        # map to the same source pixel. This means the same mapping matrix as above will be computed with no fractional
        # values in the final matrix.

        sub_image_pixel_to_image_pixel_index = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                                                6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10]

        sub_image_pixel_to_source_pixel_index = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                                 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 3, 3, 3, 3]

        mapping_matrix = covariance_matrix.create_mapping_matrix(source_pixel_total, image_pixel_total, sub_grid_size,
                                          sub_image_pixel_to_source_pixel_index,
                                          sub_image_pixel_to_image_pixel_index)

     #   assert (mapping_matrix == np.array(
     #       [[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Image pixels 0 and 3 map to source pixel 0.
     #        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Image pixels 1 and 4 map to source pixel 1.
     #        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
     #        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
     #       [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])).all()  # Image pixels 2 and 5 map to source pixel 2


        assert (mapping_matrix == [{0:1, 3:1}, {1:1, 4:1}, {2:1, 5:1, 8:1},
                                   {7:1, 10:1}, {6:1, 9:1}])

    def test__coordinates_to_source_pixel_index__3x6_grid_size_2_not_fully_overlapping(self):
        source_pixel_total = 3
        image_pixel_total = 6
        sub_grid_size = 2

        # all sub-pixels to pixel / source_pixel mappings below have been set up such that all sub-pixels in an image pixel
        # map to the same source pixel. This means the same mapping matrix as above will be computed with no fractional
        # values in the final matrix.

        sub_image_pixel_to_image_pixel_index = [0, 1, 1, 0, 1, 4, 4, 1, 2, 2, 2, 0,
                                                3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]

        sub_image_pixel_to_source_pixel_index = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                                 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

        mapping_matrix = covariance_matrix.create_mapping_matrix(source_pixel_total, image_pixel_total, sub_grid_size,
                                          sub_image_pixel_to_source_pixel_index,
                                          sub_image_pixel_to_image_pixel_index)

     #   assert (mapping_matrix == np.array([[0.5, 0.5, 0, 1, 0, 0],  # Image pixels 0 and 3 map to source pixel 0.
     #                                       [0, 0.5, 0, 0, 1.5, 0],  # Image pixels 1 and 4 map to source pixel 1.
     #                                       [0.25, 0, 0.75, 0, 0,
     #                                        1]])).all()  # Image pixels 2 and 5 map to source pixel 2


        assert (mapping_matrix == [{0:0.5, 1:0.5, 3:1}, {1:0.5, 4:1.5}, {0:0.25, 2:0.75, 5:1}])

    def test__coordinates_to_source_pixel_index__5x11_grid_size_2_not_fully_overlapping(self):
        source_pixel_total = 5
        image_pixel_total = 11
        sub_grid_size = 2

        # Moving one of every 4 sub-pixels to the right compared to the example above. This should turn each 1 in the
        # mapping matrix to a 0.75, and add a 0.25 to the element to its right

        # Note the last value retains all 4 of it's '10's, so keeps a 1 in the mapping matrix

        sub_image_pixel_to_image_pixel_index = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6,
                                                6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10]

        sub_image_pixel_to_source_pixel_index = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                                 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 3, 3, 3, 3]


        mapping_matrix = covariance_matrix.create_mapping_matrix(source_pixel_total, image_pixel_total, sub_grid_size,
                                          sub_image_pixel_to_source_pixel_index,
                                          sub_image_pixel_to_image_pixel_index)

    #    assert (mapping_matrix == np.array(
    #        [[0.75, 0.25, 0, 0.75, 0.25, 0, 0, 0, 0, 0, 0],  # Image pixels 0 and 3 map to source pixel 0.
    #         [0, 0.75, 0.25, 0, 0.75, 0.25, 0, 0, 0, 0, 0],  # Image pixels 1 and 4 map to source pixel 1.
    #         [0, 0, 0.75, 0.25, 0, 0.75, 0.25, 0, 0.75, 0.25, 0],
    #         [0, 0,    0,    0, 0,    0, 0, 0.75, 0.25, 0, 1],
    #         [0, 0, 0, 0, 0, 0, 0.75, 0.25, 0, 0.75, 0.25]])).all()  # Image pixels 2 and 5 map to source pixel 2

        assert (mapping_matrix == [{0:0.75, 1:0.25, 3:0.75, 4:0.25},
                                   {1:0.75, 2:0.25, 4:0.75, 5:0.25},
                                   {2:0.75, 3:0.25, 5:0.75, 6:0.25, 8:0.75, 9:0.25},
                                   {7:0.75, 8:0.25, 10:1},
                                   {6:0.75, 7:0.25, 9:0.75, 10:0.25}])

    def test__coordinates_to_source_pixel_index__2x3_grid_size_4(self):
        source_pixel_total = 2
        image_pixel_total = 3
        sub_grid_size = 4

        # 4x4 sub pixel, so 16 sub-pixels per pixel, so 48 sub-image pixels,

        # No sub-pixels labelled 0 map to source_pixel 0, so f(0,0) remains 0
        # 15 sub-pixels labelled 1 map to source_pixel_index 0, so add 4 * (1/16) = 0.9375 to f(1,1)
        # 1 sub-pixel labelled 2 map to source_pixel_index 0, so add (1/16) = 0.0625 to f(2,1)
        # 4 sub-pixels labelled 0 map to source_pixel_index 1, so add 4 * (1/16) = 0.25 to f(0,2)
        # 12 sub-pixels labelled 1 map to source_pixel_index 1, so add 12 * (1/16) = 0.75 to f(1,2)
        # 16 sub-pixels labelled 2 map to source_pixel_index 1, so add (16/16) = 1.0 to f(2,2)

        sub_image_pixel_to_image_pixel_index = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                # 50:50 ratio so 1 in each entry of the mapping matrix
                                                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]

        sub_image_pixel_to_source_pixel_index = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        mapping_matrix = covariance_matrix.create_mapping_matrix(source_pixel_total, image_pixel_total, sub_grid_size,
                                          sub_image_pixel_to_source_pixel_index,
                                          sub_image_pixel_to_image_pixel_index)

    #    assert (mapping_matrix == np.array([[0, 0.9375, 0.0625],
    #                                        [0.25, 0.75, 1.0]])).all()

        assert (mapping_matrix == [{1:0.9375, 2: 0.0625}, {0:0.25, 1:0.75, 2:1.0}])



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
        """Is covariance zero when two source pixels share no image pixels?"""
        generator = covariance_matrix.CovarianceMatrixGenerator([{1: 3}, {0: 1}], [1, 1], None)

        assert generator.calculate_covariance(0, 1) == 0

    def test_variable_noise(self):
        """Is the result correct when noise is taken into account?"""
        generator = covariance_matrix.CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1, 1: 1}], [2, 3], None)

        assert generator.calculate_covariance(0, 1) == 2

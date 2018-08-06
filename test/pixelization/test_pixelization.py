from autolens.pixelization import pixelization

import pytest
import numpy as np

from autolens.imaging import mask
from test.mock.mock_mask import MockSubGridCoords, MockGridCollection, MockBorderCollection


def grid_to_pix_pixels_via_nearest_neighbour(grid, pix_centers):
    def compute_squared_separation(coordinate1, coordinate2):
        """Computes the squared separation of two image_grid (no square root for efficiency)"""
        return (coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2

    image_pixels = grid.shape[0]

    image_to_pix = np.zeros((image_pixels,))

    for image_index, image_coordinate in enumerate(grid):
        distances = list(map(lambda centers: compute_squared_separation(image_coordinate, centers),
                             pix_centers))

        image_to_pix[image_index] = (np.argmin(distances))

    return image_to_pix


class MockSparseMask(object):

    def __init__(self, sparse_to_image, image_to_sparse):
        """ The KMeans clustering used to derive an amorphous pixelization uses a set of image-grid grid. For \
        high resolution imaging, the large number of grid makes KMeans clustering (unfeasibly) slow.

        Therefore, for efficiency, we define a 'clustering-grid', which is a sparsely sampled set of image-grid \
        grid used by the KMeans algorithm instead. However, we don't need the actual grid of this \
        clustering grid (as they are already calculated for the image-grid). Instead, we just need a mapper between \
        clustering-data_to_image and image-data_to_image.

        Thus, the *cluster_to_image* attribute maps every pixel on the clustering grid to its closest image pixel \
        (via the image pixel's 1D index). This is used before the KMeans clustering algorithm, to extract the sub-set \
        of grid that the algorithm uses.

        By giving the KMeans algorithm only clustering-grid grid, it will only tell us the mappings between \
        pix-data_to_image and clustering-data_to_image. However, to perform the pix reconstruction, we need to
        know all of the mappings between pix data_to_image and image data_to_image / sub-image data_to_image. This
        would require a (computationally expensive) nearest-neighbor search (over all clustering data_to_image and
        image / sub data_to_image) to calculate. The calculation can be sped-up by using the attribute
        *image_to_cluster*, which maps every image-pixel to its closest pixel on the clustering grid (see
        *pixelization.sub_grid_to_pix_pixels_via_sparse_pairs*).
        """

        self.sparse_to_image = sparse_to_image
        self.image_to_sparse = image_to_sparse


class MockGeometry(object):

    def __init__(self, y_min, y_max, x_min, x_max, y_pixel_scale, x_pixel_scale):
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_pixel_scale = y_pixel_scale
        self.x_pixel_scale = x_pixel_scale


@pytest.fixture(name="five_pixels")
def make_five_pixels():
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])


@pytest.fixture(name="three_pixels")
def make_three_pixels():
    return np.array([[0, 0], [0, 1], [1, 0]])


class TestPixelizationGrid:

    class TestCoordinateGridWithinAnnulus:

        def test__shape_3x3__circle_radius_15__all_9_pixels_in_grid_with_correct_coordinates(self):

            pix_grid = pixelization.PixelizationGrid(shape=(3,3))

            coordinate_grid = pix_grid.coordinate_grid_within_annulus(inner_radius=0.0, outer_radius=1.5)

            assert (coordinate_grid == np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                                  [0.0, -1.0],  [0.0, 0.0],  [0.0, 1.0],
                                                  [1.0, -1.0],  [1.0, 0.0],  [1.0, 1.0]])).all()

        def test__shape_3x3__circle_radius_3__all_9_pixels_in_grid_with_correct_coordinates(self):

            pix_grid = pixelization.PixelizationGrid(shape=(3,3))

            coordinate_grid = pix_grid.coordinate_grid_within_annulus(inner_radius=0.0, outer_radius=3)

            assert (coordinate_grid == np.array([[-2.0, -2.0], [-2.0, 0.0], [-2.0, 2.0],
                                                  [0.0, -2.0],  [0.0, 0.0],  [0.0, 2.0],
                                                  [2.0, -2.0],  [2.0, 0.0],  [2.0, 2.0]])).all()

        def test__shape_3x2__circle_radius_15__all_6_pixels_in_grid_with_correct_coordinates(self):

            pix_grid = pixelization.PixelizationGrid(shape=(3, 2))

            coordinate_grid = pix_grid.coordinate_grid_within_annulus(inner_radius=0.0, outer_radius=1.5)

            assert (coordinate_grid == np.array([[-1.0, -0.75], [-1.0, 0.75],
                                                  [0.0, -0.75],  [0.0, 0.75],
                                                  [1.0, -0.75],  [1.0, 0.75]])).all()

        def test__shape_2x3__circle_radius_15__all_6_pixels_in_grid_with_correct_coordinates(self):

            pix_grid = pixelization.PixelizationGrid(shape=(2, 3))

            coordinate_grid = pix_grid.coordinate_grid_within_annulus(inner_radius=0.0, outer_radius=1.5)

            assert (coordinate_grid == np.array([[-0.75, -1.0], [-0.75, 0.0], [-0.75, 1.0],
                                                  [0.75, -1.0],  [0.75, 0.0], [0.75, 1.0]])).all()



class TestPixelization:


    class TestMappingMatrix:

        def test__3_image_pixels__6_pix_pixels__sub_grid_1x1(self, three_pixels):
            sub_to_pix = np.array([0, 1, 2])
            sub_to_image = np.array([0, 1, 2])

            grids = MockGridCollection(image=three_pixels,
                                       sub=MockSubGridCoords(three_pixels,
                                                                   sub_to_image,
                                                                   sub_grid_size=1))

            pix = pixelization.Pixelization(pixels=6)
            mapping_matrix = pix.mapping_matrix_from_sub_to_pix_jit(sub_to_pix, grids)

            assert (mapping_matrix == np.array([[1, 0, 0, 0, 0, 0],  # Image pixel 0 maps to pix pixel 0.
                                                [0, 1, 0, 0, 0, 0],  # Image pixel 1 maps to pix pixel 1.
                                                [0, 0, 1, 0, 0, 0]])).all()  # Image pixel 2 maps to pix pixel 2

        def test__5_image_pixels__8_pix_pixels__sub_grid_1x1(self, five_pixels):
            sub_to_pix = np.array([0, 1, 2, 7, 6])
            sub_to_image = np.array([0, 1, 2, 3, 4])

            grids = MockGridCollection(
                image=five_pixels,
                sub=MockSubGridCoords(five_pixels, sub_to_image,
                                      sub_grid_size=1))

            pix = pixelization.Pixelization(pixels=8)
            mapping_matrix = pix.mapping_matrix_from_sub_to_pix_jit(sub_to_pix, grids)

            assert (mapping_matrix == np.array(
                [[1, 0, 0, 0, 0, 0, 0, 0],  # Image image_to_pixel 0 and 3 map to pix pixel 0.
                 [0, 1, 0, 0, 0, 0, 0, 0],  # Image image_to_pixel 1 and 4 map to pix pixel 1.
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 1, 0]])).all()  # Image image_to_pixel 2 and 5 map to pix pixel 2

        def test__5_image_pixels__8_pix_pixels__sub_grid_2x2__no_overlapping_pixels(self, five_pixels):
            sub_to_pix = np.array([0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2])
            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

            grids = MockGridCollection(image=five_pixels,
                                       sub=MockSubGridCoords(five_pixels, sub_to_image,
                                                                   sub_grid_size=2))

            pix = pixelization.Pixelization(pixels=8)
            mapping_matrix = pix.mapping_matrix_from_sub_to_pix_jit(sub_to_pix, grids)

            assert (mapping_matrix == np.array(
                [[0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0],
                 [0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0],
                 [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
                 [0.25, 0.25, 0, 0.25, 0, 0, 0, 0.25],
                 [0, 0, 0.25, 0, 0.25, 0, 0.25, 0.25]])).all()

        def test__5_image_pixels__8_pix_pixels__sub_grid_2x2__include_overlapping_pixels(self, five_pixels):
            sub_to_pix = np.array([0, 0, 0, 1, 1, 1, 0, 0, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2])
            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

            grids = MockGridCollection(image=five_pixels,
                                       sub=MockSubGridCoords(five_pixels, sub_to_image,
                                                                   sub_grid_size=2))

            pix = pixelization.Pixelization(pixels=8)
            mapping_matrix = pix.mapping_matrix_from_sub_to_pix_jit(sub_to_pix, grids)

            assert (mapping_matrix == np.array(
                [[0.75, 0.25, 0, 0, 0, 0, 0, 0],
                 [0.5, 0.5, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
                 [0.25, 0.25, 0, 0.25, 0, 0, 0, 0.25],
                 [0, 0, 0.25, 0, 0.25, 0, 0.25, 0.25]])).all()

        def test__3_image_pixels__6_pix_pixels__sub_grid_4x4(self, three_pixels):
            sub_to_pix = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                   0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3])

            sub_to_image = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

            grids = MockGridCollection(image=three_pixels,
                                       sub=MockSubGridCoords(three_pixels,
                                                                   sub_to_image, sub_grid_size=4))

            pix = pixelization.Pixelization(pixels=6)
            mapping_matrix = pix.mapping_matrix_from_sub_to_pix_jit(sub_to_pix, grids)

            assert (mapping_matrix == np.array(
                [[0.75, 0.25, 0, 0, 0, 0],
                 [0, 0, 1.0, 0, 0, 0],
                 [0.1875, 0.1875, 0.1875, 0.1875, 0.125, 0.125]])).all()


class TestRectangularPixelization:

    class TestConstructor:

        def test__number_of_pixels_and_regularization_set_up_correctly(self):
            pix = pixelization.Rectangular(shape=(3, 3), regularization_coefficients=(2.0,))

            assert pix.shape == (3, 3)
            assert pix.pixels == 9
            assert pix.regularization_coefficients == (2.0,)

    class TestSetupGeometry:

        def test__3x3_grid__buffer_is_small__grid_give_min_minus_1_max_1__sets_up_geometry_correctly(self):

            pix = pixelization.Rectangular(shape=(3, 3), regularization_coefficients=(1.0,))

            pix_grid = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                 [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                 [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            geometry = pix.geometry_from_pix_sub_grid(pix_grid, buffer=1e-8)

            assert geometry.x_min == -1.0 - 1e-8
            assert geometry.x_max == 1.0 + 1e-8
            assert geometry.x_pixel_scale == (geometry.x_max - geometry.x_min) / 3
            assert geometry.y_min == -1.0 - 1e-8
            assert geometry.y_max == 1.0 + 1e-8
            assert geometry.y_pixel_scale == (geometry.y_max - geometry.y_min) / 3

        def test__3x3_grid__same_as_above_change_buffer(self):

            pix = pixelization.Rectangular(shape=(3, 3), regularization_coefficients=(1.0,))

            pix_grid = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                 [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                 [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            geometry = pix.geometry_from_pix_sub_grid(pix_grid, buffer=1e-4)

            assert geometry.x_min == -1.0 - 1e-4
            assert geometry.x_max == 1.0 + 1e-4
            assert geometry.x_pixel_scale == (geometry.x_max - geometry.x_min) / 3
            assert geometry.y_pixel_scale == (geometry.y_max - geometry.y_min) / 3
            assert geometry.y_min == -1.0 - 1e-4
            assert geometry.y_max == 1.0 + 1e-4

        def test__5x4_grid__buffer_is_small(self):

            pix = pixelization.Rectangular(shape=(5, 4), regularization_coefficients=(1.0,))

            pix_grid = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                 [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                 [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            geometry = pix.geometry_from_pix_sub_grid(pix_grid, buffer=1e-8)

            assert geometry.x_min == -1.0 - 1e-8
            assert geometry.x_max == 1.0 + 1e-8
            assert geometry.x_pixel_scale == (geometry.x_max - geometry.x_min) / 5
            assert geometry.y_min == -1.0 - 1e-8
            assert geometry.y_max == 1.0 + 1e-8
            assert geometry.y_pixel_scale == (geometry.y_max - geometry.y_min) / 4


        def test__3x3_grid__larger_range_of_grid(self):

            pix = pixelization.Rectangular(shape=(3, 3), regularization_coefficients=(1.0,))

            pix_grid = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

            geometry = pix.geometry_from_pix_sub_grid(pix_grid, buffer=1e-8)

            assert geometry.x_min == 1.0 - 1e-8
            assert geometry.x_max == 7.0 + 1e-8
            assert geometry.x_pixel_scale == pytest.approx((geometry.x_max - geometry.x_min) / 3, 1e-4)
            assert geometry.y_min == 2.0 - 1e-8
            assert geometry.y_max == 8.0 + 1e-8
            assert geometry.y_pixel_scale == pytest.approx((geometry.y_max - geometry.y_min) / 3, 1e-4)

    class TestComputeSourceNeighbors:

        def test__compute_pix_neighbors__3x3_grid(self):
            # |0|1|2|
            # |3|4|5|
            # |6|7|8|

            pix = pixelization.Rectangular(shape=(3, 3), regularization_coefficients=(1.0,))

            pix_neighbors = pix.neighbors_from_pixelization()

            assert pix_neighbors[0] == [1, 3]
            assert pix_neighbors[1] == [0, 2, 4]
            assert pix_neighbors[2] == [1, 5]
            assert pix_neighbors[3] == [0, 4, 6]
            assert pix_neighbors[4] == [1, 3, 5, 7]
            assert pix_neighbors[5] == [2, 4, 8]
            assert pix_neighbors[6] == [3, 7]
            assert pix_neighbors[7] == [4, 6, 8]
            assert pix_neighbors[8] == [5, 7]

        def test__compute_pix_neighbors__3x4_grid(self):
            # |0|1| 2| 3|
            # |4|5| 6| 7|
            # |8|9|10|11|

            pix = pixelization.Rectangular(shape=(3, 4), regularization_coefficients=(1.0,))

            pix_neighbors = pix.neighbors_from_pixelization()

            assert pix_neighbors[0] == [1, 4]
            assert pix_neighbors[1] == [0, 2, 5]
            assert pix_neighbors[2] == [1, 3, 6]
            assert pix_neighbors[3] == [2, 7]
            assert pix_neighbors[4] == [0, 5, 8]
            assert pix_neighbors[5] == [1, 4, 6, 9]
            assert pix_neighbors[6] == [2, 5, 7, 10]
            assert pix_neighbors[7] == [3, 6, 11]
            assert pix_neighbors[8] == [4, 9]
            assert pix_neighbors[9] == [5, 8, 10]
            assert pix_neighbors[10] == [6, 9, 11]
            assert pix_neighbors[11] == [7, 10]

        def test__compute_pix_neighbors__4x3_grid(self):
            # |0| 1| 2|
            # |3| 4| 5|
            # |6| 7| 8|
            # |9|10|11|

            pix = pixelization.Rectangular(shape=(4, 3), regularization_coefficients=(1.0,))

            pix_neighbors = pix.neighbors_from_pixelization()

            assert pix_neighbors[0] == [1, 3]
            assert pix_neighbors[1] == [0, 2, 4]
            assert pix_neighbors[2] == [1, 5]
            assert pix_neighbors[3] == [0, 4, 6]
            assert pix_neighbors[4] == [1, 3, 5, 7]
            assert pix_neighbors[5] == [2, 4, 8]
            assert pix_neighbors[6] == [3, 7, 9]
            assert pix_neighbors[7] == [4, 6, 8, 10]
            assert pix_neighbors[8] == [5, 7, 11]
            assert pix_neighbors[9] == [6, 10]
            assert pix_neighbors[10] == [7, 9, 11]
            assert pix_neighbors[11] == [8, 10]

        def test__compute_pix_neighbors__4x4_grid(self):
            # |0 | 1| 2| 3|
            # |4 | 5| 6| 7|
            # |8 | 9|10|11|
            # |12|13|14|15|

            pix = pixelization.Rectangular(shape=(4, 4), regularization_coefficients=(1.0,))

            pix_neighbors = pix.neighbors_from_pixelization()

            assert pix_neighbors[0] == [1, 4]
            assert pix_neighbors[1] == [0, 2, 5]
            assert pix_neighbors[2] == [1, 3, 6]
            assert pix_neighbors[3] == [2, 7]
            assert pix_neighbors[4] == [0, 5, 8]
            assert pix_neighbors[5] == [1, 4, 6, 9]
            assert pix_neighbors[6] == [2, 5, 7, 10]
            assert pix_neighbors[7] == [3, 6, 11]
            assert pix_neighbors[8] == [4, 9, 12]
            assert pix_neighbors[9] == [5, 8, 10, 13]
            assert pix_neighbors[10] == [6, 9, 11, 14]
            assert pix_neighbors[11] == [7, 10, 15]
            assert pix_neighbors[12] == [8, 13]
            assert pix_neighbors[13] == [9, 12, 14]
            assert pix_neighbors[14] == [10, 13, 15]
            assert pix_neighbors[15] == [11, 14]

    class TestComputeImageToSource:

        def test__3x3_grid_of_pix_grid__1_coordinate_per_square_pix_pixel__in_centre_of_pixels(self):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.0 to -(1/3)
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 - (1/3) to (1/3)
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2 - (1/3)" to 1.0"

            pix_grid = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                 [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                 [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            pix = pixelization.Rectangular(shape=(3, 3), regularization_coefficients=(1.0,))

            image_to_pix = pix.grid_to_pix_from_grid_jit(pix_grid, pix.geometry_from_pix_sub_grid(pix_grid))

            assert (image_to_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

        def test__3x3_grid_of_pix_grid__1_coordinate_per_square_pix_pixel__near_edges_of_pixels(self):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.0 to -(1/3)
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 - (1/3) to (1/3)
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2 - (1/3)" to 1.0"

            pix_grid = np.array([[-0.34, -0.34], [-0.34, 0.325], [-1.0, 1.0],
                                 [-0.32, -1.0], [-0.32, 0.32], [0.0, 1.0],
                                 [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            pix = pixelization.Rectangular(shape=(3, 3), regularization_coefficients=(1.0,))

            image_to_pix = pix.grid_to_pix_from_grid_jit(pix_grid, pix.geometry_from_pix_sub_grid(pix_grid))

            assert (image_to_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

        def test__3x3_grid_of_pix_grid__add_multiple_grid_to_1_pix_pixel(self):
            #                  _ _ _
            # -1.0 to -(1/3)  |_|_|_|
            # -(1/3) to (1/3) |_|_|_|
            #  (1/3) to 1.0   |_|_|_|

            pix_grid = np.array([[-1.0, -1.0], [0.0, 0.0], [-1.0, 1.0],
                                 [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                 [1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])

            pix = pixelization.Rectangular(shape=(3, 3), regularization_coefficients=(1.0,))

            image_to_pix = pix.grid_to_pix_from_grid_jit(pix_grid, pix.geometry_from_pix_sub_grid(pix_grid))

            assert (image_to_pix == np.array([0, 4, 2, 4, 4, 4, 6, 4, 8])).all()

        def test__4x3_grid_of_pix_grid__1_coordinate_in_each_pixel(self):
            #   _ _ _
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|

            # Boundaries for column pixel 0 -1.0 to -(1/3)
            # Boundaries for column pixel 1 -(1/3) to (1/3)
            # Boundaries for column pixel 2  (1/3) to 1.0

            # Bounadries for row pixel 0 -1.0 to -0.5
            # Bounadries for row pixel 1 -0.5 to 0.0
            # Bounadries for row pixel 2  0.0 to 0.5
            # Bounadries for row pixel 3  0.5 to 1.0

            pix_grid = np.array([[-1.0, -1.0], [-1.0, -0.32], [-1.0, 0.34], [-0.49, -1.0],
                                 [0.01, 0.34], [1.0, 1.0]])

            pix = pixelization.Rectangular(shape=(4, 3), regularization_coefficients=(1.0,))

            image_to_pix = pix.grid_to_pix_from_grid_jit(pix_grid, pix.geometry_from_pix_sub_grid(pix_grid))

            assert (image_to_pix == np.array([0, 1, 2, 3, 8, 11])).all()

        def test__3x4_grid_of_pix_grid__1_coordinate_in_each_pixel(self):
            #   _ _ _ _
            #  |_|_|_|_|
            #  |_|_|_|_|
            #  |_|_|_|_|

            # Boundaries for row pixel 0 -1.0 to -(1/3)
            # Boundaries for row pixel 1 -(1/3) to (1/3)
            # Boundaries for row pixel 2  (1/3) to 1.0

            # Bounadries for column pixel 0 -1.0 to -0.5
            # Bounadries for column pixel 1 -0.5 to 0.0
            # Bounadries for column pixel 2  0.0 to 0.5
            # Bounadries for column pixel 3  0.5 to 1.0

            pix_grid = np.array([[-1.0, -1.0], [-1.0, -0.49], [-1.0, 0.01], [-0.32, 0.01],
                                 [0.34, -0.01], [1.0, 1.0]])

            pix = pixelization.Rectangular(shape=(3, 4), regularization_coefficients=(1.0,))

            image_to_pix = pix.grid_to_pix_from_grid_jit(pix_grid, pix.geometry_from_pix_sub_grid(pix_grid))

            assert (image_to_pix == np.array([0, 1, 2, 6, 9, 11])).all()

        def test__3x3_grid__change_arcsecond_dimensions_size__grid_adapts_accordingly(self):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.5 to -0.5
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 -0.5 to 0.5
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2  0.5 to 1.5

            pix_grid = np.array([[-1.5, -1.5], [-1.0, 0.0], [-1.0, 0.6], [1.4, 0.0], [1.5, 1.5]])

            pix = pixelization.Rectangular(shape=(3, 3), regularization_coefficients=(1.0,))

            image_to_pix = pix.grid_to_pix_from_grid_jit(pix_grid, pix.geometry_from_pix_sub_grid(pix_grid))

            assert (image_to_pix == np.array([0, 1, 2, 7, 8])).all()

        def test__3x3_grid__change_arcsecond_dimensions__not_symmetric(self):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.5 to -0.5
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 -0.5 to 0.5
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2  0.5 to 1.5

            pix_grid = np.array([[-1.0, -1.5], [-1.0, -0.49], [-0.32, -1.5], [-0.32, 0.51], [1.0, 1.5]])

            pix = pixelization.Rectangular(shape=(3, 3), regularization_coefficients=(1.0,))

            image_to_pix = pix.grid_to_pix_from_grid_jit(pix_grid, pix.geometry_from_pix_sub_grid(pix_grid))

            assert (image_to_pix == np.array([0, 1, 3, 5, 8])).all()

        def test__4x3_grid__change_arcsecond_dimensions__not_symmetric(self):
            #   _ _ _
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|

            pix_grid = np.array([[-1.0, -1.5], [-1.0, -0.49], [-0.49, -1.5], [0.6, 0.0], [1.0, 1.5]])

            pix = pixelization.Rectangular(shape=(4, 3), regularization_coefficients=(1.0,))

            image_to_pix = pix.grid_to_pix_from_grid_jit(pix_grid, pix.geometry_from_pix_sub_grid(pix_grid))

            assert (image_to_pix == np.array([0, 1, 3, 10, 11])).all()

        def test__3x4_grid__change_arcsecond_dimensions__not_symmetric(self):
            #   _ _ _ _
            #  |_|_|_|_|
            #  |_|_|_|_|
            #  |_|_|_|_|

            pix_grid = np.array([[-1.0, -1.5], [-1.0, -0.49], [-0.32, -1.5], [0.34, 0.49], [1.0, 1.5]])

            pix = pixelization.Rectangular(shape=(3, 4), regularization_coefficients=(1.0,))

            image_to_pix = pix.grid_to_pix_from_grid_jit(pix_grid, pix.geometry_from_pix_sub_grid(pix_grid))

            assert (image_to_pix == np.array([0, 1, 4, 10, 11])).all()

    class TestComputeInversion:

        def test__5_simple_grid__no_sub_grid__sets_up_correct_reconstructor(self):

            # Source-plane comprises 5 grid, so 5 image pixels traced to the pix-plane.
            pix_grid = np.array([[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]])
            pix_border = mask.ImageGridBorder(arr=np.array([0, 1, 3, 4]))

            pix_sub_grid = np.array([[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]])
            pix_sub_border = mask.SubGridBorder(arr=np.array([0, 1, 3, 4]), sub_grid_size=1)

            sub_to_image = np.array([0, 1, 2, 3, 4])

            grids = MockGridCollection(image=pix_grid,
                                       sub=MockSubGridCoords(pix_sub_grid, sub_to_image,
                                                                   sub_grid_size=1))

            borders = MockBorderCollection(image=pix_border, sub=pix_sub_border)

            # There is no sub-grid, so our sub_grid are just the image grid (note the NumPy weighted_data structure
            # ensures this has no sub-gridding)

            pix = pixelization.RectangularRegConst(shape=(3, 3), regularization_coefficients=(1.0,))

            reconstructor = pix.reconstructor_from_pix_grids(grids, borders)

            assert (reconstructor.mapping == np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])).all()

            assert (reconstructor.regularization == np.array([[2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                             [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                             [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                                                             [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                                                             [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                                                             [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                                                             [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0,
                                                              2.00000001]])).all()

            assert (reconstructor.image_to_pix == np.array([0, 2, 4, 6, 8])).all()
            assert (reconstructor.sub_to_pix == np.array([0, 2, 4, 6, 8])).all()

        def test__15_grid__no_sub_grid__sets_up_correct_reconstructor(self):

            # Source-plane comprises 15 grid, so 15 image pixels traced to the pix-plane.

            pix_grid = np.array([[-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1],
                                 [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                 [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                 [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                 [0.9, 0.9], [1.0, 1.0], [1.1, 1.1]])

            pix_border = mask.ImageGridBorder(arr=np.array([2, 5, 11, 14]))

            # There is no sub-grid, so our sub_grid are just the image grid (note the NumPy weighted_data structure
            # ensures this has no sub-gridding)
            pix_sub_grid = np.array([[-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1],
                                     [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                     [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                     [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                     [0.9, 0.9], [1.0, 1.0], [1.1, 1.1]])

            pix_sub_border = mask.SubGridBorder(arr=np.array([2, 5, 11, 14]))

            sub_to_image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

            grids = MockGridCollection(image=pix_grid,
                                       sub=MockSubGridCoords(pix_sub_grid, sub_to_image,
                                                                   sub_grid_size=1))

            borders = MockBorderCollection(image=pix_border, sub=pix_sub_border)

            pix = pixelization.RectangularRegConst(shape=(3, 3), regularization_coefficients=(1.0,))

            reconstructor = pix.reconstructor_from_pix_grids(grids, borders)

            assert (reconstructor.mapping == np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])).all()

            assert (reconstructor.regularization == np.array([[2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                             [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                             [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                                                             [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                                                             [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                                                             [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                                                             [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0,
                                                              2.00000001]])).all()

            assert (reconstructor.image_to_pix == np.array([0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8])).all()
            assert (reconstructor.sub_to_pix == np.array([0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8])).all()

        def test__5_simple_grid__include_sub_grid__sets_up_correct_reconstructor(self):

            # Source-plane comprises 5 grid, so 5 image pixels traced to the pix-plane.
            pix_grid = np.array([[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]])
            pix_border = mask.ImageGridBorder(arr=np.array([0, 1, 3, 4]))
            # Assume a 2x2 sub-grid, so each of our 5 image-pixels are split into 4.
            # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
            # happen for a real lensing calculation. This is to make a mapping matrix which explicitly tests the 
            # sub-grid.
            pix_sub_grid = np.array([[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.0, 0.0],
                                     [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
                                     [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])

            sub_to_image = np.array([0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 2])
            pix_sub_border = mask.SubGridBorder(arr=np.array([0, 1, 2, 4, 5, 6, 12, 13, 14, 16, 17, 18]))

            grids = MockGridCollection(image=pix_grid,
                                       sub=MockSubGridCoords(pix_sub_grid, sub_to_image,
                                                                   sub_grid_size=2))

            borders = MockBorderCollection(image=pix_border, sub=pix_sub_border)

            pix = pixelization.RectangularRegConst(shape=(3, 3), regularization_coefficients=(1.0,))

            reconstructor = pix.reconstructor_from_pix_grids(grids, borders)

            assert (reconstructor.mapping == np.array([[0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75]])).all()

            assert (reconstructor.regularization == np.array([[2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                             [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                             [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                                                             [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                                                             [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                                                             [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                                                             [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0,
                                                              2.00000001]])).all()

            assert (reconstructor.image_to_pix == np.array([0, 2, 4, 6, 8])).all()
            assert (reconstructor.sub_to_pix == np.array(
                [0, 0, 0, 4, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 4, 8, 8, 8, 4])).all()

        def test__same_as_above_but_grid_requires_border_relocation(self):

            # Source-plane comprises 5 grid, so 5 image pixels traced to the pix-plane.
            pix_grid = np.array([[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]])
            pix_border = mask.ImageGridBorder(arr=np.array([0, 1, 3, 4]))
            # Assume a 2x2 sub-grid, so each of our 5 image-pixels are split into 4.
            # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
            # happen for a real lensing calculation. This is to make a mapping matrix which explicitly tests the
            # sub-grid.
            pix_sub_grid = np.array([[-1.0, -1.0], [-2.0, -2.0], [-2.0, -2.0], [0.0, 0.0],
                                     [-1.0, 1.0], [-2.0, 2.0], [-2.0, 2.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [1.0, -1.0], [2.0, -2.0], [2.0, -2.0], [0.0, 0.0],
                                     [1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]])


            pix_sub_border = mask.SubGridBorder(arr=np.array([0, 4, 12, 16]))

            sub_to_image = np.array([0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 2])

            grids = MockGridCollection(image=pix_grid,
                                       sub=MockSubGridCoords(pix_sub_grid, sub_to_image,
                                                                   sub_grid_size=2))

            borders = MockBorderCollection(image=pix_border, sub=pix_sub_border)

            pix = pixelization.RectangularRegConst(shape=(3, 3), regularization_coefficients=(1.0,))

            reconstructor = pix.reconstructor_from_pix_grids(grids, borders)

            assert (reconstructor.mapping == np.array([[0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75]])).all()

            assert (reconstructor.regularization == np.array([[2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                             [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                             [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                                                             [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                                                             [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                                                             [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                                                             [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0,
                                                              2.00000001]])).all()

            assert (reconstructor.image_to_pix == np.array([0, 2, 4, 6, 8])).all()
            assert (reconstructor.sub_to_pix == np.array(
                [0, 0, 0, 4, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 4, 8, 8, 8, 4])).all()


class TestVoronoiPixelization:

    class TestComputeVoronoi:

        def test__points_in_x_cross_shape__sets_up_diamond_voronoi_vertices(self):
            # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

            points = np.array([[-1.0, 1.0], [1.0, 1.0],
                               [0.0, 0.0],
                               [-1.0, -1.0], [1.0, -1.0]])

            pix = pixelization.Voronoi(pixels=5, regularization_coefficients=1.0)
            voronoi = pix.voronoi_from_cluster_grid(points)

            voronoi.vertices = list(map(lambda x: list(x), voronoi.vertices))

            assert [0, 1.] in voronoi.vertices
            assert [-1., 0.] in voronoi.vertices
            assert [1., 0.] in voronoi.vertices
            assert [0., -1.] in voronoi.vertices

        def test__9_points_in_square___sets_up_square_of_voronoi_vertices(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                               [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
                               [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])

            pix = pixelization.Voronoi(pixels=9, regularization_coefficients=1.0)
            voronoi = pix.voronoi_from_cluster_grid(points)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            voronoi.vertices = list(map(lambda x: list(x), voronoi.vertices))

            assert [0.5, 1.5] in voronoi.vertices
            assert [1.5, 0.5] in voronoi.vertices
            assert [0.5, 0.5] in voronoi.vertices
            assert [1.5, 1.5] in voronoi.vertices

        def test__points_in_x_cross_shape__sets_up_pairs_of_voronoi_cells(self):
            # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

            points = np.array([[-1.0, 1.0], [1.0, 1.0],
                               [0.0, 0.0],
                               [-1.0, -1.0], [1.0, -1.0]])

            pix = pixelization.Voronoi(pixels=5, regularization_coefficients=1.0)
            voronoi = pix.voronoi_from_cluster_grid(points)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            voronoi.ridge_points = list(map(lambda x: list(x), voronoi.ridge_points))

            assert len(voronoi.ridge_points) == 8

            assert [2, 0] in voronoi.ridge_points or [0, 2] in voronoi.ridge_points
            assert [2, 1] in voronoi.ridge_points or [1, 2] in voronoi.ridge_points
            assert [2, 3] in voronoi.ridge_points or [3, 2] in voronoi.ridge_points
            assert [2, 4] in voronoi.ridge_points or [4, 2] in voronoi.ridge_points
            assert [0, 1] in voronoi.ridge_points or [1, 0] in voronoi.ridge_points
            assert [0.3] in voronoi.ridge_points or [3, 0] in voronoi.ridge_points
            assert [3, 4] in voronoi.ridge_points or [4, 3] in voronoi.ridge_points
            assert [4, 1] in voronoi.ridge_points or [1, 4] in voronoi.ridge_points

        def test__9_points_in_square___sets_up_pairs_of_voronoi_cells(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                               [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
                               [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])

            pix = pixelization.Voronoi(pixels=9, regularization_coefficients=1.0)
            voronoi = pix.voronoi_from_cluster_grid(points)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            voronoi.ridge_points = list(map(lambda x: list(x), voronoi.ridge_points))

            assert len(voronoi.ridge_points) == 12

            assert [0, 1] in voronoi.ridge_points or [1, 0] in voronoi.ridge_points
            assert [1, 2] in voronoi.ridge_points or [2, 1] in voronoi.ridge_points
            assert [3, 4] in voronoi.ridge_points or [4, 3] in voronoi.ridge_points
            assert [4, 5] in voronoi.ridge_points or [5, 4] in voronoi.ridge_points
            assert [6, 7] in voronoi.ridge_points or [7, 6] in voronoi.ridge_points
            assert [7, 8] in voronoi.ridge_points or [8, 7] in voronoi.ridge_points

            assert [0, 3] in voronoi.ridge_points or [3, 0] in voronoi.ridge_points
            assert [1, 4] in voronoi.ridge_points or [4, 1] in voronoi.ridge_points
            assert [4, 7] in voronoi.ridge_points or [7, 4] in voronoi.ridge_points
            assert [2, 5] in voronoi.ridge_points or [5, 2] in voronoi.ridge_points
            assert [5, 8] in voronoi.ridge_points or [8, 5] in voronoi.ridge_points
            assert [3, 6] in voronoi.ridge_points or [6, 3] in voronoi.ridge_points

    class TestComputeNeighbors:

        def test__points_in_x_cross_shape__neighbors_of_each_pix_pixel_correct(self):
            # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

            points = np.array([[-1.0, 1.0], [1.0, 1.0],
                               [0.0, 0.0],
                               [-1.0, -1.0], [1.0, -1.0]])

            pix = pixelization.Voronoi(pixels=5, regularization_coefficients=1.0)
            voronoi = pix.voronoi_from_cluster_grid(points)
            neighbors = pix.neighbors_from_pixelization(voronoi.ridge_points)

            assert set(neighbors[0]) == {2, 1, 3}
            assert set(neighbors[1]) == {2, 0, 4}
            assert set(neighbors[2]) == {0, 1, 3, 4}
            assert set(neighbors[3]) == {2, 0, 4}
            assert set(neighbors[4]) == {2, 1, 3}

        def test__9_points_in_square___neighbors_of_each_pix_pixel_correct(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                               [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
                               [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])

            pix = pixelization.Voronoi(pixels=9, regularization_coefficients=1.0)
            voronoi = pix.voronoi_from_cluster_grid(points)
            neighbors = pix.neighbors_from_pixelization(voronoi.ridge_points)

            assert set(neighbors[0]) == {1, 3}
            assert set(neighbors[1]) == {0, 2, 4}
            assert set(neighbors[2]) == {1, 5}
            assert set(neighbors[3]) == {0, 4, 6}
            assert set(neighbors[4]) == {1, 3, 5, 7}
            assert set(neighbors[5]) == {2, 4, 8}
            assert set(neighbors[6]) == {3, 7}
            assert set(neighbors[7]) == {4, 6, 8}
            assert set(neighbors[8]) == {5, 7}

    class TestImageToSourceViaNearestNeighborsForTesting:

        def test__grid_to_pix_pixels_via_nearest_neighbour__case1__correct_pairs(self):
            pix_pixels = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
            image_grid = np.array([[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1]])

            image_to_pix = grid_to_pix_pixels_via_nearest_neighbour(image_grid, pix_pixels)

            assert image_to_pix[0] == 0
            assert image_to_pix[1] == 1
            assert image_to_pix[2] == 2
            assert image_to_pix[3] == 3

        def test__grid_to_pix_pixels_via_nearest_neighbour___case2__correct_pairs(self):
            pix_pixels = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
            image_grid = np.array([[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1],
                                   [0.9, -0.9], [-0.9, -0.9], [-0.9, 0.9], [0.9, 0.9]])

            image_to_pix = grid_to_pix_pixels_via_nearest_neighbour(image_grid, pix_pixels)

            assert image_to_pix[0] == 0
            assert image_to_pix[1] == 1
            assert image_to_pix[2] == 2
            assert image_to_pix[3] == 3
            assert image_to_pix[4] == 3
            assert image_to_pix[5] == 2
            assert image_to_pix[6] == 1
            assert image_to_pix[7] == 0

        def test__grid_to_pix_pixels_via_nearest_neighbour___case3__correct_pairs(self):
            pix_pixels = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 0.0], [2.0, 2.0]])
            image_grid = np.array([[0.1, 0.1], [-0.1, -0.1], [0.49, 0.49],
                                   [0.51, 0.51], [1.01, 1.01], [1.51, 1.51]])

            image_to_pix = grid_to_pix_pixels_via_nearest_neighbour(image_grid, pix_pixels)

            assert image_to_pix[0] == 4
            assert image_to_pix[1] == 4
            assert image_to_pix[2] == 4
            assert image_to_pix[3] == 0
            assert image_to_pix[4] == 0
            assert image_to_pix[5] == 5

    class TestSubToSourceViaNearestNeighborsForTesting:

        def test__grid_to_pix_pixels_via_nearest_neighbour__case1__correct_pairs(self):
            pix_pixels = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
            sub_grid = np.array([[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1]])

            sub_to_pix = grid_to_pix_pixels_via_nearest_neighbour(sub_grid, pix_pixels)

            assert sub_to_pix[0] == 0
            assert sub_to_pix[1] == 1
            assert sub_to_pix[2] == 2
            assert sub_to_pix[3] == 3

        def test__grid_to_pix_pixels_via_nearest_neighbour___case2__correct_pairs(self):
            pix_pixels = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
            sub_grid = np.array([[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1],
                                 [0.9, -0.9], [-0.9, -0.9], [-0.9, 0.9], [0.9, 0.9]])

            sub_to_pix = grid_to_pix_pixels_via_nearest_neighbour(sub_grid, pix_pixels)

            assert sub_to_pix[0] == 0
            assert sub_to_pix[1] == 1
            assert sub_to_pix[2] == 2
            assert sub_to_pix[3] == 3
            assert sub_to_pix[4] == 3
            assert sub_to_pix[5] == 2
            assert sub_to_pix[6] == 1
            assert sub_to_pix[7] == 0

        def test__grid_to_pix_pixels_via_nearest_neighbour___case3__correct_pairs(self):
            pix_pixels = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 0.0], [2.0, 2.0]])
            sub_grid = np.array([[0.1, 0.1], [-0.1, -0.1], [0.49, 0.49],
                                 [0.51, 0.51], [1.01, 1.01], [1.51, 1.51]])

            sub_to_pix = grid_to_pix_pixels_via_nearest_neighbour(sub_grid, pix_pixels)

            assert sub_to_pix[0] == 4
            assert sub_to_pix[1] == 4
            assert sub_to_pix[2] == 4
            assert sub_to_pix[3] == 0
            assert sub_to_pix[4] == 0
            assert sub_to_pix[5] == 5

    class TestImageToSource:

        def test__image_grid_to_pix_pixels_via_cluster_pairs__grid_of_pix_pixels__correct_pairs(self):
            pix_centers = np.array([[-1.0, -1.0], [-0.9, 0.9],
                                    [1.0, -1.1], [1.2, 1.2]])

            pix_grid = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                 [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                 [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            image_to_pix = np.array([0, 1, 1, 0, 1, 1, 2, 2, 3])

            image_to_pix_via_nearest_neighbour = grid_to_pix_pixels_via_nearest_neighbour(pix_grid,
                                                                                          pix_centers)

            image_to_cluster = np.array([0, 0, 1, 0, 0, 1, 2, 2, 3])
            cluster_to_image = np.array([0, 2, 6, 8])
            cluster_to_pix = np.array([0, 1, 2, 3])
            cluster_mask = MockSparseMask(sparse_to_image=cluster_to_image, image_to_sparse=image_to_cluster)

            sub_to_image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            grids = MockGridCollection(image=pix_grid,
                                       sub=MockSubGridCoords(np.array([]), sub_to_image,
                                                                   sub_grid_size=1))

            pix = pixelization.Voronoi(pixels=6, regularization_coefficients=1.0)
            voronoi = pix.voronoi_from_cluster_grid(pix_centers)
            pix_neighbors = pix.neighbors_from_pixelization(voronoi.ridge_points)

            image_to_pix_via_pairs = pix.image_to_pix_from_pixelization(grids, pix_centers, pix_neighbors,
                                                                        cluster_to_pix, cluster_mask)

            assert (image_to_pix_via_pairs == image_to_pix).all()
            assert (image_to_pix_via_pairs == image_to_pix_via_nearest_neighbour).all()

    class TestComputeSubToSource:

        def test__sub_grid_to_pix_pixels_via_cluster_pairs__grid_of_pix_pixels__correct_pairs(self):
            pix_centers = np.array([[0.1, 0.1], [1.1, 0.1], [2.1, 0.1],
                                    [0.1, 1.1], [1.1, 1.1], [2.1, 1.1]])
            pix_sub_grid = np.array([[0.05, 0.15], [0.15, 0.15], [0.05, 0.05], [0.15, 0.05],
                                     [1.05, 0.15], [1.15, 0.15], [1.05, 0.05], [1.15, 0.05],
                                     [2.05, 0.15], [2.15, 0.15], [2.05, 0.05], [2.15, 0.05],
                                     [0.05, 1.15], [0.15, 1.15], [0.05, 1.05], [0.15, 1.05],
                                     [1.05, 1.15], [1.15, 1.15], [1.05, 1.05], [1.15, 1.05],
                                     [2.05, 1.15], [2.15, 1.15], [2.05, 1.05], [2.15, 1.05]])

            sub_to_pix_via_nearest_neighbour = grid_to_pix_pixels_via_nearest_neighbour(pix_sub_grid,
                                                                                        pix_centers)

            image_to_cluster = np.array([0, 0, 1, 0, 0, 1, 2, 2, 3])
            cluster_to_image = np.array([0, 2, 6, 8])
            cluster_to_pix = np.array([0, 1, 2, 3])
            cluster_mask = MockSparseMask(sparse_to_image=cluster_to_image, image_to_sparse=image_to_cluster)

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])
            grids = MockGridCollection(image=np.array([]),
                                       sub=MockSubGridCoords(pix_sub_grid, sub_to_image,
                                                                   sub_grid_size=1))

            pix = pixelization.Voronoi(pixels=6, regularization_coefficients=1.0)
            voronoi = pix.voronoi_from_cluster_grid(pix_centers)
            pix_neighbors = pix.neighbors_from_pixelization(voronoi.ridge_points)

            sub_to_pix_via_pairs = pix.sub_to_pix_from_pixelization(grids, pix_centers, pix_neighbors,
                                                                    cluster_to_pix, cluster_mask)

            assert (sub_to_pix_via_nearest_neighbour == sub_to_pix_via_pairs).all()


class TestClusterRegConst:
    
    class TestComputeInversion:

        def test__5_simple_grid__no_sub_grid__sets_up_correct_reconstructor(self):

            pix_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            pix_border = mask.ImageGridBorder(arr=np.array([0, 1, 3, 4]))

            pix_sub_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            pix_sub_border = mask.SubGridBorder(arr=np.array([0, 1, 3, 4]), sub_grid_size=1)

            sub_to_image = np.array([0, 1, 2, 3, 4])

            cluster_to_image = np.array([0, 1, 2, 3, 4])
            image_to_cluster = np.array([0, 1, 2, 3, 4])
            cluster_mask = MockSparseMask(sparse_to_image=cluster_to_image, image_to_sparse=image_to_cluster)

            grids = MockGridCollection(image=pix_grid,
                                       sub=MockSubGridCoords(pix_sub_grid, sub_to_image, sub_grid_size=1))

            borders = MockBorderCollection(image=pix_border, sub=pix_sub_border)

            pix = pixelization.ClusterRegConst(pixels=5, regularization_coefficients=(1.0,))

            reconstructor = pix.reconstructor_from_pix_grids(grids, borders, cluster_mask)

            assert (reconstructor.mapping == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 1.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 1.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 1.0]])).all()

            assert (reconstructor.regularization == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                             [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                             [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

            assert (reconstructor.image_to_pix == np.array([0, 1, 2, 3, 4])).all()
            assert (reconstructor.sub_to_pix == np.array([0, 1, 2, 3, 4])).all()

        def test__15_grid__no_sub_grid__sets_up_correct_reconstructor(self):

            cluster_to_image = np.array([1, 4, 7, 10, 13])
            image_to_cluster = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
            cluster_mask = MockSparseMask(sparse_to_image=cluster_to_image, image_to_sparse=image_to_cluster)

            pix_grid = np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                 [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                 [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                 [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                 [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1]])

            pix_border = mask.ImageGridBorder(arr=np.array([2, 5, 11, 14]))

            pix_sub_grid = np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                     [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                     [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                     [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                     [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1]])

            pix_sub_border = mask.SubGridBorder(arr=np.array([2, 5, 11, 14]))

            sub_to_image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            grids = MockGridCollection(image=pix_grid,
                                       sub=MockSubGridCoords(pix_sub_grid, sub_to_image,
                                                                   sub_grid_size=1))

            borders = MockBorderCollection(image=pix_border, sub=pix_sub_border)


            pix = pixelization.ClusterRegConst(pixels=5, regularization_coefficients=(1.0,))

            reconstructor = pix.reconstructor_from_pix_grids(grids, borders, cluster_mask)

            assert (reconstructor.mapping == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 1.0, 0.0, 0.0, 0.0],
                                                      [0.0, 1.0, 0.0, 0.0, 0.0],
                                                      [0.0, 1.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 1.0, 0.0],
                                                      [0.0, 0.0, 0.0, 1.0, 0.0],
                                                      [0.0, 0.0, 0.0, 1.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 1.0],
                                                      [0.0, 0.0, 0.0, 0.0, 1.0],
                                                      [0.0, 0.0, 0.0, 0.0, 1.0]])).all()

            assert (reconstructor.regularization == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                             [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                             [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

            assert (reconstructor.image_to_pix == np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])).all()
            assert (reconstructor.sub_to_pix == np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])).all()

        def test__5_simple_grid__include_sub_grid__sets_up_correct_reconstructor(self):

            pix_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            pix_border = mask.ImageGridBorder(arr=np.array([0, 1, 3, 4]))

            cluster_to_image = np.array([0, 1, 2, 3, 4])
            image_to_cluster = np.array([0, 1, 2, 3, 4])
            cluster_mask = MockSparseMask(sparse_to_image=cluster_to_image, image_to_sparse=image_to_cluster)

            pix_sub_grid = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0],
                                     [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
                                     [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.0, 0.0]])

            pix_sub_border = mask.SubGridBorder(arr=np.array([0, 1, 2, 4, 5, 6, 12, 13, 14, 16, 17, 18]))

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

            grids = MockGridCollection(image=pix_grid,
                                       sub=MockSubGridCoords(pix_sub_grid, sub_to_image,
                                                                   sub_grid_size=2))

            borders = MockBorderCollection(image=pix_border, sub=pix_sub_border)

            pix = pixelization.ClusterRegConst(pixels=5, regularization_coefficients=(1.0,))

            reconstructor = pix.reconstructor_from_pix_grids(grids, borders, cluster_mask)

            assert (reconstructor.mapping == np.array([[0.75, 0.0, 0.25, 0.0, 0.0],
                                                      [0.0, 0.75, 0.25, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.25, 0.75, 0.0],
                                                      [0.0, 0.0, 0.25, 0.0, 0.75]])).all()

            assert (reconstructor.regularization == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                             [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                             [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

            assert (reconstructor.image_to_pix == np.array([0, 1, 2, 3, 4])).all()
            assert (reconstructor.sub_to_pix == np.array(
                [0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 2])).all()

        def test__same_as_above_but_grid_requires_border_relocation(self):

            pix_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            pix_border = mask.ImageGridBorder(arr=np.array([0, 1, 3, 4]))

            cluster_to_image = np.array([0, 1, 2, 3, 4])
            image_to_cluster = np.array([0, 1, 2, 3, 4])
            cluster_mask = MockSparseMask(sparse_to_image=cluster_to_image, image_to_sparse=image_to_cluster)

            pix_sub_grid = np.array([[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0],
                                     [-1.0, 1.0], [-2.0, 2.0], [-2.0, 2.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [1.0, -1.0], [2.0, -2.0], [2.0, -2.0], [0.0, 0.0],
                                     [-1.0, -1.0], [-2.0, -2.0], [-2.0, -2.0], [0.0, 0.0]])

            pix_sub_border = mask.SubGridBorder(arr=np.array([0, 4, 12, 16]))

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

            grids = MockGridCollection(image=pix_grid,
                                       sub=MockSubGridCoords(pix_sub_grid, sub_to_image,
                                                                   sub_grid_size=2))

            borders = MockBorderCollection(image=pix_border, sub=pix_sub_border)

            pix = pixelization.ClusterRegConst(pixels=5, regularization_coefficients=(1.0,))

            reconstructor = pix.reconstructor_from_pix_grids(grids, borders, cluster_mask)


            assert (reconstructor.mapping == np.array([[0.75, 0.0, 0.25, 0.0, 0.0],
                                                      [0.0, 0.75, 0.25, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.25, 0.75, 0.0],
                                                      [0.0, 0.0, 0.25, 0.0, 0.75]])).all()

            assert (reconstructor.regularization == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                             [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                             [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

            assert (reconstructor.image_to_pix == np.array([0, 1, 2, 3, 4])).all()
            assert (reconstructor.sub_to_pix == np.array(
                [0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 2])).all()


class TestAmorphousPixelization:

    class TestKMeans:

        def test__simple_points__sets_up_two_clusters(self):
            cluster_grid = np.array([[0.99, 0.99], [1.0, 1.0], [1.01, 1.01],
                                     [1.99, 1.99], [2.0, 2.0], [2.01, 2.01]])

            pix = pixelization.Amorphous(pixels=2)

            pix_centers, pix_to_image = pix.kmeans_cluster(cluster_grid)

            assert [2.0, 2.0] in pix_centers
            assert [1.0, 1.0] in pix_centers

            assert list(pix_to_image).count(0) == 3
            assert list(pix_to_image).count(1) == 3

        def test__simple_points__sets_up_three_clusters(self):
            cluster_grid = np.array([[-0.99, -0.99], [-1.0, -1.0], [-1.01, -1.01],
                                     [0.99, 0.99], [1.0, 1.0], [1.01, 1.01],
                                     [1.99, 1.99], [2.0, 2.0], [2.01, 2.01]])

            pix = pixelization.Amorphous(pixels=3)

            pix_centers, pix_to_image = pix.kmeans_cluster(cluster_grid)

            assert [2.0, 2.0] in pix_centers
            assert [1.0, 1.0] in pix_centers
            assert [-1.0, -1.0] in pix_centers

            assert list(pix_to_image).count(0) == 3
            assert list(pix_to_image).count(1) == 3
            assert list(pix_to_image).count(2) == 3

        def test__simple_points__sets_up_three_clusters_more_points_in_third_cluster(self):
            cluster_grid = np.array([[-0.99, -0.99], [-1.0, -1.0], [-1.01, -1.01],

                                     [0.99, 0.99], [1.0, 1.0], [1.01, 1.01],
                                     [0.99, 0.99], [1.0, 1.0], [1.01, 1.01],

                                     [1.99, 1.99], [2.0, 2.0], [2.01, 2.01],
                                     [1.99, 1.99], [2.0, 2.0], [2.01, 2.01],
                                     [1.99, 1.99], [2.0, 2.0], [2.01, 2.01],
                                     [1.99, 1.99], [2.0, 2.0], [2.01, 2.01]])

            pix = pixelization.Amorphous(pixels=3)

            pix_centers, pix_to_image = pix.kmeans_cluster(cluster_grid)

            pix_centers = list(map(lambda x: pytest.approx(list(x), 1e-3), pix_centers))

            assert [2.0, 2.0] in pix_centers
            assert [1.0, 1.0] in pix_centers
            assert [-1.0, -1.0] in pix_centers

            assert list(pix_to_image).count(0) == 3 or 6 or 12
            assert list(pix_to_image).count(1) == 3 or 6 or 12
            assert list(pix_to_image).count(2) == 3 or 6 or 12

            assert list(pix_to_image).count(0) != list(pix_to_image).count(1) != list(pix_to_image).count(2)

    class TestComputeInversion:

        def test__5_simple_grid__no_sub_grid__sets_up_correct_reconstructor(self):

            pix_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            pix_border = mask.ImageGridBorder(arr=np.array([0, 1, 3, 4]))

            pix_sub_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            pix_sub_border = mask.SubGridBorder(arr=np.array([0, 1, 3, 4]), sub_grid_size=1)

            sub_to_image = np.array([0, 1, 2, 3, 4])

            cluster_to_image = np.array([0, 1, 2, 3, 4])
            image_to_cluster = np.array([0, 1, 2, 3, 4])
            cluster_mask = MockSparseMask(sparse_to_image=cluster_to_image, image_to_sparse=image_to_cluster)

            grids = MockGridCollection(image=pix_grid,
                                       sub=MockSubGridCoords(pix_sub_grid, sub_to_image, sub_grid_size=1))

            borders = MockBorderCollection(image=pix_border, sub=pix_sub_border)

            pix = pixelization.AmorphousRegConst(pixels=5, regularization_coefficients=(1.0,))

            reconstructor = pix.reconstructor_from_pix_grids(grids, borders, cluster_mask)

            assert np.sum(reconstructor.mapping) == 5.0
            assert np.sum(reconstructor.mapping[:, 0]) == 1.0
            assert np.sum(reconstructor.mapping[:, 1]) == 1.0
            assert np.sum(reconstructor.mapping[:, 2]) == 1.0
            assert np.sum(reconstructor.mapping[:, 3]) == 1.0
            assert np.sum(reconstructor.mapping[:, 4]) == 1.0
            assert np.sum(reconstructor.mapping[0, :]) == 1.0
            assert np.sum(reconstructor.mapping[1, :]) == 1.0
            assert np.sum(reconstructor.mapping[2, :]) == 1.0
            assert np.sum(reconstructor.mapping[3, :]) == 1.0
            assert np.sum(reconstructor.mapping[4, :]) == 1.0

            assert np.sum(np.diag(reconstructor.regularization)) == 16.00000005
            assert np.sum(reconstructor.regularization) - np.sum(np.diag(reconstructor.regularization)) == -16.0

            assert set(reconstructor.image_to_pix) == set(np.array([0, 1, 2, 3, 4]))
            assert set(reconstructor.sub_to_pix) == set(np.array([0, 1, 2, 3, 4]))

        def test__15_grid__no_sub_grid__sets_up_correct_reconstructor(self):

            cluster_to_image = np.array([1, 4, 7, 10, 13])
            image_to_cluster = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
            cluster_mask = MockSparseMask(sparse_to_image=cluster_to_image, image_to_sparse=image_to_cluster)

            pix_grid = np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                 [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                 [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                 [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                 [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1]])

            pix_border = mask.ImageGridBorder(arr=np.array([2, 5, 11, 14]))

            pix_sub_grid = np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                     [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                     [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                     [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                     [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1]])

            pix_sub_border = mask.SubGridBorder(arr=np.array([2, 5, 11, 14]))

            sub_to_image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            grids = MockGridCollection(image=pix_grid,
                                       sub=MockSubGridCoords(pix_sub_grid, sub_to_image,
                                                                   sub_grid_size=1))

            borders = MockBorderCollection(image=pix_border, sub=pix_sub_border)


            pix = pixelization.AmorphousRegConst(pixels=5, regularization_coefficients=(1.0,))

            reconstructor = pix.reconstructor_from_pix_grids(grids, borders, cluster_mask)

            assert np.sum(reconstructor.mapping) == 15.0

            assert np.sum(reconstructor.mapping[:, 0]) == 3.0
            assert np.sum(reconstructor.mapping[:, 1]) == 3.0
            assert np.sum(reconstructor.mapping[:, 2]) == 3.0
            assert np.sum(reconstructor.mapping[:, 3]) == 3.0
            assert np.sum(reconstructor.mapping[:, 4]) == 3.0

            assert np.sum(reconstructor.mapping[0, :]) == 1.0
            assert np.sum(reconstructor.mapping[1, :]) == 1.0
            assert np.sum(reconstructor.mapping[2, :]) == 1.0
            assert np.sum(reconstructor.mapping[3, :]) == 1.0
            assert np.sum(reconstructor.mapping[4, :]) == 1.0
            assert np.sum(reconstructor.mapping[5, :]) == 1.0
            assert np.sum(reconstructor.mapping[6, :]) == 1.0
            assert np.sum(reconstructor.mapping[7, :]) == 1.0
            assert np.sum(reconstructor.mapping[8, :]) == 1.0
            assert np.sum(reconstructor.mapping[9, :]) == 1.0
            assert np.sum(reconstructor.mapping[10, :]) == 1.0
            assert np.sum(reconstructor.mapping[11, :]) == 1.0
            assert np.sum(reconstructor.mapping[12, :]) == 1.0
            assert np.sum(reconstructor.mapping[13, :]) == 1.0
            assert np.sum(reconstructor.mapping[14, :]) == 1.0

            assert np.sum(np.diag(reconstructor.regularization)) == 16.00000005
            assert np.sum(reconstructor.regularization) - np.sum(np.diag(reconstructor.regularization)) == -16.0

            assert set(reconstructor.image_to_pix) == set(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]))
            assert set(reconstructor.sub_to_pix) == set(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]))

        def test__5_simple_grid__include_sub_grid__sets_up_correct_reconstructor(self):

            pix_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            pix_border = mask.ImageGridBorder(arr=np.array([0, 1, 3, 4]))

            cluster_to_image = np.array([0, 1, 2, 3, 4])
            image_to_cluster = np.array([0, 1, 2, 3, 4])
            cluster_mask = MockSparseMask(sparse_to_image=cluster_to_image, image_to_sparse=image_to_cluster)

            pix_sub_grid = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0],
                                     [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
                                     [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.0, 0.0]])

            pix_sub_border = mask.SubGridBorder(arr=np.array([0, 1, 2, 4, 5, 6, 12, 13, 14, 16, 17, 18]))

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

            grids = MockGridCollection(image=pix_grid,
                                       sub=MockSubGridCoords(pix_sub_grid, sub_to_image,
                                                                   sub_grid_size=2))

            borders = MockBorderCollection(image=pix_border, sub=pix_sub_border)

            pix = pixelization.AmorphousRegConst(pixels=5, regularization_coefficients=(1.0,))

            reconstructor = pix.reconstructor_from_pix_grids(grids, borders, cluster_mask)

            assert np.sum(reconstructor.mapping) == 5.0

            assert np.sum(reconstructor.mapping[0, :]) == 1.0
            assert np.sum(reconstructor.mapping[1, :]) == 1.0
            assert np.sum(reconstructor.mapping[2, :]) == 1.0
            assert np.sum(reconstructor.mapping[3, :]) == 1.0
            assert np.sum(reconstructor.mapping[4, :]) == 1.0

            assert np.sum(reconstructor.mapping[:, 0]) or np.sum(reconstructor.mapping[:, 1]) or np.sum(
                reconstructor.mapping[:, 2]) or np.sum(reconstructor.mapping[:, 3]) or np.sum(
                reconstructor.mapping[:, 4]) == 0.75

            assert np.sum(np.diag(reconstructor.regularization)) == 16.00000005
            assert np.sum(reconstructor.regularization) - np.sum(np.diag(reconstructor.regularization)) == -16.0

            assert set(reconstructor.image_to_pix) == set(np.array([0, 1, 2, 3, 4]))
            assert set(reconstructor.sub_to_pix) == set(
                np.array([0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 2]))

        def test__same_as_above_but_grid_requires_border_relocation(self):


            pix_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            pix_border = mask.ImageGridBorder(arr=np.array([0, 1, 3, 4]))

            cluster_to_image = np.array([0, 1, 2, 3, 4])
            image_to_cluster = np.array([0, 1, 2, 3, 4])
            cluster_mask = MockSparseMask(sparse_to_image=cluster_to_image, image_to_sparse=image_to_cluster)

            pix_sub_grid = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0],
                                     [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
                                     [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.0, 0.0]])

            pix_sub_border = mask.SubGridBorder(arr=np.array([0, 4, 12, 16]))

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

            grids = MockGridCollection(image=pix_grid,
                                       sub=MockSubGridCoords(pix_sub_grid, sub_to_image,
                                                                   sub_grid_size=2))

            borders = MockBorderCollection(image=pix_border, sub=pix_sub_border)

            pix = pixelization.AmorphousRegConst(pixels=5, regularization_coefficients=(1.0,))

            reconstructor = pix.reconstructor_from_pix_grids(grids, borders, cluster_mask)

            assert np.sum(reconstructor.mapping) == 5.0

            assert np.sum(reconstructor.mapping[0, :]) == 1.0
            assert np.sum(reconstructor.mapping[1, :]) == 1.0
            assert np.sum(reconstructor.mapping[2, :]) == 1.0
            assert np.sum(reconstructor.mapping[3, :]) == 1.0
            assert np.sum(reconstructor.mapping[4, :]) == 1.0

            assert np.sum(reconstructor.mapping[:, 0]) or np.sum(reconstructor.mapping[:, 1]) or np.sum(
                reconstructor.mapping[:, 2]) or np.sum(reconstructor.mapping[:, 3]) or np.sum(
                reconstructor.mapping[:, 4]) == 0.75

            assert np.sum(np.diag(reconstructor.regularization)) == 16.00000005
            assert np.sum(reconstructor.regularization) - np.sum(np.diag(reconstructor.regularization)) == -16.0

            assert set(reconstructor.image_to_pix) == set(np.array([0, 1, 2, 3, 4]))
            assert set(reconstructor.sub_to_pix) == set(
                np.array([0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 2]))


            
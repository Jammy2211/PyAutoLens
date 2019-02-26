import numpy as np
import pytest
import scipy.spatial

from autolens.data.array import grids, mask
from autolens.model.inversion.util import pixelization_util
from autolens.model.inversion import regularization
from autolens.model.galaxy import galaxy as g


class TestRectangular:

    class TestNeighbors:

        def test__3x3_grid(self):

            # |0|1|2|
            # |3|4|5|
            # |6|7|8|

            pixel_neighbors, pixel_neighbors_size = pixelization_util.rectangular_neighbors_from_shape(shape=(3, 3))

            assert (pixel_neighbors[0] == [1, 3, -1, -1]).all()
            assert (pixel_neighbors[1] == [0, 2, 4, -1]).all()
            assert (pixel_neighbors[2] == [1, 5, -1, -1]).all()
            assert (pixel_neighbors[3] == [0, 4, 6, -1]).all()
            assert (pixel_neighbors[4] == [1, 3, 5, 7]).all()
            assert (pixel_neighbors[5] == [2, 4, 8, -1]).all()
            assert (pixel_neighbors[6] == [3, 7, -1, -1]).all()
            assert (pixel_neighbors[7] == [4, 6, 8, -1]).all()
            assert (pixel_neighbors[8] == [5, 7, -1, -1]).all()

            assert (pixel_neighbors_size == np.array([2, 3, 2, 3, 4, 3, 2, 3, 2])).all()

        def test__3x4_grid(self):

            # |0|1| 2| 3|
            # |4|5| 6| 7|
            # |8|9|10|11|

            pixel_neighbors, pixel_neighbors_size = pixelization_util.rectangular_neighbors_from_shape(shape=(3, 4))

            assert (pixel_neighbors[0] == [1, 4, -1, -1]).all()
            assert (pixel_neighbors[1] == [0, 2, 5, -1]).all()
            assert (pixel_neighbors[2] == [1, 3, 6, -1]).all()
            assert (pixel_neighbors[3] == [2, 7, -1, -1]).all()
            assert (pixel_neighbors[4] == [0, 5, 8, -1]).all()
            assert (pixel_neighbors[5] == [1, 4, 6, 9]).all()
            assert (pixel_neighbors[6] == [2, 5, 7, 10]).all()
            assert (pixel_neighbors[7] == [3, 6, 11, -1]).all()
            assert (pixel_neighbors[8] == [4, 9, -1, -1]).all()
            assert (pixel_neighbors[9] == [5, 8, 10, -1]).all()
            assert (pixel_neighbors[10] == [6, 9, 11, -1]).all()
            assert (pixel_neighbors[11] == [7, 10, -1, -1]).all()

            assert (pixel_neighbors_size == np.array([2, 3, 3, 2, 3, 4, 4, 3, 2, 3, 3, 2])).all()

        def test__4x3_grid(self):

            # |0| 1| 2|
            # |3| 4| 5|
            # |6| 7| 8|
            # |9|10|11|

            pixel_neighbors, pixel_neighbors_size = pixelization_util.rectangular_neighbors_from_shape(shape=(4, 3))

            assert (pixel_neighbors[0] == [1, 3, -1, -1]).all()
            assert (pixel_neighbors[1] == [0, 2, 4, -1]).all()
            assert (pixel_neighbors[2] == [1, 5, -1, -1]).all()
            assert (pixel_neighbors[3] == [0, 4, 6, -1]).all()
            assert (pixel_neighbors[4] == [1, 3, 5, 7]).all()
            assert (pixel_neighbors[5] == [2, 4, 8, -1]).all()
            assert (pixel_neighbors[6] == [3, 7, 9, -1]).all()
            assert (pixel_neighbors[7] == [4, 6, 8, 10]).all()
            assert (pixel_neighbors[8] == [5, 7, 11, -1]).all()
            assert (pixel_neighbors[9] == [6, 10, -1, -1]).all()
            assert (pixel_neighbors[10] == [7, 9, 11, -1]).all()
            assert (pixel_neighbors[11] == [8, 10, -1, -1]).all()

            assert (pixel_neighbors_size == np.array([2, 3, 2, 3, 4, 3, 3, 4, 3, 2, 3, 2])).all()

        def test__4x4_grid(self):

            # |0 | 1| 2| 3|
            # |4 | 5| 6| 7|
            # |8 | 9|10|11|
            # |12|13|14|15|

            pixel_neighbors, pixel_neighbors_size = pixelization_util.rectangular_neighbors_from_shape(shape=(4, 4))

            assert (pixel_neighbors[0] == [1, 4, -1, -1]).all()
            assert (pixel_neighbors[1] == [0, 2, 5, -1]).all()
            assert (pixel_neighbors[2] == [1, 3, 6, -1]).all()
            assert (pixel_neighbors[3] == [2, 7, -1, -1]).all()
            assert (pixel_neighbors[4] == [0, 5, 8, -1]).all()
            assert (pixel_neighbors[5] == [1, 4, 6, 9]).all()
            assert (pixel_neighbors[6] == [2, 5, 7, 10]).all()
            assert (pixel_neighbors[7] == [3, 6, 11, -1]).all()
            assert (pixel_neighbors[8] == [4, 9, 12, -1]).all()
            assert (pixel_neighbors[9] == [5, 8, 10, 13]).all()
            assert (pixel_neighbors[10] == [6, 9, 11, 14]).all()
            assert (pixel_neighbors[11] == [7, 10, 15, -1]).all()
            assert (pixel_neighbors[12] == [8, 13, -1, -1]).all()
            assert (pixel_neighbors[13] == [9, 12, 14, -1]).all()
            assert (pixel_neighbors[14] == [10, 13, 15, -1]).all()
            assert (pixel_neighbors[15] == [11, 14, -1, -1]).all()

            assert (pixel_neighbors_size == np.array([2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 2, 3, 3, 2])).all()


class TestVoronoi:

    class TestNeighbors:

        def test__points_in_x_cross_shape__neighbors_of_each_pixel_correct(self):

            points = np.array([[1.0, -1.0], [1.0, 1.0],
                               [0.0, 0.0],
                               [-1.0, -1.0], [-1.0, 1.0]])

            voronoi = scipy.spatial.Voronoi(points, qhull_options='Qbb Qc Qx Qm')
            pixel_neighbors, pixel_neighbors_size = pixelization_util.voronoi_neighbors_from_pixels_and_ridge_points(
                pixels=5, ridge_points=np.array(voronoi.ridge_points))

            assert set(pixel_neighbors[0]) == {1, 2, 3, -1}
            assert set(pixel_neighbors[1]) == {0, 2, 4, -1}
            assert set(pixel_neighbors[2]) == {0, 1, 3, 4}
            assert set(pixel_neighbors[3]) == {0, 2, 4, -1}
            assert set(pixel_neighbors[4]) == {1, 2, 3, -1}

            assert (pixel_neighbors_size == np.array([3, 3, 4, 3, 3])).all()

        def test__9_points_in_square___neighbors_of_each_pixel_correct(self):

            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array([[2.0, 0.0], [2.0, 1.0], [2.0, 2.0],
                               [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
                               [0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])

            voronoi = scipy.spatial.Voronoi(points, qhull_options='Qbb Qc Qx Qm')
            pixel_neighbors, pixel_neighbors_size = pixelization_util.voronoi_neighbors_from_pixels_and_ridge_points(
                pixels=9, ridge_points=np.array(voronoi.ridge_points))

            assert set(pixel_neighbors[0]) == {1, 3, -1, -1}
            assert set(pixel_neighbors[1]) == {0, 2, 4, -1}
            assert set(pixel_neighbors[2]) == {1, 5, -1, -1}
            assert set(pixel_neighbors[3]) == {0, 4, 6, -1}
            assert set(pixel_neighbors[4]) == {1, 3, 5, 7}
            assert set(pixel_neighbors[5]) == {2, 4, 8, -1}
            assert set(pixel_neighbors[6]) == {3, 7, -1, -1}
            assert set(pixel_neighbors[7]) == {4, 6, 8, -1}
            assert set(pixel_neighbors[8]) == {5, 7, -1, -1}

            assert (pixel_neighbors_size == np.array([2, 3, 2, 3, 4, 3, 2, 3, 2])).all()
import os

import numpy as np
import pytest

from autolens import exc
from autolens.data.array.util import array_util, grid_util
from autolens.data.array import mask as msk
from autolens.data.array import scaled_array

test_data_dir = "{}/../../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name="array_grid")
def make_array_grid():
    return scaled_array.ScaledSquarePixelArray(np.zeros((5, 5)), pixel_scale=0.5)


@pytest.fixture(name="array_grid_rectangular")
def make_array_grid_rectangular():
    return scaled_array.ScaledRectangularPixelArray(np.zeros((5, 5)), pixel_scales=(1.0, 0.5))


class TestArrayGeometry:


    class TestArrayAndTuples:

        def test__square_pixel_array__input_data_grid_3x3__centre_is_origin(self):

            data_grid = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)

            assert data_grid.pixel_scale == 1.0
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arcsec == pytest.approx((3.0, 3.0))
            assert data_grid.arc_second_maxima == (1.5, 1.5)
            assert data_grid.arc_second_minima == (-1.5, -1.5)
            assert (data_grid == np.ones((3, 3))).all()

        def test__square_pixel_array__input_data_grid_rectangular__change_origin(self):

            data_grid = scaled_array.ScaledSquarePixelArray(array=np.ones((4, 3)), pixel_scale=0.1, origin=(1.0, 1.0))

            assert (data_grid == np.ones((4, 3))).all()
            assert data_grid.pixel_scale == 0.1
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arcsec == pytest.approx((0.4, 0.3))
            assert data_grid.arc_second_maxima == pytest.approx((1.2, 1.15), 1e-4)
            assert data_grid.arc_second_minima == pytest.approx((0.8, 0.85), 1e-4)

            data_grid = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 4)), pixel_scale=0.1)

            assert (data_grid == np.ones((3, 4))).all()
            assert data_grid.pixel_scale == 0.1
            assert data_grid.shape == (3, 4)
            assert data_grid.central_pixel_coordinates == (1.0, 1.5)
            assert data_grid.shape_arcsec == pytest.approx((0.3, 0.4))
            assert data_grid.arc_second_maxima == pytest.approx((0.15, 0.2), 1e-4)
            assert data_grid.arc_second_minima == pytest.approx((-0.15, -0.2), 1e-4)

        def test__rectangular_pixel_grid__input_data_grid_3x3(self):

            data_grid = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 3)), pixel_scales=(2.0, 1.0))

            assert data_grid == pytest.approx(np.ones((3, 3)), 1e-4)
            assert data_grid.pixel_scales == (2.0, 1.0)
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arcsec == pytest.approx((6.0, 3.0))
            assert data_grid.arc_second_maxima == pytest.approx((3.0, 1.5), 1e-4)
            assert data_grid.arc_second_minima == pytest.approx((-3.0, -1.5), 1e-4)

        def test__rectangular_pixel_grid__input_data_grid_rectangular(self):

            data_grid = scaled_array.ScaledRectangularPixelArray(array=np.ones((4, 3)), pixel_scales=(0.2, 0.1))

            assert data_grid == pytest.approx(np.ones((4, 3)), 1e-4)
            assert data_grid.pixel_scales == (0.2, 0.1)
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arcsec == pytest.approx((0.8, 0.3), 1e-3)
            assert data_grid.arc_second_maxima == pytest.approx((0.4, 0.15), 1e-4)
            assert data_grid.arc_second_minima == pytest.approx((-0.4, -0.15), 1e-4)

            data_grid = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 4)), pixel_scales=(0.1, 0.2))

            assert data_grid == pytest.approx(np.ones((3, 4)), 1e-4)
            assert data_grid.pixel_scales == (0.1, 0.2)
            assert data_grid.shape == (3, 4)
            assert data_grid.central_pixel_coordinates == (1.0, 1.5)
            assert data_grid.shape_arcsec == pytest.approx((0.3, 0.8), 1e-3)
            assert data_grid.arc_second_maxima == pytest.approx((0.15, 0.4), 1e-4)
            assert data_grid.arc_second_minima == pytest.approx((-0.15, -0.4), 1e-4)

        def test__rectangular_pixel_array__input_data_grid_3x3__centre_is_yminus1_xminuss2(self):

            data_grid = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 3)), pixel_scales=(2.0, 1.0),
                                                                 origin=(-1.0, -2.0))

            assert data_grid == pytest.approx(np.ones((3, 3)), 1e-4)
            assert data_grid.pixel_scales == (2.0, 1.0)
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arcsec == pytest.approx((6.0, 3.0))

            assert data_grid.origin == (-1.0, -2.0)
            assert data_grid.arc_second_maxima == pytest.approx((2.0, -0.5), 1e-4)
            assert data_grid.arc_second_minima == pytest.approx((-4.0, -3.5), 1e-4)


    class TestCentralPixel:

        def test__square_pixel_grid(self):

            grid = scaled_array.ScaledSquarePixelArray(np.zeros((3, 3)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (1, 1)

            grid = scaled_array.ScaledSquarePixelArray(np.zeros((4, 4)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (1.5, 1.5)

            grid = scaled_array.ScaledSquarePixelArray(np.zeros((5, 3)), pixel_scale=0.1, origin=(1.0, 2.0))
            assert grid.central_pixel_coordinates == (2.0, 1.0)

        def test__rectangular_pixel_grid(self):

            grid = scaled_array.ScaledRectangularPixelArray(np.zeros((3, 3)), pixel_scales=(2.0, 1.0))
            assert grid.central_pixel_coordinates == (1, 1)

            grid = scaled_array.ScaledRectangularPixelArray(np.zeros((4, 4)), pixel_scales=(2.0, 1.0))
            assert grid.central_pixel_coordinates == (1.5, 1.5)

            grid = scaled_array.ScaledRectangularPixelArray(np.zeros((5, 3)), pixel_scales=(2.0, 1.0), origin=(1.0, 2.0))
            assert grid.central_pixel_coordinates == (2, 1)


    class TestGrids:

        def test__square_pixel_grid__grid_2d__compare_to_array_util(self):

            grid_2d_util = grid_util.regular_grid_2d_from_shape_pixel_scales_and_origin(shape=(4, 7),
                                                                                        pixel_scales=(0.56, 0.56))

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((4, 7)), pixel_scale=0.56)

            assert sca.grid_2d == pytest.approx(grid_2d_util, 1e-4)

        def test__square_pixel_grid__array_3x3__sets_up_arc_second_grid(self):
            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((3, 3)), pixel_scale=1.0)

            assert (sca.grid_2d == np.array([[[1., -1.], [1., 0.], [1., 1.]],
                                             [[0., -1.], [0., 0.], [0., 1.]],
                                             [[-1., -1.], [-1., 0.], [-1., 1.]]])).all()

        def test__square_pixel_grid__grid_1d__compare_to_array_util(self):

            grid_1d_util = grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=(4, 7),
                                                                                        pixel_scales=(0.56, 0.56))

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((4, 7)), pixel_scale=0.56)

            assert sca.grid_1d == pytest.approx(grid_1d_util, 1e-4)

        def test__square_pixel_grid__nonzero_centres__compure_to_array_util(self):

            grid_2d_util = grid_util.regular_grid_2d_from_shape_pixel_scales_and_origin(shape=(4, 7),
                                                                                        pixel_scales=(0.56, 0.56),
                                                                                        origin=(1.0, 3.0))

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((4, 7)), pixel_scale=0.56, origin=(1.0, 3.0))

            assert sca.grid_2d == pytest.approx(grid_2d_util, 1e-4)

            grid_1d_util = grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=(4, 7),
                                                                                        pixel_scales=(0.56, 0.56),
                                                                                        origin=(-1.0, -4.0))

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((4, 7)), pixel_scale=0.56, origin=(-1.0, -4.0))

            assert sca.grid_1d == pytest.approx(grid_1d_util, 1e-4)

        def test__rectangular_pixel_grid__grid_2d__compare_to_array_util(self):
            grid_2d_util = grid_util.regular_grid_2d_from_shape_pixel_scales_and_origin(shape=(4, 7),
                                                                                        pixel_scales=(0.8, 0.56))

            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((4, 7)), pixel_scales=(0.8, 0.56))

            assert sca.grid_2d == pytest.approx(grid_2d_util, 1e-4)

        def test__rectangular_pixel_grid__array_3x3__sets_up_arcsecond_grid(self):
            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((3, 3)), pixel_scales=(1.0, 2.0))

            assert (sca.grid_2d == np.array([[[1., -2.], [1., 0.], [1., 2.]],
                                             [[0., -2.], [0., 0.], [0., 2.]],
                                             [[-1., -2.], [-1., 0.], [-1., 2.]]])).all()

        def test__rectangular_pixel_grid__grid_1d__compare_to_array_util(self):

            grid_1d_util = grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=(4, 7),
                                                                                        pixel_scales=(0.8, 0.56))

            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((4, 7)), pixel_scales=(0.8, 0.56))

            assert sca.grid_1d == pytest.approx(grid_1d_util, 1e-4)

        def test__rectangular_pixel_grid__nonzero_centres__compure_to_array_util(self):

            grid_2d_util = grid_util.regular_grid_2d_from_shape_pixel_scales_and_origin(shape=(4, 7),
                                                                                        pixel_scales=(0.8, 0.56),
                                                                                        origin=(1.0, 2.0))

            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((4, 7)), pixel_scales=(0.8, 0.56),
                                                           origin=(1.0, 2.0))

            assert sca.grid_2d == pytest.approx(grid_2d_util, 1e-4)

            grid_1d_util = grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=(4, 7),
                                                                                        pixel_scales=(0.8, 0.56),
                                                                                        origin=(-1.0, -4.0))

            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((4, 7)), pixel_scales=(0.8, 0.56),
                                                           origin=(-1.0, -4.0))

            assert sca.grid_1d == pytest.approx(grid_1d_util, 1e-4)


    class TestConversion:

        def test__arc_second_coordinates_to_pixel_coordinates__arcsec_are_pixel_centres(self):

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((2, 2)), pixel_scale=2.0)

            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(1.0, -1.0)) == (0, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(1.0, 1.0)) == (0, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-1.0, -1.0)) == (1, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-1.0, 1.0)) == (1, 1)

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((3, 3)), pixel_scale=3.0)

            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(3.0, -3.0)) == (0, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(3.0, 0.0)) == (0, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(3.0, 3.0)) == (0, 2)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.0, -3.0)) == (1, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.0, 0.0)) == (1, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.0, 3.0)) == (1, 2)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-3.0, -3.0)) == (2, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-3.0, 0.0)) == (2, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-3.0, 3.0)) == (2, 2)

        def test__arc_second_coordinates_to_pixel_coordinates__arcsec_are_pixel_corners(self):

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((2, 2)), pixel_scale=2.0)

            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(1.99, -1.99)) == (0, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(1.99, -0.01)) == (0, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.01, -1.99)) == (0, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.01, -0.01)) == (0, 0)

            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(2.01, 0.01)) == (0, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(2.01, 1.99)) == (0, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.01, 0.01)) == (0, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.01, 1.99)) == (0, 1)

            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-0.01, -1.99)) == (1, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-0.01, -0.01)) == (1, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-1.99, -1.99)) == (1, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-1.99, -0.01)) == (1, 0)

            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-0.01, 0.01)) == (1, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-0.01, 1.99)) == (1, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-1.99, 0.01)) == (1, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-1.99, 1.99)) == (1, 1)

        def test__arc_second_coordinates_to_pixel_coordinates__arcsec_are_pixel_centres__nonzero_centre(self):

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((2, 2)), pixel_scale=2.0, origin=(1.0, 1.0))

            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(2.0, 0.0)) == (0, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(2.0, 2.0)) == (0, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.0, 0.0)) == (1, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.0, 2.0)) == (1, 1)

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((3, 3)), pixel_scale=3.0, origin=(3.0, 3.0))

            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(6.0, 0.0)) == (0, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(6.0, 3.0)) == (0, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(6.0, 6.0)) == (0, 2)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(3.0, 0.0)) == (1, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(3.0, 3.0)) == (1, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(3.0, 6.0)) == (1, 2)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.0, 0.0)) == (2, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.0, 3.0)) == (2, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.0, 6.0)) == (2, 2)

        def test__arc_second_coordinates_to_pixel_coordinates__arcsec_are_pixel_corners__nonzero_centre(self):

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((2, 2)), pixel_scale=2.0, origin=(1.0, 1.0))

            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(2.99, -0.99)) == (0, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(2.99, 0.99)) == (0, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(1.01, -0.99)) == (0, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(1.01, 0.99)) == (0, 0)

            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(3.01, 1.01)) == (0, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(3.01, 2.99)) == (0, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(1.01, 1.01)) == (0, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(1.01, 2.99)) == (0, 1)

            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.99, -0.99)) == (1, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.99, 0.99)) == (1, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-0.99, -0.99)) == (1, 0)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-0.99, 0.99)) == (1, 0)

            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.99, 1.01)) == (1, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(0.99, 2.99)) == (1, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-0.99, 1.01)) == (1, 1)
            assert sca.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=(-0.99, 2.99)) == (1, 1)

        def test__square_pixel_grid__1d_arc_second_grid_to_1d_pixel_centred_grid__same_as_imaging_util(self):

            grid_arcsec = np.array([[0.5, -0.5], [0.5, 0.5],
                                         [-0.5, -0.5], [-0.5, 0.5]])

            grid_pixels_util = grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(
                grid_arcsec_1d=grid_arcsec,
                shape=(2, 2),
                pixel_scales=(2.0, 2.0))

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((2, 2)), pixel_scale=2.0)

            grid_pixels = sca.grid_arcsec_to_grid_pixel_centres(grid_arcsec=grid_arcsec)

            assert (grid_pixels == grid_pixels_util).all()

        def test__square_pixel_grid__1d_arc_second_grid_to_1d_pixel_indexes_grid__same_as_imaging_util(self):

            grid_arcsec = np.array([[1.0, -1.0], [1.0, 1.0],
                                         [-1.0, -1.0], [-1.0, 1.0]])

            grid_pixel_indexes_util = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(
                grid_arcsec_1d=grid_arcsec,
                shape=(2, 2),
                pixel_scales=(2.0, 2.0))

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((2, 2)), pixel_scale=2.0)

            grid_pixel_indexes = sca.grid_arcsec_to_grid_pixel_indexes(grid_arcsec=grid_arcsec)

            assert (grid_pixel_indexes == grid_pixel_indexes_util).all()

        def test__rectangular_pixel_grid__1d_arc_second_grid_to_1d_pixel_centred_grid__same_as_imaging_util(self):
            grid_arcsec = np.array([[1.0, -2.0], [1.0, 2.0],
                                         [-1.0, -2.0], [-1.0, 2.0]])

            grid_pixels_util = grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(
                grid_arcsec_1d=grid_arcsec,
                shape=(2, 2),
                pixel_scales=(7.0, 2.0))

            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((2, 2)),
                                                           pixel_scales=(7.0, 2.0))

            grid_pixels = sca.grid_arcsec_to_grid_pixel_centres(grid_arcsec=grid_arcsec)

            assert (grid_pixels == grid_pixels_util).all()

        def test__rectangular_pixel_grid__1d_arc_second_grid_to_1d_pixel_indexes_grid__same_as_imaging_util(self):
            grid_arcsec = np.array([[1.0, -2.0], [1.0, 2.0],
                                         [-1.0, -2.0], [-1.0, 2.0]])

            grid_pixels_util = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(
                grid_arcsec_1d=grid_arcsec,
                shape=(2, 2),
                pixel_scales=(2.0, 4.0))

            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((2, 2)),
                                                           pixel_scales=(2.0, 4.0))

            grid_pixels = sca.grid_arcsec_to_grid_pixel_indexes(grid_arcsec=grid_arcsec)

            assert (grid_pixels == grid_pixels_util).all()

        def test__rectangular_pixel_grid__1d_arc_second_grid_to_1d_pixel_grid__same_as_imaging_util(self):

            grid_arcsec = np.array([[1.0, -2.0], [1.0, 2.0],
                                         [-1.0, -2.0], [-1.0, 2.0]])

            grid_pixels_util = grid_util.grid_arcsec_1d_to_grid_pixels_1d(
                grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(2.0, 4.0))

            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((2, 2)),
                                                           pixel_scales=(2.0, 4.0))

            grid_pixels = sca.grid_arcsec_to_grid_pixels(grid_arcsec=grid_arcsec)

            assert (grid_pixels == grid_pixels_util).all()

        def test__square_pixel_grid__1d_pixel_grid_to_1d_pixel_centred_grid__same_as_imaging_util(self):

            grid_pixels = np.array([[0, 0], [0, 1],
                                    [1, 0], [1, 1]])

            grid_pixels_util = grid_util.grid_pixels_1d_to_grid_arcsec_1d(
                grid_pixels_1d=grid_pixels,
                shape=(2, 2),
                pixel_scales=(2.0, 2.0))

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((2, 2)), pixel_scale=2.0)

            grid_pixels = sca.grid_pixels_to_grid_arcsec(grid_pixels=grid_pixels)

            assert (grid_pixels == grid_pixels_util).all()

        def test__square_pixel_grid__1d_pixel_grid_to_1d_pixel_grid__same_as_imaging_util(self):

            grid_pixels = np.array([[0, 0], [0, 1],
                                    [1, 0], [1, 1]])

            grid_pixels_util = grid_util.grid_pixels_1d_to_grid_arcsec_1d(
                grid_pixels_1d=grid_pixels, shape=(2, 2), pixel_scales=(2.0, 2.0))

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((2, 2)), pixel_scale=2.0)

            grid_pixels = sca.grid_pixels_to_grid_arcsec(grid_pixels=grid_pixels)

            assert (grid_pixels == grid_pixels_util).all()

        def test__square_pixel_grid__grids_with_nonzero_centres__same_as_imaging_util(self):

            grid_arcsec = np.array([[1.0, -2.0], [1.0, 2.0],
                                         [-1.0, -2.0], [-1.0, 2.0]])

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((2, 2)), pixel_scale=2.0, origin=(1.0, 2.0))

            grid_pixels_util = grid_util.grid_arcsec_1d_to_grid_pixels_1d(
                grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(2.0, 2.0), origin=(1.0, 2.0))
            grid_pixels = sca.grid_arcsec_to_grid_pixels(grid_arcsec=grid_arcsec)
            assert (grid_pixels == grid_pixels_util).all()

            grid_pixels_util = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(
                grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(2.0, 2.0), origin=(1.0, 2.0))
            grid_pixels = sca.grid_arcsec_to_grid_pixel_indexes(grid_arcsec=grid_arcsec)
            assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

            grid_pixels_util = grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(
                grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(2.0, 2.0), origin=(1.0, 2.0))
            grid_pixels = sca.grid_arcsec_to_grid_pixel_centres(grid_arcsec=grid_arcsec)
            assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

            grid_pixels = np.array([[0, 0], [0, 1],
                                    [1, 0], [1, 1]])

            grid_arcsec_util = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=grid_pixels,
                                                                                    shape=(2, 2), pixel_scales=(2.0, 2.0), origin=(1.0, 2.0))

            grid_arcsec = sca.grid_pixels_to_grid_arcsec(grid_pixels=grid_pixels)

            assert (grid_arcsec == grid_arcsec_util).all()

        def test__rectangular_pixel_grid__grids_with_nonzero_centres__same_as_imaging_util(self):

            grid_arcsec = np.array([[1.0, -2.0], [1.0, 2.0],
                                         [-1.0, -2.0], [-1.0, 2.0]])

            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((2, 2)), pixel_scales=(2.0, 1.0), origin=(1.0, 2.0))

            grid_pixels_util = grid_util.grid_arcsec_1d_to_grid_pixels_1d(
                grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(2.0, 1.0), origin=(1.0, 2.0))
            grid_pixels = sca.grid_arcsec_to_grid_pixels(grid_arcsec=grid_arcsec)
            assert (grid_pixels == grid_pixels_util).all()

            grid_pixels_util = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(
                grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(2.0, 1.0), origin=(1.0, 2.0))
            grid_pixels = sca.grid_arcsec_to_grid_pixel_indexes(grid_arcsec=grid_arcsec)
            assert (grid_pixels == grid_pixels_util).all()

            grid_pixels_util = grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(
                grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(2.0, 1.0), origin=(1.0, 2.0))
            grid_pixels = sca.grid_arcsec_to_grid_pixel_centres(grid_arcsec=grid_arcsec)
            assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

            grid_pixels = np.array([[0, 0], [0, 1],
                                    [1, 0], [1, 1]])

            grid_arcsec_util = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=grid_pixels,
                                                                                    shape=(2, 2), pixel_scales=(2.0, 1.0), origin=(1.0, 2.0))

            grid_arcsec = sca.grid_pixels_to_grid_arcsec(grid_pixels=grid_pixels)

            assert (grid_arcsec == grid_arcsec_util).all()

    class TestTicks:

        def test__square_pixel_grid__yticks(self):

            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
            assert sca.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=0.5)
            assert sca.yticks == pytest.approx(np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3)

            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((6, 3)), pixel_scale=1.0)
            assert sca.yticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 1)), pixel_scale=1.0)
            assert sca.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        def test__square_pixel_grid__xticks(self):
            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
            assert sca.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=0.5)
            assert sca.xticks == pytest.approx(np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3)

            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 6)), pixel_scale=1.0)
            assert sca.xticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((1, 3)), pixel_scale=1.0)
            assert sca.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        def test__rectangular_pixel_grid__yticks(self):
            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 3)), pixel_scales=(1.0, 5.0))
            assert sca.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 3)), pixel_scales=(0.5, 5.0))
            assert sca.yticks == pytest.approx(np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3)

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((6, 3)), pixel_scales=(1.0, 5.0))
            assert sca.yticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 6)), pixel_scales=(1.0, 5.0))
            assert sca.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        def test__rectangular_pixel_grid__xticks(self):
            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 3)), pixel_scales=(5.0, 1.0))
            assert sca.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 3)), pixel_scales=(5.0, 0.5))
            assert sca.xticks == pytest.approx(np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3)

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 6)), pixel_scales=(5.0, 1.0))
            assert sca.xticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((6, 3)), pixel_scales=(5.0, 1.0))
            assert sca.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)


class TestArray:

    class TestResizing:

        def test__pad__compare_to_imaging_util(self):

            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledSquarePixelArray(array, pixel_scale=1.0)

            modified = array.resized_scaled_array_from_array(new_shape=(7, 7), new_centre_pixels=(1, 1))

            modified_util = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(7, 7), origin=(1, 1))

            assert type(modified) == scaled_array.ScaledSquarePixelArray
            assert (modified == modified_util).all()
            assert modified.pixel_scale == 1.0

        def test__trim__compare_to_imaging_util(self):

            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledSquarePixelArray(array, pixel_scale=1.0)

            modified = array.resized_scaled_array_from_array(new_shape=(3, 3), new_centre_pixels=(4, 4))

            modified_util = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(3, 3), origin=(4, 4))

            assert type(modified) == scaled_array.ScaledSquarePixelArray
            assert (modified == modified_util).all()
            assert modified.pixel_scale == 1.0

        def test__new_centre_is_in_arcsec(self):

            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledSquarePixelArray(array, pixel_scale=3.0)

            modified = array.resized_scaled_array_from_array(new_shape=(3, 3), new_centre_arcsec=(6.0, 6.0))
            modified_util = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(3, 3), origin=(0, 4))
            assert (modified == modified_util).all()

            modified = array.resized_scaled_array_from_array(new_shape=(3, 3), new_centre_arcsec=(7.49, 4.51))
            modified_util = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(3, 3), origin=(0, 4))
            assert (modified == modified_util).all()

            modified = array.resized_scaled_array_from_array(new_shape=(3, 3), new_centre_arcsec=(7.49, 7.49))
            modified_util = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(3, 3), origin=(0, 4))
            assert (modified == modified_util).all()

            modified = array.resized_scaled_array_from_array(new_shape=(3, 3), new_centre_arcsec=(4.51, 4.51))
            modified_util = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(3, 3), origin=(0, 4))
            assert (modified == modified_util).all()

            modified = array.resized_scaled_array_from_array(new_shape=(3, 3), new_centre_arcsec=(4.51, 7.49))
            modified_util = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(3, 3), origin=(0, 4))
            assert (modified == modified_util).all()


class TestScaledSquarePixelArray:

    class TestConstructors(object):

        def test__constructor(self, array_grid):
            # Does the array grid class correctly instantiate as an instance of ndarray?
            assert array_grid.shape == (5, 5)
            assert array_grid.pixel_scale == 0.5
            assert isinstance(array_grid, np.ndarray)
            assert isinstance(array_grid, scaled_array.ScaledSquarePixelArray)

        def test__init__input_data_grid_single_value__all_attributes_correct_including_data_inheritance(self):

            data_grid = scaled_array.ScaledSquarePixelArray.single_value(value=5.0, shape=(3, 3),
                                                                         pixel_scale=1.0, origin=(1.0, 1.0))

            assert (data_grid == 5.0 * np.ones((3, 3))).all()
            assert data_grid.pixel_scale == 1.0
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arcsec == pytest.approx((3.0, 3.0))
            assert data_grid.origin == (1.0, 1.0)

        def test__from_fits__input_data_grid_3x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=test_data_dir + '3x3_ones.fits', hdu=0,
                                                                                       pixel_scale=1.0, origin=(1.0, 1.0))

            assert (data_grid == np.ones((3, 3))).all()
            assert data_grid.pixel_scale == 1.0
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arcsec == pytest.approx((3.0, 3.0))
            assert data_grid.origin == (1.0, 1.0)

        def test__from_fits__input_data_grid_4x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=test_data_dir + '4x3_ones.fits', hdu=0,
                                                                                       pixel_scale=0.1)

            assert (data_grid == np.ones((4, 3))).all()
            assert data_grid.pixel_scale == 0.1
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arcsec == pytest.approx((0.4, 0.3))

        def test__zero_or_negative_pixel_scale__raises_exception(self):

            with pytest.raises(exc.ScaledArrayException):
                scaled_array.ScaledSquarePixelArray(array=np.ones((2,2)), pixel_scale=0.0)

            with pytest.raises(exc.ScaledArrayException):
                scaled_array.ScaledSquarePixelArray(array=np.ones((2,2)), pixel_scale=-0.5)

    class TestExtract:

        def test__mask_extract_2d_array__uses_the_limits_of_the_mask(self):
    
            array = np.array([[ 1.0,  2.0,  3.0,  4.0],
                              [ 5.0,  6.0,  7.0,  8.0],
                              [ 9.0, 10.0, 11.0, 12.0],
                              [13.0, 14.0, 15.0, 16.0]])

            array = scaled_array.ScaledSquarePixelArray(array=array, pixel_scale=1.0)

            mask = msk.Mask(array=np.array([[True,  True,  True, True],
                                            [True, False, False, True],
                                            [True, False, False, True],
                                            [True,  True,  True, True]]), pixel_scale=1.0)

            array_extracted = array.extract_scaled_array_around_mask(mask=mask, buffer=0)
            assert (array_extracted == np.array([[6.0,   7.0],
                                                 [10.0, 11.0]])).all()

            mask = msk.Mask(array=np.array([[True,  True,  True, True],
                                            [True, False, False, True],
                                            [True, False, False, False],
                                            [True,  True,  True, True]]), pixel_scale=1.0)

            array_extracted = array.extract_scaled_array_around_mask(mask=mask, buffer=0)
            assert (array_extracted == np.array([[6.0,   7.0,  8.0],
                                                 [10.0, 11.0, 12.0]])).all()

            mask = msk.Mask(array=np.array([[True,  True,  True, True],
                                            [True, False, False, True],
                                            [True, False, False, True],
                                            [True,  True, False, True]]), pixel_scale=1.0)

            array_extracted = array.extract_scaled_array_around_mask(mask=mask, buffer=0)
            assert (array_extracted == np.array([[6.0,   7.0],
                                                 [10.0, 11.0],
                                                 [14.0, 15.0]])).all()

            mask = msk.Mask(array=np.array([[True,  True,   True, True],
                                            [True, False,  False, True],
                                            [False, False, False, True],
                                            [True,  True,  True, True]]), pixel_scale=1.0)

            array_extracted = array.extract_scaled_array_around_mask(mask=mask, buffer=0)
            print(array_extracted)
            assert (array_extracted == np.array([[5.0,  6.0,  7.0],
                                                 [9.0, 10.0, 11.0]])).all()

            mask = msk.Mask(array=np.array([[True, False,  True, True],
                                            [True, False, False, True],
                                            [True, False, False, True],
                                            [True,  True,  True, True]]), pixel_scale=1.0)
    
            array_extracted = array.extract_scaled_array_around_mask(mask=mask, buffer=0)
            assert (array_extracted == np.array([[2.0,   3.0],
                                                 [6.0,   7.0],
                                                 [10.0, 11.0]])).all()

            mask = msk.Mask(array=np.array([[True,  True,  True, True],
                                            [True, False, False, True],
                                            [True, False, False, True],
                                            [True,  True,  True, True]]), pixel_scale=1.0)

            array_extracted = array.extract_scaled_array_around_mask(mask=mask, buffer=1)
            assert (array_extracted == np.array([[ 1.0,  2.0,  3.0,  4.0],
                                                 [ 5.0,  6.0,  7.0,  8.0],
                                                 [ 9.0, 10.0, 11.0, 12.0],
                                                 [13.0, 14.0, 15.0, 16.0]])).all()

    class TestBinUp:

        def test__compare_all_extract_methods_to_array_util(self):

            array_2d = np.array([[1.0, 6.0, 3.0, 7.0, 3.0, 2.0],
                                 [2.0, 5.0, 3.0, 7.0, 7.0, 7.0],
                                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

            array_2d = scaled_array.ScaledSquarePixelArray(array=array_2d, pixel_scale=0.1)

            binned_up_array_2d_util = array_util.bin_up_array_2d_using_mean(array_2d=array_2d, bin_up_factor=4)
            binned_up_scaled_array = array_2d.binned_up_array_from_array(bin_up_factor=4, method='mean')
            assert (binned_up_array_2d_util == binned_up_scaled_array).all()

            binned_up_array_2d_util = array_util.bin_up_array_2d_using_quadrature(array_2d=array_2d, bin_up_factor=4)
            binned_up_scaled_array = array_2d.binned_up_array_from_array(bin_up_factor=4, method='quadrature')
            assert (binned_up_array_2d_util == binned_up_scaled_array).all()

            binned_up_array_2d_util = array_util.bin_up_array_2d_using_sum(array_2d=array_2d, bin_up_factor=4)
            binned_up_scaled_array = array_2d.binned_up_array_from_array(bin_up_factor=4, method='sum')
            assert (binned_up_array_2d_util == binned_up_scaled_array).all()

        def test__pixel_scale_of_scaled_arrays_are_updated(self):

            array_2d = np.array([[1.0, 6.0, 3.0, 7.0, 3.0, 2.0],
                                 [2.0, 5.0, 3.0, 7.0, 7.0, 7.0],
                                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

            array_2d = scaled_array.ScaledSquarePixelArray(array=array_2d, pixel_scale=0.1)

            binned_up_scaled_array = array_2d.binned_up_array_from_array(bin_up_factor=4, method='mean')
            assert binned_up_scaled_array.pixel_scale == pytest.approx(0.4, 1.0e-4)
            binned_up_scaled_array = array_2d.binned_up_array_from_array(bin_up_factor=6, method='mean')
            assert binned_up_scaled_array.pixel_scale == pytest.approx(0.6, 1.0e-4)

            binned_up_scaled_array = array_2d.binned_up_array_from_array(bin_up_factor=4, method='quadrature')
            assert binned_up_scaled_array.pixel_scale == pytest.approx(0.4, 1.0e-4)
            binned_up_scaled_array = array_2d.binned_up_array_from_array(bin_up_factor=6, method='quadrature')
            assert binned_up_scaled_array.pixel_scale == pytest.approx(0.6, 1.0e-4)

            binned_up_scaled_array = array_2d.binned_up_array_from_array(bin_up_factor=4, method='sum')
            assert binned_up_scaled_array.pixel_scale == pytest.approx(0.4, 1.0e-4)
            binned_up_scaled_array = array_2d.binned_up_array_from_array(bin_up_factor=6, method='sum')
            assert binned_up_scaled_array.pixel_scale == pytest.approx(0.6, 1.0e-4)

        def test__invalid_method__raises_exception(self):

            array_2d = np.array([[1.0, 6.0, 3.0, 7.0, 3.0, 2.0],
                                 [2.0, 5.0, 3.0, 7.0, 7.0, 7.0],
                                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

            array_2d = scaled_array.ScaledSquarePixelArray(array=array_2d, pixel_scale=0.1)
            with pytest.raises(exc.ImagingException):
                array_2d.binned_up_array_from_array(bin_up_factor=4, method='wrong')


class TestScaledRectangularPixelArray:

    class TestConstructors(object):

        def test__init__input_data_grid_single_value__all_attributes_correct_including_data_inheritance(self):

            data_grid = scaled_array.ScaledRectangularPixelArray.single_value(value=5.0, shape=(3, 3),
                                                                              pixel_scales=(2.0, 1.0), origin=(1.0, 1.0))

            assert (data_grid == 5.0 * np.ones((3, 3))).all()
            assert data_grid.pixel_scales == (2.0, 1.0)
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arcsec == pytest.approx((6.0, 3.0))
            assert data_grid.origin == (1.0, 1.0)

        def test__from_fits__input_data_grid_3x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledRectangularPixelArray.from_fits_with_pixel_scale(
                file_path=test_data_dir + '3x3_ones.fits', hdu=0, pixel_scales=(2.0, 1.0), origin=(1.0, 1.0))

            assert data_grid == pytest.approx(np.ones((3, 3)), 1e-4)
            assert data_grid.pixel_scales == (2.0, 1.0)
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arcsec == pytest.approx((6.0, 3.0))
            assert data_grid.origin == (1.0, 1.0)


        def test__from_fits__input_data_grid_4x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledRectangularPixelArray.from_fits_with_pixel_scale(
                file_path=test_data_dir + '4x3_ones.fits', hdu=0, pixel_scales=(0.2, 0.1))

            assert data_grid == pytest.approx(np.ones((4, 3)), 1e-4)
            assert data_grid.pixel_scales == (0.2, 0.1)
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arcsec == pytest.approx((0.8, 0.3))

        def test__zero_or_negative_pixel_scale__raises_exception(self):

            with pytest.raises(exc.ScaledArrayException):
                scaled_array.ScaledRectangularPixelArray(array=np.ones((2,2)), pixel_scales=(0.0, 0.5))

            with pytest.raises(exc.ScaledArrayException):
                scaled_array.ScaledRectangularPixelArray(array=np.ones((2,2)), pixel_scales=(0.5, 0.0))

            with pytest.raises(exc.ScaledArrayException):
                scaled_array.ScaledRectangularPixelArray(array=np.ones((2,2)), pixel_scales=(-0.5, 0.5))

            with pytest.raises(exc.ScaledArrayException):
                scaled_array.ScaledRectangularPixelArray(array=np.ones((2,2)), pixel_scales=(0.5, -0.5))
import os

import numpy as np
import pytest

from autolens.imaging import imaging_util
from autolens.imaging import scaled_array

test_data_dir = "{}/../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name="array_grid")
def make_array_grid():
    return scaled_array.ScaledSquarePixelArray(np.zeros((5, 5)), pixel_scale=0.5)


@pytest.fixture(name="array_grid_rectangular")
def make_array_grid_rectangular():
    return scaled_array.ScaledRectangularPixelArray(np.zeros((5, 5)), pixel_scales=(1.0, 0.5))


class TestArrayGeometry:

    class TestArrayAndTuples:

        def test__square_pixel_array__input_data_grid_3x3(self):

            data_grid = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)

            assert data_grid.pixel_scale == 1.0
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((3.0, 3.0))
            assert data_grid.arc_second_maxima == (1.5, 1.5)
            assert data_grid.arc_second_minima == (-1.5, -1.5)
            assert (data_grid == np.ones((3, 3))).all()

        def test__square_pixel_array__input_data_grid_rectangular(self):

            data_grid = scaled_array.ScaledSquarePixelArray(array=np.ones((4, 3)), pixel_scale=0.1)

            assert (data_grid == np.ones((4, 3))).all()
            assert data_grid.pixel_scale == 0.1
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((0.4, 0.3))
            assert data_grid.arc_second_maxima == pytest.approx((0.2, 0.15), 1e-4)
            assert data_grid.arc_second_minima == pytest.approx((-0.2, -0.15), 1e-4)

            data_grid = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 4)), pixel_scale=0.1)

            assert (data_grid == np.ones((3, 4))).all()
            assert data_grid.pixel_scale == 0.1
            assert data_grid.shape == (3, 4)
            assert data_grid.central_pixel_coordinates == (1.0, 1.5)
            assert data_grid.shape_arc_seconds == pytest.approx((0.3, 0.4))
            assert data_grid.arc_second_maxima == pytest.approx((0.15, 0.2), 1e-4)
            assert data_grid.arc_second_minima == pytest.approx((-0.15, -0.2), 1e-4)

        def test__rectangular_pixel_grid__input_data_grid_3x3(self):

            data_grid = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 3)), pixel_scales=(2.0, 1.0))

            assert data_grid == pytest.approx(np.ones((3, 3)), 1e-4)
            assert data_grid.pixel_scales == (2.0, 1.0)
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((6.0, 3.0))
            assert data_grid.arc_second_maxima == pytest.approx((3.0, 1.5), 1e-4)
            assert data_grid.arc_second_minima == pytest.approx((-3.0, -1.5), 1e-4)

        def test__rectangular_pixel_grid__input_data_grid_rectangular(self):

            data_grid = scaled_array.ScaledRectangularPixelArray(array=np.ones((4, 3)), pixel_scales=(0.2, 0.1))

            assert data_grid == pytest.approx(np.ones((4, 3)), 1e-4)
            assert data_grid.pixel_scales == (0.2, 0.1)
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((0.8, 0.3), 1e-3)
            assert data_grid.arc_second_maxima == pytest.approx((0.4, 0.15), 1e-4)
            assert data_grid.arc_second_minima == pytest.approx((-0.4, -0.15), 1e-4)

            data_grid = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 4)), pixel_scales=(0.1, 0.2))

            assert data_grid == pytest.approx(np.ones((3, 4)), 1e-4)
            assert data_grid.pixel_scales == (0.1, 0.2)
            assert data_grid.shape == (3, 4)
            assert data_grid.central_pixel_coordinates == (1.0, 1.5)
            assert data_grid.shape_arc_seconds == pytest.approx((0.3, 0.8), 1e-3)
            assert data_grid.arc_second_maxima == pytest.approx((0.15, 0.4), 1e-4)
            assert data_grid.arc_second_minima == pytest.approx((-0.15, -0.4), 1e-4)


    class TestCentralPixel:

        def test__square_pixel_grid(self):

            grid = scaled_array.ScaledSquarePixelArray(np.zeros((3, 3)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (1, 1)

            grid = scaled_array.ScaledSquarePixelArray(np.zeros((4, 4)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (1.5, 1.5)

            grid = scaled_array.ScaledSquarePixelArray(np.zeros((5, 3)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (2.0, 1.0)

        def test__rectangular_pixel_grid(self):

            grid = scaled_array.ScaledRectangularPixelArray(np.zeros((3, 3)), pixel_scales=(2.0, 1.0))
            assert grid.central_pixel_coordinates == (1, 1)

            grid = scaled_array.ScaledRectangularPixelArray(np.zeros((4, 4)), pixel_scales=(2.0, 1.0))
            assert grid.central_pixel_coordinates == (1.5, 1.5)

            grid = scaled_array.ScaledRectangularPixelArray(np.zeros((5, 3)), pixel_scales=(2.0, 1.0))
            assert grid.central_pixel_coordinates == (2, 1)


    class TestGrids:

        def test__square_pixel_grid__grid_2d__compare_to_array_util(self):

            grid_2d_util = imaging_util.image_grid_2d_from_shape_and_pixel_scales(shape=(4, 7),
                                                                                  pixel_scales=(0.56, 0.56))

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((4, 7)), pixel_scale=0.56)

            assert sca.grid_2d == pytest.approx(grid_2d_util, 1e-4)

        def test__square_pixel_grid__array_3x3__sets_up_arc_second_grid(self):
            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((3, 3)), pixel_scale=1.0)

            assert (sca.grid_2d == np.array([[[1., -1.], [1., 0.], [1., 1.]],
                                             [[0., -1.], [0., 0.], [0., 1.]],
                                             [[-1., -1.], [-1., 0.], [-1., 1.]]])).all()

        def test__square_pixel_grid__grid_1d__compare_to_array_util(self):

            grid_1d_util = imaging_util.image_grid_1d_from_shape_and_pixel_scales(shape=(4, 7),
                                                                                  pixel_scales=(0.56, 0.56))

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((4, 7)), pixel_scale=0.56)

            assert sca.grid_1d == pytest.approx(grid_1d_util, 1e-4)

        def test__rectangular_pixel_grid__grid_2d__compare_to_array_util(self):
            grid_2d_util = imaging_util.image_grid_2d_from_shape_and_pixel_scales(shape=(4, 7),
                                                                                  pixel_scales=(0.8, 0.56))

            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((4, 7)), pixel_scales=(0.8, 0.56))

            assert sca.grid_2d == pytest.approx(grid_2d_util, 1e-4)

        def test__rectangular_pixel_grid__array_3x3__sets_up_arcsecond_grid(self):
            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((3, 3)), pixel_scales=(1.0, 2.0))

            assert (sca.grid_2d == np.array([[[1., -2.], [1., 0.], [1., 2.]],
                                             [[0., -2.], [0., 0.], [0., 2.]],
                                             [[-1., -2.], [-1., 0.], [-1., 2.]]])).all()

        def test__rectangular_pixel_grid__grid_1d__compare_to_array_util(self):

            grid_1d_util = imaging_util.image_grid_1d_from_shape_and_pixel_scales(shape=(4, 7),
                                                                                  pixel_scales=(0.8, 0.56))

            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((4, 7)), pixel_scales=(0.8, 0.56))

            assert sca.grid_1d == pytest.approx(grid_1d_util, 1e-4)

    class TestConversion:

        def test__square_pixel_grid__1d_arc_second_grid_to_1d_pixel_centred_grid__same_as_imaging_util(self):
            grid_arc_seconds = np.array([[1.0, -2.0], [1.0, 2.0],
                                         [-1.0, -2.0], [-1.0, 2.0]])

            grid_arc_seconds_util = imaging_util.grid_arc_seconds_1d_to_grid_pixel_centres_1d(
                grid_arc_seconds=grid_arc_seconds,
                shape=(2, 2),
                pixel_scales=(2.0, 2.0))

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((2, 2)), pixel_scale=2.0)

            grid_arc_seconds = sca.grid_arc_seconds_to_grid_pixel_centres(grid_arc_seconds=grid_arc_seconds)

            assert (grid_arc_seconds == grid_arc_seconds_util).all()

        def test__square_pixel_grid__1d_arc_second_grid_to_1d_pixel_grid__same_as_imaging_util(self):

            grid_arc_seconds = np.array([[1.0, -2.0], [1.0, 2.0],
                                         [-1.0, -2.0], [-1.0, 2.0]])

            grid_arc_seconds_util = imaging_util.grid_arc_seconds_1d_to_grid_pixels_1d(
                grid_arc_seconds=grid_arc_seconds, shape=(2, 2), pixel_scales=(2.0, 2.0))

            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((2, 2)), pixel_scale=2.0)

            grid_arc_seconds = sca.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=grid_arc_seconds)

            assert (grid_arc_seconds == grid_arc_seconds_util).all()

        def test__rectangular_pixel_grid__1d_arc_second_grid_to_1d_pixel_centred_grid__same_as_imaging_util(self):
            grid_arc_seconds = np.array([[1.0, -2.0], [1.0, 2.0],
                                         [-1.0, -2.0], [-1.0, 2.0]])

            grid_arc_seconds_util = imaging_util.grid_arc_seconds_1d_to_grid_pixel_centres_1d(
                grid_arc_seconds=grid_arc_seconds,
                shape=(2, 2),
                pixel_scales=(7.0, 2.0))

            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((2, 2)),
                                                           pixel_scales=(7.0, 2.0))

            grid_arc_seconds = sca.grid_arc_seconds_to_grid_pixel_centres(grid_arc_seconds=grid_arc_seconds)

            assert (grid_arc_seconds == grid_arc_seconds_util).all()

        def test__rectangular_pixel_grid__1d_arc_second_grid_to_1d_pixel_grid__same_as_imaging_util(self):

            grid_arc_seconds = np.array([[1.0, -2.0], [1.0, 2.0],
                                         [-1.0, -2.0], [-1.0, 2.0]])

            grid_arc_seconds_util = imaging_util.grid_arc_seconds_1d_to_grid_pixels_1d(
                grid_arc_seconds=grid_arc_seconds, shape=(2, 2), pixel_scales=(7.0, 2.0))

            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((2, 2)),
                                                           pixel_scales=(7.0, 2.0))

            grid_arc_seconds = sca.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=grid_arc_seconds)

            assert (grid_arc_seconds == grid_arc_seconds_util).all()


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

        def test__pad_around_centre__compare_to_imaging_util(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledSquarePixelArray(array, pixel_scale=1.0)

            modified = array.pad_around_centre(new_shape=(7, 7))

            modified_util = imaging_util.pad_array_2d_around_centre(array_2d=array, new_shape=(7, 7))

            assert type(modified) == scaled_array.ScaledSquarePixelArray
            assert (modified == modified_util).all()
            assert modified.pixel_scale == 1.0

        def test__trim_around_centre__compare_to_imaging_util(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledSquarePixelArray(array, pixel_scale=1.0)

            modified = array.trim_around_centre(new_shape=(3, 3))

            modified_util = imaging_util.trim_array_2d_around_centre(array_2d=array, new_shape=(3, 3))

            assert type(modified) == scaled_array.ScaledSquarePixelArray
            assert (modified == modified_util).all()
            assert modified.pixel_scale == 1.0

        def test__trim_around_region__compare_to_imaging_util(self):
            array = np.ones((6, 6))
            array[4, 4] = 2.0

            array = scaled_array.ScaledSquarePixelArray(array, pixel_scale=1.0)

            modified = array.trim_around_region(y0=3, y1=6, x0=3, x1=6)

            modified_util = imaging_util.trim_array_2d_around_region(array_2d=array, y0=3, y1=6, x0=3, x1=6)

            assert type(modified) == scaled_array.ScaledSquarePixelArray
            assert (modified == modified_util).all()
            assert modified.pixel_scale == 1.0

        def test__resize_image_around_centre__smaller_shape_trims_image(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledSquarePixelArray(array, pixel_scale=1.0)

            modified = array.resize_around_centre(new_shape=(3, 3))

            modified_util = imaging_util.trim_array_2d_around_centre(array_2d=array, new_shape=(3, 3))

            assert type(modified) == scaled_array.ScaledSquarePixelArray
            assert (modified == modified_util).all()
            assert modified.pixel_scale == 1.0

        def test__resize_image_around_centre__large_shape_pads_image(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledSquarePixelArray(array, pixel_scale=1.0)

            modified = array.resize_around_centre(new_shape=(7, 7))

            modified_util = imaging_util.pad_array_2d_around_centre(array_2d=array, new_shape=(7, 7))

            assert type(modified) == scaled_array.ScaledSquarePixelArray
            assert (modified == modified_util).all()
            assert modified.pixel_scale == 1.0

        def test__works_for_rectangular_array(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledRectangularPixelArray(array, pixel_scales=(1.0, 2.0))

            modified = array.resize_around_centre(new_shape=(3, 3))

            modified_util = imaging_util.trim_array_2d_around_centre(array_2d=array, new_shape=(3, 3))

            assert type(modified) == scaled_array.ScaledRectangularPixelArray
            assert (modified == modified_util).all()
            assert modified.pixel_scales == (1.0, 2.0)

        def test__raises_error_if_one_dimension_trims_one_pads(self):
            array = np.ones((5, 5))
            array = scaled_array.ScaledSquarePixelArray(array, pixel_scale=1.0)

            with pytest.raises(ValueError):
                array.resize_around_centre(new_shape=(3, 6))

            with pytest.raises(ValueError):
                array.resize_around_centre(new_shape=(6, 3))


class TestScaledSquarePixelArray:

    class TestConstructors(object):

        def test__constructor(self, array_grid):
            # Does the array grid class correctly instantiate as an instance of ndarray?
            assert array_grid.shape == (5, 5)
            assert array_grid.pixel_scale == 0.5
            assert isinstance(array_grid, np.ndarray)
            assert isinstance(array_grid, scaled_array.ScaledSquarePixelArray)

        def test__init__input_data_grid_single_value__all_attributes_correct_including_data_inheritance(
                self):
            data_grid = scaled_array.ScaledSquarePixelArray.single_value(value=5.0, shape=(3, 3),
                                                                         pixel_scale=1.0)

            assert (data_grid == 5.0 * np.ones((3, 3))).all()
            assert data_grid.pixel_scale == 1.0
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((3.0, 3.0))

        def test__from_fits__input_data_grid_3x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=test_data_dir + '3x3_ones.fits', hdu=0,
                                                                                       pixel_scale=1.0)

            assert (data_grid == np.ones((3, 3))).all()
            assert data_grid.pixel_scale == 1.0
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((3.0, 3.0))

        def test__from_fits__input_data_grid_4x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=test_data_dir + '4x3_ones.fits', hdu=0,
                                                                                       pixel_scale=0.1)

            assert (data_grid == np.ones((4, 3))).all()
            assert data_grid.pixel_scale == 0.1
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((0.4, 0.3))


class TestScaledRectangularPixelArray:

    class TestConstructors(object):

        def test__init__input_data_grid_single_value__all_attributes_correct_including_data_inheritance(self):

            data_grid = scaled_array.ScaledRectangularPixelArray.single_value(value=5.0, shape=(3, 3),
                                                                              pixel_scales=(2.0, 1.0))

            assert (data_grid == 5.0 * np.ones((3, 3))).all()
            assert data_grid.pixel_scales == (2.0, 1.0)
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((6.0, 3.0))

        def test__from_fits__input_data_grid_3x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledRectangularPixelArray.from_fits_with_pixel_scale(
                file_path=test_data_dir + '3x3_ones.fits', hdu=0, pixel_scales=(2.0, 1.0))

            assert data_grid == pytest.approx(np.ones((3, 3)), 1e-4)
            assert data_grid.pixel_scales == (2.0, 1.0)
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((6.0, 3.0))

        def test__from_fits__input_data_grid_4x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledRectangularPixelArray.from_fits_with_pixel_scale(
                file_path=test_data_dir + '4x3_ones.fits', hdu=0, pixel_scales=(0.2, 0.1))

            assert data_grid == pytest.approx(np.ones((4, 3)), 1e-4)
            assert data_grid.pixel_scales == (0.2, 0.1)
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((0.8, 0.3))


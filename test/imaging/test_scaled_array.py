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
    return scaled_array.ScaledRectangularPixelArray(np.zeros((5, 5)), pixel_scales=(0.5, 1.0))


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
    
        def test__init__input_data_grid_3x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
    
            assert data_grid.pixel_scale == 1.0
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((3.0, 3.0))
            assert (data_grid == np.ones((3, 3))).all()
    
        def test__init__input_data_grid_4x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledSquarePixelArray(array=np.ones((4, 3)), pixel_scale=0.1)
    
            assert (data_grid == np.ones((4, 3))).all()
            assert data_grid.pixel_scale == 0.1
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((0.4, 0.3))
    
        def test__from_fits__input_data_grid_3x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledSquarePixelArray.from_fits_with_scale(file_path=test_data_dir + '3x3_ones.fits', hdu=0,
                                                                      pixel_scale=1.0)
    
            assert (data_grid == np.ones((3, 3))).all()
            assert data_grid.pixel_scale == 1.0
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((3.0, 3.0))
    
        def test__from_fits__input_data_grid_4x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledSquarePixelArray.from_fits_with_scale(file_path=test_data_dir + '4x3_ones.fits', hdu=0,
                                                                      pixel_scale=0.1)
    
            assert (data_grid == np.ones((4, 3))).all()
            assert data_grid.pixel_scale == 0.1
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((0.4, 0.3))
    
    
    class TestTrim:
    
        def test__trim_around_centre__compare_to_imaging_util(self):
    
            array = np.ones((5, 5))
            array[2, 2] = 2.0
    
            array = scaled_array.ScaledSquarePixelArray(array, pixel_scale=1.0)
    
            modified = array.trim_around_centre(new_shape=(3, 3))
    
            assert type(modified) == scaled_array.ScaledSquarePixelArray
            assert (modified == np.array([[1.0, 1.0, 1.0],
                                          [1.0, 2.0, 1.0],
                                          [1.0, 1.0, 1.0]])).all()
            assert modified.shape == (3, 3)
            assert modified.shape_arc_seconds == (3.0, 3.0)
    
        def test__trim_around_region__compare_to_imaging_util(self):
    
            array = np.ones((6, 6))
            array[4, 4] = 2.0

            array = scaled_array.ScaledSquarePixelArray(array, pixel_scale=1.0)
    
            modified = imaging_util.trim_array_2d_around_region(array_2d=array, x0=3, x1=6, y0=3, y1=6)
    
            assert type(modified) == scaled_array.ScaledSquarePixelArray
            assert (modified == np.array([[1.0, 1.0, 1.0],
                                          [1.0, 2.0, 1.0],
                                          [1.0, 1.0, 1.0]])).all()
            assert modified.shape == (3, 3)
            assert modified.shape_arc_seconds == (3.0, 3.0)
    
    
    class TestCentralPixel:
    
        def test__3x3_grid__central_pixel_is_1_and_1(self):
            grid = scaled_array.ScaledSquarePixelArray(np.zeros((3, 3)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (1, 1)
    
        def test__4x4_grid__central_pixel_is_1dot5_and_1dot5(self):
            grid = scaled_array.ScaledSquarePixelArray(np.zeros((4, 4)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (1.5, 1.5)
    
        def test__5x3_grid__central_pixel_is_2_and_1(self):
            grid = scaled_array.ScaledSquarePixelArray(np.zeros((5, 3)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (2, 1)
    
        def test__central_pixel_coordinates_5x5(self, array_grid):
            assert array_grid.central_pixel_coordinates == (2, 2)
    
    
    class TestConversion:
    
        def test__pixels_to_arcseconds(self, array_grid):
            assert array_grid.pixels_to_arc_seconds(1) == 0.5
    
        def test__arcseconds_to_pixels(self, array_grid):
            assert array_grid.arc_seconds_to_pixels(1) == 2
    
        def test__1d_pixel_grid_to_1d_arc_second_grid(self):
    
            array_grid = scaled_array.ScaledSquarePixelArray(np.zeros((2, 2)), pixel_scale=2.0)
    
            grid_pixels = np.array([[0,0], [0,1],
                                    [1,0], [1,1]])
    
            grid_arc_seconds = array_grid.grid_pixels_to_grid_arcseconds(grid_pixels=grid_pixels)
    
            assert (grid_arc_seconds == np.array([[-1.0, -1.0], [-1.0, 1.0],
                                                  [ 1.0, -1.0], [ 1.0, 1.0]])).all()
    
            array_grid = scaled_array.ScaledSquarePixelArray(np.zeros((3, 3)), pixel_scale=2.0)
    
            grid_pixels = np.array([[0,0], [0,1], [0,2],
                                    [1,0], [1,1], [1,2],
                                    [2,0], [2,1], [2,2]])
    
            grid_arc_seconds = array_grid.grid_pixels_to_grid_arcseconds(grid_pixels=grid_pixels)
    
            assert (grid_arc_seconds == np.array([[-2.0, -2.0], [-2.0, 0.0], [-2.0, 2.0],
                                                  [ 0.0, -2.0], [ 0.0, 0.0], [ 0.0, 2.0],
                                                  [ 2.0, -2.0], [ 2.0, 0.0], [ 2.0, 2.0]])).all()
    
        def test__2d_pixel_grid_to_2d_arc_second_grid(self):
    
            array_grid = scaled_array.ScaledSquarePixelArray(np.zeros((2, 2)), pixel_scale=2.0)
    
            grid_pixels = np.array([[[0,0], [0,1]],
                                    [[1,0], [1,1]]])
    
            grid_arc_seconds = array_grid.grid_pixels_to_grid_arcseconds(grid_pixels=grid_pixels)
    
            assert (grid_arc_seconds == np.array([[[-1.0, -1.0], [-1.0, 1.0]],
                                                  [[ 1.0, -1.0], [ 1.0, 1.0]]])).all()
    
            array_grid = scaled_array.ScaledSquarePixelArray(np.zeros((3, 3)), pixel_scale=2.0)
    
            grid_pixels = np.array([[[0,0], [0,1], [0,2]],
                                    [[1,0], [1,1], [1,2]],
                                    [[2,0], [2,1], [2,2]]])
    
            grid_arc_seconds = array_grid.grid_pixels_to_grid_arcseconds(grid_pixels=grid_pixels)
    
            assert (grid_arc_seconds == np.array([[[-2.0, -2.0], [-2.0, 0.0], [-2.0, 2.0]],
                                                  [[ 0.0, -2.0], [ 0.0, 0.0], [ 0.0, 2.0]],
                                                  [[ 2.0, -2.0], [ 2.0, 0.0], [ 2.0, 2.0]]])).all()
    
        def test__1d_arc_second_grid_to_1d_pixel_grid(self):
    
            array_grid = scaled_array.ScaledSquarePixelArray(np.zeros((2, 2)), pixel_scale=2.0)
    
            grid_arc_seconds = np.array([[-1.0, -1.0], [-1.0, 1.0],
                                         [ 1.0, -1.0], [ 1.0, 1.0]])
    
            grid_pixels = array_grid.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=grid_arc_seconds)
    
            assert (grid_pixels == np.array([[0,0], [0,1],
                                             [1,0], [1,1]])).all()
    
            array_grid = scaled_array.ScaledSquarePixelArray(np.zeros((3, 3)), pixel_scale=2.0)
    
            grid_arc_seconds = np.array([[-2.0, -2.0], [-2.0, 0.0], [-2.0, 2.0],
                                         [ 0.0, -2.0], [ 0.0, 0.0], [ 0.0, 2.0],
                                         [ 2.0, -2.0], [ 2.0, 0.0], [ 2.0, 2.0]])
    
            grid_pixels = array_grid.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=grid_arc_seconds)
    
            assert (grid_pixels == np.array([[0,0], [0,1], [0,2],
                                             [1,0], [1,1], [1,2],
                                             [2,0], [2,1], [2,2]])).all()
    
        def test__2d_arc_second_grid_to_2d_pixel_grid(self):
    
            array_grid = scaled_array.ScaledSquarePixelArray(np.zeros((2, 2)), pixel_scale=2.0)
    
            grid_arc_seconds = np.array([[[-1.0, -1.0], [-1.0, 1.0]],
                                         [[ 1.0, -1.0], [ 1.0, 1.0]]])
    
            grid_pixels = array_grid.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=grid_arc_seconds)
    
            assert (grid_pixels == np.array([[[0,0], [0,1]],
                                             [[1,0], [1,1]]])).all()
    
            array_grid = scaled_array.ScaledSquarePixelArray(np.zeros((3, 3)), pixel_scale=2.0)
    
            grid_arc_seconds = np.array([[[-2.0, -2.0], [-2.0, 0.0], [-2.0, 2.0]],
                                         [[ 0.0, -2.0], [ 0.0, 0.0], [ 0.0, 2.0]],
                                         [[ 2.0, -2.0], [ 2.0, 0.0], [ 2.0, 2.0]]])
    
            grid_pixels = array_grid.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=grid_arc_seconds)
    
            assert (grid_pixels == np.array([[[0,0], [0,1], [0,2]],
                                             [[1,0], [1,1], [1,2]],
                                             [[2,0], [2,1], [2,2]]])).all()
    
    
    class TestGrids:
    
        def test__grid_2d__compare_to_array_util(self):
            grid_2d_util = imaging_util.image_grid_2d_from_shape_and_pixel_scales(shape=(4, 7),
                                                                                  pixel_scales=(0.56, 0.56))
    
            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((4, 7)), pixel_scale=0.56)
    
            assert sca.grid_2d == pytest.approx(grid_2d_util, 1e-4)
    
        def test__array_3x3__sets_up_arcsecond_grid(self):
            sca = scaled_array.ScaledSquarePixelArray(array=np.zeros((3, 3)), pixel_scale=1.0)
    
            assert (sca.grid_2d == np.array([[[-1., -1.], [-1., 0.], [-1., 1.]],
                                             [[0., -1.], [0., 0.], [0., 1.]],
                                             [[1., -1.], [1., 0.], [1., 1.]]])).all()
    
    
    class TestTicks:
    
        def test__compute_xticks_property(self):
    
            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
            assert sca.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)
    
            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=0.5)
            assert sca.xticks == pytest.approx(np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3)
    
            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 6)), pixel_scale=1.0)
            assert sca.xticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)
    
            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((6, 3)), pixel_scale=1.0)
            assert sca.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)
    
        def test__compute_yticks_property(self):
    
            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
            assert sca.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)
    
            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=0.5)
            assert sca.yticks == pytest.approx(np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3)
    
            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((6, 3)), pixel_scale=1.0)
            assert sca.yticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)
    
            sca = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 6)), pixel_scale=1.0)
            assert sca.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)


class TestScaledRectangularPixelArray:
    
    class TestConstructors(object):

        def test__constructor(self, array_grid_rectangular):
            # Does the array grid class correctly instantiate as an instance of ndarray?
            assert array_grid_rectangular.shape == (5, 5)
            assert array_grid_rectangular.pixel_scales == (0.5, 1.0)
            assert isinstance(array_grid_rectangular, np.ndarray)
            assert isinstance(array_grid_rectangular, scaled_array.ScaledRectangularPixelArray)

        def test__init__input_data_grid_single_value__all_attributes_correct_including_data_inheritance(self):

            data_grid = scaled_array.ScaledRectangularPixelArray.single_value(value=5.0, shape=(3, 3),
                                                                         pixel_scales=(1.0, 2.0))

            assert (data_grid == 5.0 * np.ones((3, 3))).all()
            assert data_grid.pixel_scales == (1.0, 2.0)
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((3.0, 6.0))

        def test__init__input_data_grid_3x3__all_attributes_correct_including_data_inheritance(self):

            data_grid = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 3)), pixel_scales=(1.0, 2.0))

            assert data_grid.pixel_scales == (1.0, 2.0)
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((3.0, 6.0))
            assert (data_grid == np.ones((3, 3))).all()

        def test__init__input_data_grid_4x3__all_attributes_correct_including_data_inheritance(self):

            data_grid = scaled_array.ScaledRectangularPixelArray(array=np.ones((4, 3)), pixel_scales=(0.1, 0.2))

            assert (data_grid == np.ones((4, 3))).all()
            assert data_grid.pixel_scales == (0.1, 0.2)
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((0.4, 0.6))

        def test__from_fits__input_data_grid_3x3__all_attributes_correct_including_data_inheritance(self):

            data_grid = scaled_array.ScaledRectangularPixelArray.from_fits_with_scale(
                file_path=test_data_dir + '3x3_ones.fits', hdu=0, pixel_scales=(1.0, 2.0))

            assert (data_grid == np.ones((3, 3))).all()
            assert data_grid.pixel_scales == (1.0, 2.0)
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((3.0, 6.0))

        def test__from_fits__input_data_grid_4x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledRectangularPixelArray.from_fits_with_scale(
                file_path=test_data_dir + '4x3_ones.fits', hdu=0, pixel_scales=(0.1, 0.2))

            assert (data_grid == np.ones((4, 3))).all()
            assert data_grid.pixel_scales == (0.1, 0.2)
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((0.4, 0.6))

    class TestTrim:

        def test__trim_around_centre__compare_to_imaging_util(self):

            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledRectangularPixelArray(array, pixel_scales=(1.0, 2.0))

            modified = array.trim_around_centre(new_shape=(3, 3))

            assert type(modified) == scaled_array.ScaledRectangularPixelArray
            assert (modified == np.array([[1.0, 1.0, 1.0],
                                          [1.0, 2.0, 1.0],
                                          [1.0, 1.0, 1.0]])).all()
            assert modified.shape == (3, 3)
            assert modified.shape_arc_seconds == (3.0, 6.0)

        def test__trim_around_region__compare_to_imaging_util(self):

            array = np.ones((6, 6))
            array[4, 4] = 2.0

            array = scaled_array.ScaledRectangularPixelArray(array, pixel_scales=(1.0, 2.0))

            modified = imaging_util.trim_array_2d_around_region(array_2d=array, x0=3, x1=6, y0=3, y1=6)

            assert type(modified) == scaled_array.ScaledRectangularPixelArray
            assert (modified == np.array([[1.0, 1.0, 1.0],
                                          [1.0, 2.0, 1.0],
                                          [1.0, 1.0, 1.0]])).all()
            assert modified.shape == (3, 3)
            assert modified.shape_arc_seconds == (3.0, 6.0)

    class TestCentralPixel:

        def test__3x3_grid__central_pixel_is_1_and_1(self):
            grid = scaled_array.ScaledRectangularPixelArray(np.zeros((3, 3)), pixel_scales=(0.1, 0.2))
            assert grid.central_pixel_coordinates == (1, 1)

        def test__4x4_grid__central_pixel_is_1dot5_and_1dot5(self):
            grid = scaled_array.ScaledRectangularPixelArray(np.zeros((4, 4)), pixel_scales=(0.1, 0.2))
            assert grid.central_pixel_coordinates == (1.5, 1.5)

        def test__5x3_grid__central_pixel_is_2_and_1(self):
            grid = scaled_array.ScaledRectangularPixelArray(np.zeros((5, 3)), pixel_scales=(0.1, 0.2))
            assert grid.central_pixel_coordinates == (2, 1)

        def test__central_pixel_coordinates_5x5(self, array_grid_rectangular):
            assert array_grid_rectangular.central_pixel_coordinates == (2, 2)

    class TestConversion:

        def test__pixels_to_arcseconds(self, array_grid_rectangular):
            assert array_grid_rectangular.x_pixels_to_arc_seconds(1) == 0.5
            assert array_grid_rectangular.y_pixels_to_arc_seconds(1) == 1.0

        def test__arcseconds_to_pixels(self, array_grid_rectangular):
            assert array_grid_rectangular.x_arc_seconds_to_pixels(1) == 2
            assert array_grid_rectangular.y_arc_seconds_to_pixels(1) == 1

        def test__1d_pixel_grid_to_1d_arc_second_grid(self):

            array_grid_rectangular = scaled_array.ScaledRectangularPixelArray(np.zeros((2, 2)), pixel_scales=(2.0, 4.0))

            grid_pixels = np.array([[0, 0], [0, 1],
                                    [1, 0], [1, 1]])

            grid_arc_seconds = array_grid_rectangular.grid_pixels_to_grid_arcseconds(grid_pixels=grid_pixels)

            assert (grid_arc_seconds == np.array([[-1.0, -2.0], [-1.0, 2.0],
                                                  [1.0, -2.0], [1.0, 2.0]])).all()

            array_grid_rectangular = scaled_array.ScaledRectangularPixelArray(np.zeros((3, 3)), pixel_scales=(2.0, 4.0))

            grid_pixels = np.array([[0, 0], [0, 1], [0, 2],
                                    [1, 0], [1, 1], [1, 2],
                                    [2, 0], [2, 1], [2, 2]])

            grid_arc_seconds = array_grid_rectangular.grid_pixels_to_grid_arcseconds(grid_pixels=grid_pixels)

            assert (grid_arc_seconds == np.array([[-2.0, -4.0], [-2.0, 0.0], [-2.0, 4.0],
                                                   [0.0, -4.0],  [0.0, 0.0],  [0.0, 4.0],
                                                   [2.0, -4.0],  [2.0, 0.0],  [2.0, 4.0]])).all()

        def test__2d_pixel_grid_to_2d_arc_second_grid(self):

            array_grid_rectangular = scaled_array.ScaledRectangularPixelArray(np.zeros((2, 2)), pixel_scales=(2.0, 4.0))

            grid_pixels = np.array([[[0, 0], [0, 1]],
                                    [[1, 0], [1, 1]]])

            grid_arc_seconds = array_grid_rectangular.grid_pixels_to_grid_arcseconds(grid_pixels=grid_pixels)

            assert (grid_arc_seconds == np.array([[[-1.0, -2.0], [-1.0, 2.0]],
                                                  [[1.0, -2.0], [1.0, 2.0]]])).all()

            array_grid_rectangular = scaled_array.ScaledRectangularPixelArray(np.zeros((3, 3)), pixel_scales=(2.0, 4.0))

            grid_pixels = np.array([[[0, 0], [0, 1], [0, 2]],
                                    [[1, 0], [1, 1], [1, 2]],
                                    [[2, 0], [2, 1], [2, 2]]])

            grid_arc_seconds = array_grid_rectangular.grid_pixels_to_grid_arcseconds(grid_pixels=grid_pixels)

            assert (grid_arc_seconds == np.array([[[-2.0, -4.0], [-2.0, 0.0], [-2.0, 4.0]],
                                                   [[0.0, -4.0],  [0.0, 0.0],  [0.0, 4.0]],
                                                   [[2.0, -4.0],  [2.0, 0.0],  [2.0, 4.0]]])).all()

        def test__1d_arc_second_grid_to_1d_pixel_grid(self):
            array_grid_rectangular = scaled_array.ScaledRectangularPixelArray(np.zeros((2, 2)), pixel_scales=(2.0, 4.0))

            grid_arc_seconds = np.array([[-1.0, -2.0], [-1.0, 2.0],
                                         [1.0, -2.0], [1.0, 2.0]])

            grid_pixels = array_grid_rectangular.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=grid_arc_seconds)

            assert (grid_pixels == np.array([[0, 0], [0, 1],
                                             [1, 0], [1, 1]])).all()

            array_grid_rectangular = scaled_array.ScaledRectangularPixelArray(np.zeros((3, 3)), pixel_scales=(2.0, 4.0))

            grid_arc_seconds = np.array([[-2.0, -4.0], [-2.0, 0.0], [-2.0, 4.0],
                                         [0.0, -4.0], [0.0, 0.0], [0.0, 4.0],
                                         [2.0, -4.0], [2.0, 0.0], [2.0, 4.0]])

            grid_pixels = array_grid_rectangular.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=grid_arc_seconds)

            assert (grid_pixels == np.array([[0, 0], [0, 1], [0, 2],
                                             [1, 0], [1, 1], [1, 2],
                                             [2, 0], [2, 1], [2, 2]])).all()

        def test__2d_arc_second_grid_to_2d_pixel_grid(self):
            array_grid_rectangular = scaled_array.ScaledRectangularPixelArray(np.zeros((2, 2)), pixel_scales=(2.0, 4.0))

            grid_arc_seconds = np.array([[[-1.0, -2.0], [-1.0, 2.0]],
                                          [[1.0, -2.0],  [1.0, 2.0]]])

            grid_pixels = array_grid_rectangular.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=grid_arc_seconds)

            assert (grid_pixels == np.array([[[0, 0], [0, 1]],
                                             [[1, 0], [1, 1]]])).all()

            array_grid_rectangular = scaled_array.ScaledRectangularPixelArray(np.zeros((3, 3)), pixel_scales=(2.0, 4.0))

            grid_arc_seconds = np.array([[[-2.0, -4.0], [-2.0, 0.0], [-2.0, 4.0]],
                                          [[0.0, -4.0],  [0.0, 0.0],  [0.0, 4.0]],
                                          [[2.0, -4.0],  [2.0, 0.0],  [2.0, 4.0]]])

            grid_pixels = array_grid_rectangular.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=grid_arc_seconds)

            assert (grid_pixels == np.array([[[0, 0], [0, 1], [0, 2]],
                                             [[1, 0], [1, 1], [1, 2]],
                                             [[2, 0], [2, 1], [2, 2]]])).all()

    class TestGrids:

        def test__grid_2d__compare_to_array_util(self):
            grid_2d_util = imaging_util.image_grid_2d_from_shape_and_pixel_scales(shape=(4, 7), pixel_scales=(0.56, 0.8))

            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((4, 7)), pixel_scales=(0.56, 0.8))

            assert sca.grid_2d == pytest.approx(grid_2d_util, 1e-4)

        def test__array_3x3__sets_up_arcsecond_grid(self):
            sca = scaled_array.ScaledRectangularPixelArray(array=np.zeros((3, 3)), pixel_scales=(1.0, 2.0))

            assert (sca.grid_2d == np.array([[[-1., -2.], [-1., 0.], [-1., 2.]],
                                             [[0., -2.], [0., 0.], [0., 2.]],
                                             [[1., -2.], [1., 0.], [1., 2.]]])).all()

    class TestTicks:

        def test__compute_xticks_property(self):

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 3)), pixel_scales=(5.0, 1.0))
            assert sca.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 3)), pixel_scales=(5.0, 0.5))
            assert sca.xticks == pytest.approx(np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3)

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 6)), pixel_scales=(5.0, 1.0))
            assert sca.xticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((6, 3)), pixel_scales=(5.0, 1.0))
            assert sca.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        def test__compute_yticks_property(self):

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 3)), pixel_scales=(1.0, 5.0))
            assert sca.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 3)), pixel_scales=(0.5, 5.0))
            assert sca.yticks == pytest.approx(np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3)

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((6, 3)), pixel_scales=(1.0, 5.0))
            assert sca.yticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

            sca = scaled_array.ScaledRectangularPixelArray(array=np.ones((3, 6)), pixel_scales=(1.0, 5.0))
            assert sca.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

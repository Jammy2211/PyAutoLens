from autolens.imaging import scaled_array
import numpy as np
import pytest
import os

test_data_dir = "{}/../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))

@pytest.fixture(name="array_grid")
def make_array_grid():
    return scaled_array.ScaledArray(np.zeros((5, 5)), pixel_scale=0.5)


class TestDataGrid(object):
    class TestConstructors(object):

        def test__constructor(self, array_grid):
            # Does the array grid class correctly instantiate as an instance of ndarray?
            assert array_grid.shape == (5, 5)
            assert array_grid.pixel_scale == 0.5
            assert isinstance(array_grid, np.ndarray)
            assert isinstance(array_grid, scaled_array.ScaledArray)

        def test__init__input_data_grid_single_value__all_attributes_correct_including_data_inheritance(
                self):
            data_grid = scaled_array.ScaledArray.single_value(value=5.0, shape=(3, 3),
                                                              pixel_scale=1.0)

            assert (data_grid == 5.0 * np.ones((3, 3))).all()
            assert data_grid.pixel_scale == 1.0
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((3.0, 3.0))

        def test__init__input_data_grid_3x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledArray(array=np.ones((3, 3)), pixel_scale=1.0)

            assert data_grid.pixel_scale == 1.0
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((3.0, 3.0))
            assert (data_grid == np.ones((3, 3))).all()

        def test__init__input_data_grid_4x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledArray(array=np.ones((4, 3)), pixel_scale=0.1)

            assert (data_grid == np.ones((4, 3))).all()
            assert data_grid.pixel_scale == 0.1
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((0.4, 0.3))

        def test__from_fits__input_data_grid_3x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledArray.from_fits(file_path=test_data_dir + '3x3_ones', hdu=0,
                                                           pixel_scale=1.0)

            assert (data_grid == np.ones((3, 3))).all()
            assert data_grid.pixel_scale == 1.0
            assert data_grid.shape == (3, 3)
            assert data_grid.central_pixel_coordinates == (1.0, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((3.0, 3.0))

        def test__from_fits__input_data_grid_4x3__all_attributes_correct_including_data_inheritance(self):
            data_grid = scaled_array.ScaledArray.from_fits(file_path=test_data_dir + '4x3_ones', hdu=0,
                                                           pixel_scale=0.1)

            assert (data_grid == np.ones((4, 3))).all()
            assert data_grid.pixel_scale == 0.1
            assert data_grid.shape == (4, 3)
            assert data_grid.central_pixel_coordinates == (1.5, 1.0)
            assert data_grid.shape_arc_seconds == pytest.approx((0.4, 0.3))

    class TestCentralPixel:

        def test__3x3_grid__central_pixel_is_1_and_1(self):
            grid = scaled_array.ScaledArray(np.zeros((3, 3)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (1, 1)

        def test__4x4_grid__central_pixel_is_1dot5_and_1dot5(self):
            grid = scaled_array.ScaledArray(np.zeros((4, 4)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (1.5, 1.5)

        def test__5x3_grid__central_pixel_is_2_and_1(self):
            grid = scaled_array.ScaledArray(np.zeros((5, 3)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (2, 1)

        def test__central_pixel_coordinates_5x5(self, array_grid):
            assert array_grid.central_pixel_coordinates == (2, 2)

    class TestConversion:

        def test__pixels_to_arcseconds(self, array_grid):
            assert array_grid.pixels_to_arc_seconds(1) == 0.5

        def test__arcseconds_to_pixels(self, array_grid):
            assert array_grid.arc_seconds_to_pixels(1) == 2

        def test__pixel_coordinates_to_arc_second_coordinates(self, array_grid):
            # Does the central pixel have (0, 0) coordinates in arcseconds?
            assert array_grid.pixel_coordinates_to_arc_second_coordinates((2, 2)) == (0, 0)

        def test__arc_second_coordinates_to_pixel_coordinates(self, array_grid):
            # Does the (0, 0) coordinates correspond to the central pixel?
            assert array_grid.arc_second_coordinates_to_pixel_coordinates((0, 0)) == (2, 2)

    class TestPad:

        def test__from_3x3_to_5x5(self):
            array = np.ones((3, 3))
            array[1, 1] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)
            modified = array.pad(new_dimensions=(5, 5))

            assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 1.0, 1.0, 1.0, 0.0],
                                          [0.0, 1.0, 2.0, 1.0, 0.0],
                                          [0.0, 1.0, 1.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert modified.shape == (5, 5)
            assert modified.shape_arc_seconds == (5.0, 5.0)

        def test__from_5x5_to_9x9(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)
            modified = array.pad(new_dimensions=(9, 9))

            assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert modified.shape == (9, 9)
            assert modified.shape_arc_seconds == (9.0, 9.0)

        def test__from_3x3_to_4x4__goes_to_5x5_to_keep_symmetry(self):
            array = np.ones((3, 3))
            array[1, 1] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)
            modified = array.pad(new_dimensions=(4, 4))

            assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 1.0, 1.0, 1.0, 0.0],
                                          [0.0, 1.0, 2.0, 1.0, 0.0],
                                          [0.0, 1.0, 1.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert modified.shape == (5, 5)
            assert modified.shape_arc_seconds == (5.0, 5.0)

        def test__from_5x5_to_8x8__goes_to_9x9_to_keep_symmetry(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)
            modified = array.pad(new_dimensions=(8, 8))

            assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert modified.shape == (9, 9)
            assert modified.shape_arc_seconds == (9.0, 9.0)

        def test__from_4x4_to_6x6(self):
            array = np.ones((4, 4))
            array[1:3, 1:3] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)
            modified = array.pad(new_dimensions=(6, 6))

            assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                          [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                          [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                          [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert modified.shape == (6, 6)
            assert modified.shape_arc_seconds == (6.0, 6.0)

        def test__from_4x4_to_8x8(self):
            array = np.ones((4, 4))
            array[1:3, 1:3] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)
            modified = array.pad(new_dimensions=(8, 8))

            assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert modified.shape == (8, 8)
            assert modified.shape_arc_seconds == (8.0, 8.0)

        def test__from_4x4_to_5x5__goes_to_6x6_to_keep_symmetry(self):
            array = np.ones((4, 4))
            array[1:3, 1:3] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)
            modified = array.pad(new_dimensions=(5, 5))

            assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                          [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                          [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                          [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert modified.shape == (6, 6)
            assert modified.shape_arc_seconds == (6.0, 6.0)

        def test__from_4x4_to_7x7__goes_to_8x8_to_keep_symmetry(self):
            array = np.ones((4, 4))
            array[1:3, 1:3] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)
            modified = array.pad(new_dimensions=(7, 7))

            assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert modified.shape == (8, 8)
            assert modified.shape_arc_seconds == (8.0, 8.0)

        def test__from_5x4_to_7x6(self):
            array = np.ones((5, 4))
            array[2, 1:3] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)
            modified = array.pad(new_dimensions=(7, 6))

            assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                          [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                          [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                          [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                          [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert modified.shape == (7, 6)
            assert modified.shape_arc_seconds == (7.0, 6.0)

        def test__from_2x3_to_6x7(self):
            array = np.ones((2, 3))
            array[0:2, 1] = 2.0
            array[1, 2] = 9

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)
            modified = array.pad(new_dimensions=(6, 7))

            assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 2.0, 9.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert modified.shape == (6, 7)
            assert modified.shape_arc_seconds == (6.0, 7.0)

        def test__from_2x3_to_5x6__goes_to_6x7_to_keep_symmetry(self):
            array = np.ones((2, 3))
            array[0:2, 1] = 2.0
            array[1, 2] = 9

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)
            modified = array.pad(new_dimensions=(5, 6))

            assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 2.0, 9.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert modified.shape == (6, 7)
            assert modified.shape_arc_seconds == (6.0, 7.0)

        def test__x_size_smaller_than_array__raises_error(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            with pytest.raises(ValueError):
                array.trim(new_dimensions=(3, 8))

        def test__y_size_smaller_than_array__raises_error(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            with pytest.raises(ValueError):
                array.trim(new_dimensions=(8, 3))

        def test__pad_with_1s_instead(self):

            array = np.ones((3, 3))
            array[1, 1] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)
            modified = array.pad(new_dimensions=(5, 5), pad_value=1)

            assert (modified == np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 2.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0]])).all()

            assert modified.shape == (5, 5)
            assert modified.shape_arc_seconds == (5.0, 5.0)

    class TestTrim:

        def test__from_5x5_to_3x3(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(3, 3))

            assert (modified == np.array([[1.0, 1.0, 1.0],
                                          [1.0, 2.0, 1.0],
                                          [1.0, 1.0, 1.0]])).all()

            assert modified.shape == (3, 3)
            assert modified.shape_arc_seconds == (3.0, 3.0)

        def test__from_7x7_to_3x3(self):
            array = np.ones((7, 7))
            array[3, 3] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(3, 3))

            assert (modified == np.array([[1.0, 1.0, 1.0],
                                          [1.0, 2.0, 1.0],
                                          [1.0, 1.0, 1.0]])).all()

            assert modified.shape == (3, 3)
            assert modified.shape_arc_seconds == (3.0, 3.0)

        def test__from_11x11_to_5x5(self):
            array = np.ones((11, 11))
            array[5, 5] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(5, 5))

            assert (modified == np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 2.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0]])).all()

            assert modified.shape == (5, 5)
            assert modified.shape_arc_seconds == (5.0, 5.0)

        def test__from_5x5_to_2x2__goes_to_3x3_to_keep_symmetry(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(2, 2))

            assert (modified == np.array([[1.0, 1.0, 1.0],
                                          [1.0, 2.0, 1.0],
                                          [1.0, 1.0, 1.0]])).all()

            assert modified.shape == (3, 3)
            assert modified.shape_arc_seconds == (3.0, 3.0)

        def test__from_5x5_to_4x4__goes_to_5x5_to_keep_symmetry(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(4, 4))

            assert (modified == np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 2.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0]])).all()

            assert modified.shape == (5, 5)
            assert modified.shape_arc_seconds == (5.0, 5.0)

        def test__from_11x11_to_4x4__goes_to_5x5_to_keep_symmetry(self):
            array = np.ones((11, 11))
            array[5, 5] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(4, 4))

            assert (modified == np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 2.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0]])).all()

            assert modified.shape == (5, 5)
            assert modified.shape_arc_seconds == (5.0, 5.0)

        def test__from_4x4_to_2x2(self):
            array = np.ones((4, 4))
            array[1:3, 1:3] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(2, 2))

            assert (modified == np.array([[2.0, 2.0],
                                          [2.0, 2.0]])).all()

            assert modified.shape == (2, 2)
            assert modified.shape_arc_seconds == (2.0, 2.0)

        def test__from_6x6_to_4x4(self):
            array = np.ones((6, 6))
            array[2:4, 2:4] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(4, 4))

            assert (modified == np.array([[1.0, 1.0, 1.0, 1.0],
                                          [1.0, 2.0, 2.0, 1.0],
                                          [1.0, 2.0, 2.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0]])).all()

            assert modified.shape == (4, 4)
            assert modified.shape_arc_seconds == (4.0, 4.0)

        def test__from_12x12_to_6x6(self):
            array = np.ones((12, 12))
            array[5:7, 5:7] = 2.0
            array[4, 4] = 9.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(6, 6))

            assert (modified == np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 9.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                          [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])).all()

            assert modified.shape == (6, 6)
            assert modified.shape_arc_seconds == (6.0, 6.0)

        def test__from_4x4_to_3x3__goes_to_4x4_to_keep_symmetry(self):
            array = np.ones((4, 4))
            array[1:3, 1:3] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(3, 3))

            assert (modified == np.array([[1.0, 1.0, 1.0, 1.0],
                                          [1.0, 2.0, 2.0, 1.0],
                                          [1.0, 2.0, 2.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0]])).all()

            assert modified.shape == (4, 4)
            assert modified.shape_arc_seconds == (4.0, 4.0)

        def test__from_6x6_to_3x3_goes_to_4x4_to_keep_symmetry(self):
            array = np.ones((6, 6))
            array[2:4, 2:4] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(3, 3))

            assert (modified == np.array([[1.0, 1.0, 1.0, 1.0],
                                          [1.0, 2.0, 2.0, 1.0],
                                          [1.0, 2.0, 2.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0]])).all()

            assert modified.shape == (4, 4)
            assert modified.shape_arc_seconds == (4.0, 4.0)

        def test__from_12x12_to_5x5__goes_to_6x6_to_keep_symmetry(self):
            array = np.ones((12, 12))
            array[5:7, 5:7] = 2.0
            array[4, 4] = 9.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(5, 5))

            assert (modified == np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 9.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                          [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])).all()

            assert modified.shape == (6, 6)
            assert modified.shape_arc_seconds == (6.0, 6.0)

        def test__from_5x4_to_3x2(self):
            array = np.ones((5, 4))
            array[2, 1:3] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(3, 2))

            assert (modified == np.array([[1.0, 1.0],
                                          [2.0, 2.0],
                                          [1.0, 1.0]])).all()

            assert modified.shape == (3, 2)
            assert modified.shape_arc_seconds == (3.0, 2.0)

        def test__from_4x5_to_2x3(self):
            array = np.ones((4, 5))
            array[1:3, 2] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(2, 3))

            assert (modified == np.array([[1.0, 2.0, 1.0],
                                          [1.0, 2.0, 1.0]])).all()

            assert modified.shape == (2, 3)
            assert modified.shape_arc_seconds == (2.0, 3.0)

        def test__from_5x4_to_4x3__goes_to_5x4_to_keep_symmetry(self):
            array = np.ones((5, 4))
            array[2, 1:3] = 2.0
            array[4, 3] = 9.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            modified = array.trim(new_dimensions=(4, 3))

            assert (modified == np.array([[1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0],
                                          [1.0, 2.0, 2.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 9.0]])).all()

            assert modified.shape == (5, 4)
            assert modified.shape_arc_seconds == (5.0, 4.0)

        def test__x_size_bigger_than_array__raises_error(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            with pytest.raises(ValueError):
                array.trim(new_dimensions=(8, 3))

        def test__y_size_bigger_than_array__raises_error(self):
            array = np.ones((5, 5))
            array[2, 2] = 2.0

            array = scaled_array.ScaledArray(array, pixel_scale=1.0)

            with pytest.raises(ValueError):
                array.trim(new_dimensions=(3, 8))

    class TestGridCoordinates:

        def test__array_1x1__sets_up_arcsecond_coordinates(self):
            grid = scaled_array.ScaledArray(array=np.zeros((1, 1)), pixel_scale=1.0)

            grid_coordinates = grid.grid_coordinates

            assert (grid_coordinates == np.array([[[0.0, 0.0]]])).all()

            assert (grid_coordinates[0, 0] == np.array([[0.0, 0.0]])).all()

        def test__array_2x2__sets_up_arcsecond_coordinates(self):
            grid = scaled_array.ScaledArray(array=np.zeros((2, 2)), pixel_scale=1.0)

            grid_coordinates = grid.grid_coordinates

            assert (grid_coordinates == np.array([[[-0.5, -0.5], [-0.5, 0.5]],
                                                  [[0.5, -0.5], [0.5, 0.5]]])).all()

        def test__array_3x3__sets_up_arcsecond_coordinates(self):
            grid = scaled_array.ScaledArray(array=np.zeros((3, 3)), pixel_scale=1.0)

            grid_coordinates = grid.grid_coordinates
            print(list(grid_coordinates))

            assert (grid_coordinates == np.array([[[-1., -1.], [-1., 0.], [-1., 1.]],
                                                  [[0., -1.], [0., 0.], [0., 1.]],
                                                  [[1., -1.], [1., 0.], [1., 1.]]])).all()

        def test__array_4x4__sets_up_arcsecond_coordinates(self):
            grid = scaled_array.ScaledArray(array=np.zeros((4, 4)), pixel_scale=0.5)

            grid_coordinates = grid.grid_coordinates
            print(list(grid_coordinates))

            assert (grid_coordinates == np.array([[[-0.75, -0.75], [-0.75, -0.25], [-0.75, 0.25], [-0.75, 0.75]],
                                                  [[-0.25, -0.75], [-0.25, -0.25], [-0.25, 0.25], [-0.25, 0.75]],
                                                  [[0.25, -0.75], [0.25, -0.25], [0.25, 0.25], [0.25, 0.75]],
                                                  [[0.75, -0.75], [0.75, -0.25], [0.75, 0.25], [0.75, 0.75]]])).all()

        def test__array_2x3__sets_up_arcsecond_coordinates(self):
            grid = scaled_array.ScaledArray(array=np.zeros((2, 3)), pixel_scale=1.0)

            grid_coordinates = grid.grid_coordinates
            print(list(grid_coordinates))

            assert (grid_coordinates == np.array([[[-0.5, -1.], [-0.5, 0.], [-0.5, 1.]],
                                                  [[0.5, -1.], [0.5, 0.], [0.5, 1.]]])).all()

        def test__array_3x2__sets_up_arcsecond_coordinates(self):
            grid = scaled_array.ScaledArray(array=np.zeros((3, 2)), pixel_scale=1.0)

            grid_coordinates = grid.grid_coordinates

            assert (grid_coordinates == np.array([[[-1., -0.5], [-1., 0.5]],
                                                  [[0., -0.5], [0., 0.5]],
                                                  [[1., -0.5], [1., 0.5]]])).all()

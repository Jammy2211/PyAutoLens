from autolens.imaging import imaging_util
from autolens.imaging import scaled_array
import numpy as np
import pytest
import os

test_data_dir = "{}/../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))

@pytest.fixture(name="array_grid")
def make_array_grid():
    return scaled_array.ScaledArray(np.zeros((5, 5)), pixel_scale=0.5)

    
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
        data_grid = scaled_array.ScaledArray.from_fits_with_scale(file_path=test_data_dir + '3x3_ones', hdu=0,
                                                                  pixel_scale=1.0)

        assert (data_grid == np.ones((3, 3))).all()
        assert data_grid.pixel_scale == 1.0
        assert data_grid.shape == (3, 3)
        assert data_grid.central_pixel_coordinates == (1.0, 1.0)
        assert data_grid.shape_arc_seconds == pytest.approx((3.0, 3.0))

    def test__from_fits__input_data_grid_4x3__all_attributes_correct_including_data_inheritance(self):
        data_grid = scaled_array.ScaledArray.from_fits_with_scale(file_path=test_data_dir + '4x3_ones', hdu=0,
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


class TestGrids:

    def test__grid_2d__compare_to_array_util(self):

        grid_2d_util = imaging_util.image_grid_2d_from_shape_and_pixel_scale(shape=(4, 7), pixel_scale=0.56)

        sca = scaled_array.ScaledArray(array=np.zeros((4, 7)), pixel_scale=0.56)

        assert sca.grid_2d == pytest.approx(grid_2d_util, 1e-4)

    def test__array_3x3__sets_up_arcsecond_grid(self):

        sca = scaled_array.ScaledArray(array=np.zeros((3, 3)), pixel_scale=1.0)

        assert (sca.grid_2d == np.array([[[-1., -1.], [-1., 0.], [-1., 1.]],
                                          [[0., -1.],  [0., 0.],  [0., 1.]],
                                          [[1., -1.],  [1., 0.],  [1., 1.]]])).all()


class TestPaddedGridForPSFEdges:

    def test__3x3_array__psf_size_is_1x1__no_padding(self):

        sca = scaled_array.ScaledArray(array=np.zeros((3, 3)), pixel_scale=1.0)

        image_grid = sca.padded_image_grid_for_psf_edges(psf_shape=(1, 1))

        assert len(image_grid) == 9

    def test__3x3_image__5x5_psf_size__7x7_image_grid_made(self):

        sca = scaled_array.ScaledArray(array=np.zeros((3,3)), pixel_scale=1.0)
        image_grid = sca.padded_image_grid_for_psf_edges(psf_shape=(5, 5))

        assert len(image_grid) == 49

    def test__3x3_image__7x7_psf_size__9x9_image_grid_made(self):

        sca = scaled_array.ScaledArray(array=np.zeros((3,3)), pixel_scale=1.0)
        image_grid = sca.padded_image_grid_for_psf_edges(psf_shape=(7, 7))

        assert len(image_grid) == 81

    def test__4x3_image__3x3_psf_size__6x5_image_grid_made(self):

        sca = scaled_array.ScaledArray(array=np.zeros((4,3)), pixel_scale=1.0)
        image_grid = sca.padded_image_grid_for_psf_edges(psf_shape=(3, 3))

        assert len(image_grid) == 30

    def test__4x3_image__5x5_psf_size__8x7_image_grid_made(self):

        sca = scaled_array.ScaledArray(array=np.zeros((4,3)), pixel_scale=1.0)
        image_grid = sca.padded_image_grid_for_psf_edges(psf_shape=(5, 5))

        assert len(image_grid) == 56

    def test__3x4_image__3x3_psf_size__5x6_image_grid_made(self):

        sca = scaled_array.ScaledArray(array=np.zeros((3,4)), pixel_scale=1.0)
        image_grid = sca.padded_image_grid_for_psf_edges(psf_shape=(3, 3))

        assert len(image_grid) == 30

    def test__3x4_image__5x5_psf_size__7x8_image_grid_made(self):

        sca = scaled_array.ScaledArray(array=np.zeros((3,4)), pixel_scale=1.0)
        image_grid = sca.padded_image_grid_for_psf_edges(psf_shape=(5, 5))

        assert len(image_grid) == 56

    def test__4x4_image__3x3_psf_size__6x6_image_grid_made(self):

        sca = scaled_array.ScaledArray(array=np.zeros((4,4)), pixel_scale=1.0)
        image_grid = sca.padded_image_grid_for_psf_edges(psf_shape=(3, 3))

        assert len(image_grid) == 36

    def test__4x4_image__5x5_psf_size__8x8_image_grid_made_border(self):

        sca = scaled_array.ScaledArray(array=np.zeros((4,4)), pixel_scale=1.0)
        image_grid = sca.padded_image_grid_for_psf_edges(psf_shape=(5, 5))

        assert len(image_grid) == 64

    def test__image_grid_coordinates__match_grid_2d_after_padding(self):

        sca_for_padding = scaled_array.ScaledArray(array=np.zeros((4,4)), pixel_scale=3.0)
        padded_image_grid = sca_for_padding.padded_image_grid_for_psf_edges(psf_shape=(3, 3))
        image_grid = imaging_util.image_grid_masked_from_mask_and_pixel_scale(mask=np.full((6, 6), False), pixel_scale=3.0)
        assert (padded_image_grid == image_grid).all()


        sca_for_padding = scaled_array.ScaledArray(array=np.zeros((4,5)), pixel_scale=2.0)
        padded_image_grid = sca_for_padding.padded_image_grid_for_psf_edges(psf_shape=(3, 3))
        image_grid = imaging_util.image_grid_masked_from_mask_and_pixel_scale(mask=np.full((6, 7), False), pixel_scale=2.0)
        assert (padded_image_grid == image_grid).all()


        sca_for_padding = scaled_array.ScaledArray(array=np.zeros((5,4)), pixel_scale=1.0)
        padded_image_grid = sca_for_padding.padded_image_grid_for_psf_edges(psf_shape=(3, 3))
        image_grid = imaging_util.image_grid_masked_from_mask_and_pixel_scale(mask=np.full((7, 6), False), pixel_scale=1.0)
        assert (padded_image_grid == image_grid).all()


        sca_for_padding = scaled_array.ScaledArray(array=np.zeros((2,5)), pixel_scale=8.0)
        padded_image_grid = sca_for_padding.padded_image_grid_for_psf_edges(psf_shape=(5, 5))
        image_grid = imaging_util.image_grid_masked_from_mask_and_pixel_scale(mask=np.full((6, 9), False), pixel_scale=8.0)
        assert (padded_image_grid == image_grid).all()

    def test__sub_grid_coordinates__match_grid_2d_after_padding(self):

        sca_for_padding = scaled_array.ScaledArray(array=np.zeros((4,4)), pixel_scale=3.0)
        padded_sub_grid = sca_for_padding.padded_sub_grid_for_psf_edges_from_sub_grid_size(psf_shape=(3, 3),
                                                                                             sub_grid_size=3)
        sub_grid = imaging_util.sub_grid_masked_from_mask_pixel_scale_and_sub_grid_size(mask=np.full((6, 6), False),
                                                                                        pixel_scale=3.0,
                                                                                        sub_grid_size=3)
        assert (padded_sub_grid == sub_grid).all()


        sca_for_padding = scaled_array.ScaledArray(array=np.zeros((4,5)), pixel_scale=2.0)
        padded_sub_grid = sca_for_padding.padded_sub_grid_for_psf_edges_from_sub_grid_size(psf_shape=(3, 3),
                                                                                             sub_grid_size=1)
        sub_grid = imaging_util.sub_grid_masked_from_mask_pixel_scale_and_sub_grid_size(mask=np.full((6, 7), False),
                                                                                        pixel_scale=2.0,
                                                                                        sub_grid_size=1)
        assert (padded_sub_grid == sub_grid).all()


        sca_for_padding = scaled_array.ScaledArray(array=np.zeros((5,4)), pixel_scale=1.0)
        padded_sub_grid = sca_for_padding.padded_sub_grid_for_psf_edges_from_sub_grid_size(psf_shape=(3, 3),
                                                                                             sub_grid_size=2)
        sub_grid = imaging_util.sub_grid_masked_from_mask_pixel_scale_and_sub_grid_size(mask=np.full((7, 6), False),
                                                                                        pixel_scale=1.0,
                                                                                        sub_grid_size=2)
        assert (padded_sub_grid == sub_grid).all()


        sca_for_padding = scaled_array.ScaledArray(array=np.zeros((2,5)), pixel_scale=8.0)
        padded_sub_grid = sca_for_padding.padded_sub_grid_for_psf_edges_from_sub_grid_size(psf_shape=(5, 5),
                                                                                             sub_grid_size=3)
        sub_grid = imaging_util.sub_grid_masked_from_mask_pixel_scale_and_sub_grid_size(mask=np.full((6, 9), False),
                                                                                        pixel_scale=8.0, sub_grid_size=3)
        assert (padded_sub_grid == sub_grid).all()
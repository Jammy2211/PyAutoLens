from __future__ import division, print_function
import pytest
import numpy as np
import image
import os

test_data_dir = "{}/../data/test_data/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(scope='class')
def test_image():
    im = image.Image(filename='3x3_ones.fits', hdu=0, pixel_scale=0.1, path=test_data_dir)
    return im


# noinspection PyClassHasNoInit,PyShadowingNames
class TestImage:
    def test__init__input_image_3x3__all_attributes_correct(self, test_image):
        assert (test_image.image2d == np.ones((3, 3))).all()
        assert test_image.xy_dim[0] == 3
        assert test_image.xy_dim[1] == 3
        assert test_image.xy_arcsec[0] == pytest.approx(0.3)
        assert test_image.xy_arcsec[1] == pytest.approx(0.3)

    def test__set_sky_via_edges__input_all_ones__sky_bg_level_1(self, test_image):
        test_image.image2d = np.ones((3, 3))
        test_image.xy_dim = (3, 3)
        test_image.set_sky_via_edges(no_edges=1)

        assert test_image.sky_background_level == 1.0
        assert test_image.sky_background_noise == 0.0

    def test__set_sky_via_edges__3x3_image_simple_gaussian__answer_ignores_central_pixel(self, test_image):
        test_image.image2d = np.array([[1, 1, 1],
                                       [1, 100, 1],
                                       [1, 1, 1]])
        test_image.xy_dim = (3, 3)
        test_image.set_sky_via_edges(no_edges=1)

        assert test_image.sky_background_level == 1.0
        assert test_image.sky_background_noise == 0.0

    def test__set_sky_via_edges__4x3_image_simple_gaussian__ignores_central_pixels(self, test_image):
        test_image.image2d = np.array([[1, 1, 1],
                                       [1, 100, 1],
                                       [1, 100, 1],
                                       [1, 1, 1]])
        test_image.xy_dim = (4, 3)

        test_image.set_sky_via_edges(no_edges=1)

        assert test_image.sky_background_level == 1.0
        assert test_image.sky_background_noise == 0.0

    def test__set_sky_via_edges__4x4_image_simple_gaussian__ignores_central_pixels(self, test_image):
        test_image.image2d = np.array([[1, 1, 1, 1],
                                       [1, 100, 100, 1],
                                       [1, 100, 100, 1],
                                       [1, 1, 1, 1]])
        test_image.xy_dim = (4, 4)

        test_image.set_sky_via_edges(no_edges=1)

        assert test_image.sky_background_level == 1.0
        assert test_image.sky_background_noise == 0.0

    def test__set_sky_via_edges__5x5_image_simple_gaussian_two_edges__ignores_central_pixel(self, test_image):
        test_image.image2d = np.array([[1, 1, 1, 1, 1],
                                       [1, 1, 1, 1, 1],
                                       [1, 1, 100, 1, 1],
                                       [1, 1, 1, 1, 1],
                                       [1, 1, 1, 1, 1]])
        test_image.xy_dim = [5, 5]

        test_image.set_sky_via_edges(no_edges=2)

        assert test_image.sky_background_level == 1.0
        assert test_image.sky_background_noise == 0.0

    def test__set_sky_via_edges__6x5_image_two_edges__correct_values(self, test_image):
        test_image.image2d = np.array([[0, 1, 2, 3, 4],
                                       [5, 6, 7, 8, 9],
                                       [10, 11, 100, 12, 13],
                                       [14, 15, 100, 16, 17],
                                       [18, 19, 20, 21, 22],
                                       [23, 24, 25, 26, 27]])
        test_image.xy_dim = [6, 5]

        test_image.set_sky_via_edges(no_edges=2)

        assert test_image.sky_background_level == np.mean(np.arange(28))
        assert test_image.sky_background_noise == np.std(np.arange(28))

    def test__set_sky_via_edges__7x7_image_three_edges__correct_values(self, test_image):
        test_image.image2d = np.array([[0, 1, 2, 3, 4, 5, 6],
                                       [7, 8, 9, 10, 11, 12, 13],
                                       [14, 15, 16, 17, 18, 19, 20],
                                       [21, 22, 23, 100, 24, 25, 26],
                                       [27, 28, 29, 30, 31, 32, 33],
                                       [34, 35, 36, 37, 38, 39, 40],
                                       [41, 42, 43, 44, 45, 46, 47]])
        test_image.xy_dim = [7, 7]

        test_image.set_sky_via_edges(no_edges=3)

        assert test_image.sky_background_level == np.mean(np.arange(48))
        assert test_image.sky_background_noise == np.std(np.arange(48))


# noinspection PyClassHasNoInit,PyShadowingNames
class TestMask:
    def test__set_circle__input_big_mask__correct_mask(self):
        mask = image.CircleMask(dimensions=(3, 3), pixel_scale=0.1, radius=0.5)

        assert (mask.array == np.ones((3, 3))).all()

    def test__set_circle__odd_x_odd_mask_input_radius_small__correct_mask(self):
        mask = image.CircleMask(dimensions=(3, 3), pixel_scale=0.1, radius=0.05)

        assert (mask.array == np.array([[0, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]])).all()

    def test__set_circle__odd_x_odd_mask_input_radius_medium__correct_mask(self):
        mask = image.CircleMask(dimensions=(3, 3), pixel_scale=0.1, radius=0.1)

        assert (mask.array == np.array([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]])).all()

    def test__set_circle__odd_x_odd_mask_input_radius_large__correct_mask(self):
        mask = image.CircleMask(dimensions=(3, 3), pixel_scale=0.1, radius=0.3)

        assert (mask.array == np.array([[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]])).all()

    def test__set_circle__even_x_odd_mask_input_radius_small__correct_mask(self):
        mask = image.CircleMask(dimensions=(4, 3), pixel_scale=0.1, radius=0.05)

        assert (mask.array == np.array([[0, 0, 0],
                                        [0, 1, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]])).all()

    def test__set_circle__even_x_odd_mask_input_radius_medium__correct_mask(self):
        mask = image.CircleMask(dimensions=(4, 3), pixel_scale=0.1, radius=0.150001)

        assert (mask.array == np.array([[0, 1, 0],
                                        [1, 1, 1],
                                        [1, 1, 1],
                                        [0, 1, 0]])).all()

    def test__set_circle__even_x_odd_mask_input_radius_large__correct_mask(self):
        mask = image.CircleMask(dimensions=(4, 3), pixel_scale=0.1, radius=0.3)

        assert (mask.array == np.array([[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]])).all()

    def test__set_circle__even_x_even_mask_input_radius_small__correct_mask(self):
        mask = image.CircleMask(dimensions=(4, 4), pixel_scale=0.1, radius=0.072)

        assert (mask.array == np.array([[0, 0, 0, 0],
                                        [0, 1, 1, 0],
                                        [0, 1, 1, 0],
                                        [0, 0, 0, 0]])).all()

    def test__set_circle__even_x_even_mask_input_radius_medium__correct_mask(self):
        mask = image.CircleMask(dimensions=(4, 4), pixel_scale=0.1, radius=0.17)

        assert (mask.array == np.array([[0, 1, 1, 0],
                                        [1, 1, 1, 1],
                                        [1, 1, 1, 1],
                                        [0, 1, 1, 0]])).all()

    def test__set_circle__even_x_even_mask_input_radius_large__correct_mask(self):
        mask = image.CircleMask(dimensions=(4, 4), pixel_scale=0.1, radius=0.3)

        assert (mask.array == np.array([[1, 1, 1, 1],
                                        [1, 1, 1, 1],
                                        [1, 1, 1, 1],
                                        [1, 1, 1, 1]])).all()

    def test__set_annulus__odd_x_odd_mask_inner_radius_zero_outer_radius_small__correct_mask(self):
        mask = image.AnnulusMask(dimensions=(3, 3), pixel_scale=0.1, inner_radius=0.0, outer_radius=0.05)

        assert (mask.array == np.array([[0, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]])).all()

    def test__set_annulus__odd_x_odd_mask_inner_radius_small_outer_radius_large__correct_mask(self):
        mask = image.AnnulusMask(dimensions=(3, 3), pixel_scale=0.1, inner_radius=0.05, outer_radius=0.3)

        print(mask.array)

        assert (mask.array == np.array([[1, 1, 1],
                                        [1, 0, 1],
                                        [1, 1, 1]])).all()

    def test__set_annulus__even_x_odd_mask_inner_radius_small_outer_radius_medium__correct_mask(self):
        mask = image.AnnulusMask(dimensions=(4, 3), pixel_scale=0.1, inner_radius=0.051, outer_radius=0.151)

        assert (mask.array == np.array([[0, 1, 0],
                                        [1, 0, 1],
                                        [1, 0, 1],
                                        [0, 1, 0]])).all()

    def test__set_annulus__even_x_odd_mask_inner_radius_medium_outer_radius_large__correct_mask(self):
        mask = image.AnnulusMask(dimensions=(4, 3), pixel_scale=0.1, inner_radius=0.151, outer_radius=0.3)

        assert (mask.array == np.array([[1, 0, 1],
                                        [0, 0, 0],
                                        [0, 0, 0],
                                        [1, 0, 1]])).all()

    def test__set_annulus__even_x_even_mask_inner_radius_small_outer_radius_medium__correct_mask(self):
        mask = image.AnnulusMask(dimensions=(4, 4), pixel_scale=0.1, inner_radius=0.081, outer_radius=0.2)

        assert (mask.array == np.array([[0, 1, 1, 0],
                                        [1, 0, 0, 1],
                                        [1, 0, 0, 1],
                                        [0, 1, 1, 0]])).all()

    def test__set_annulus__even_x_even_mask_inner_radius_medium_outer_radius_large__correct_mask(self):
        mask = image.AnnulusMask(dimensions=(4, 4), pixel_scale=0.1, inner_radius=0.171, outer_radius=0.3)

        assert (mask.array == np.array([[1, 0, 0, 1],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [1, 0, 0, 1]])).all()


# noinspection PyClassHasNoInit
class TestLoadFits:
    def test__load_fits__input_fits_3x3_ones__loads_data_as_type_numpy_array(self):
        assert type(image.Image('3x3_ones.fits', hdu=0, pixel_scale=1, path=test_data_dir).image2d) == np.ndarray

    def test__load_fits__input_fits_3x3_ones__loads_correct_data(self):
        assert (image.Image('3x3_ones.fits', hdu=0, pixel_scale=1, path=test_data_dir).image2d == np.ones((3, 3))).all()

    def test__load_fits__input_fits_4x3_ones__loads_correct_data(self):
        assert (image.Image('4x3_ones.fits', hdu=0, pixel_scale=1, path=test_data_dir).image2d == np.ones((4, 3))).all()

    def test__load_fits__input_files_3x3_ones__loads_correct_dimensions(self):
        xy_dim = image.Image('3x3_ones.fits', hdu=0, pixel_scale=1, path=test_data_dir).xy_dim

        assert xy_dim[0] == 3
        assert xy_dim[1] == 3

    def test__load_fits__input_files_4x3_ones__loads_correct_dimensions(self):
        xy_dim = image.Image('4x3_ones.fits', hdu=0, pixel_scale=1, path=test_data_dir).xy_dim

        assert xy_dim[0] == 4
        assert xy_dim[1] == 3

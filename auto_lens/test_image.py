from __future__ import division, print_function
import pytest
import numpy as np
import image
import os

test_data_dir = "{}/../data/test_data/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(scope='class', name='test_image')
def make_image():
    return image.Image.from_fits(filename='3x3_ones.fits', hdu=0, pixel_scale=0.1, path=test_data_dir)


@pytest.fixture(scope='class', name='test_psf')
def make_psf():
    return image.PSF.from_fits(filename='3x3_ones.fits', hdu=0, pixel_scale=0.1, path=test_data_dir)


# noinspection PyClassHasNoInit,PyShadowingNames
class TestData:
    # TODO : Implement routine to cut-out the data arrays (and update their dimensions / size)

    class TestCutOut(object):
        def test__trim_7x7_to_3x3_centred_on_image_middle(self):
            data = np.ones((7, 7))
            data[3, 3] = 2

            psf = image.PSF(data, pixel_scale=0.1)


class TestImage(object):
    class TestInput(object):
        def test__init__input_image_3x3__all_attributes_correct(self, test_image):
            assert (test_image.data == np.ones((3, 3))).all()
            assert test_image.dimensions[0] == 3
            assert test_image.dimensions[1] == 3
            assert test_image.dimensions_arc_seconds[0] == pytest.approx(0.3)
            assert test_image.dimensions_arc_seconds[1] == pytest.approx(0.3)

    class TestSetSky(object):
        def test__via_edges__input_all_ones__sky_bg_level_1(self, test_image):
            test_image.data = np.ones((3, 3))
            test_image.dimensions = (3, 3)
            test_image.set_sky_via_edges(no_edges=1)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__3x3_image_simple_gaussian__answer_ignores_central_pixel(self, test_image):
            test_image.data = np.array([[1, 1, 1],
                                        [1, 100, 1],
                                        [1, 1, 1]])
            test_image.dimensions = (3, 3)
            test_image.set_sky_via_edges(no_edges=1)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__4x3_image_simple_gaussian__ignores_central_pixels(self, test_image):
            test_image.data = np.array([[1, 1, 1],
                                        [1, 100, 1],
                                        [1, 100, 1],
                                        [1, 1, 1]])
            test_image.dimensions = (4, 3)

            test_image.set_sky_via_edges(no_edges=1)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__4x4_image_simple_gaussian__ignores_central_pixels(self, test_image):
            test_image.data = np.array([[1, 1, 1, 1],
                                        [1, 100, 100, 1],
                                        [1, 100, 100, 1],
                                        [1, 1, 1, 1]])
            test_image.dimensions = (4, 4)

            test_image.set_sky_via_edges(no_edges=1)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__5x5_image_simple_gaussian_two_edges__ignores_central_pixel(self, test_image):
            test_image.data = np.array([[1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 100, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1]])
            test_image.dimensions = [5, 5]

            test_image.set_sky_via_edges(no_edges=2)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__6x5_image_two_edges__correct_values(self, test_image):
            test_image.data = np.array([[0, 1, 2, 3, 4],
                                        [5, 6, 7, 8, 9],
                                        [10, 11, 100, 12, 13],
                                        [14, 15, 100, 16, 17],
                                        [18, 19, 20, 21, 22],
                                        [23, 24, 25, 26, 27]])
            test_image.dimensions = [6, 5]

            test_image.set_sky_via_edges(no_edges=2)

            assert test_image.sky_background_level == np.mean(np.arange(28))
            assert test_image.sky_background_noise == np.std(np.arange(28))

        def test__via_edges__7x7_image_three_edges__correct_values(self, test_image):
            test_image.data = np.array([[0, 1, 2, 3, 4, 5, 6],
                                        [7, 8, 9, 10, 11, 12, 13],
                                        [14, 15, 16, 17, 18, 19, 20],
                                        [21, 22, 23, 100, 24, 25, 26],
                                        [27, 28, 29, 30, 31, 32, 33],
                                        [34, 35, 36, 37, 38, 39, 40],
                                        [41, 42, 43, 44, 45, 46, 47]])
            test_image.dimensions = [7, 7]

            test_image.set_sky_via_edges(no_edges=3)

            assert test_image.sky_background_level == np.mean(np.arange(48))
            assert test_image.sky_background_noise == np.std(np.arange(48))


class TestPSF(object):
    class TestSetup(object):
        def test__init__input_image_3x3__all_attributes_correct(self, test_psf):
            assert (test_psf.data == np.ones((3, 3))).all()
            assert test_psf.dimensions[0] == 3
            assert test_psf.dimensions[1] == 3
            assert test_psf.dimensions_arc_seconds[0] == pytest.approx(0.3)
            assert test_psf.dimensions_arc_seconds[1] == pytest.approx(0.3)

        def test__input_image_3x3__setup_from_image(self, test_image):
            test_psf = test_image.load_psf(filename='3x3_ones.fits', hdu=0, path=test_data_dir)

            assert (test_psf.data == np.ones((3, 3))).all()
            assert test_psf.dimensions[0] == 3
            assert test_psf.dimensions[1] == 3
            assert test_psf.dimensions_arc_seconds[0] == pytest.approx(0.3)
            assert test_psf.dimensions_arc_seconds[1] == pytest.approx(0.3)


# noinspection PyClassHasNoInit,PyShadowingNames
class TestMask:
    class TestCircular(object):
        def test__input_big_mask__correct_mask(self):
            mask = image.Mask.circular(dimensions=(3, 3), pixel_scale=0.1, radius=0.5)

            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [True, True, True]])).all()

        def test__odd_x_odd_mask_input_radius_small__correct_mask(self):
            mask = image.Mask.circular(dimensions=(3, 3), pixel_scale=0.1, radius=0.05)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__odd_x_odd_mask_input_radius_medium__correct_mask(self):
            mask = image.Mask.circular(dimensions=(3, 3), pixel_scale=0.1, radius=0.1)

            assert (mask == np.array([[False, True, False],
                                      [True, True, True],
                                      [False, True, False]])).all()

        def test__odd_x_odd_mask_input_radius_large__correct_mask(self):
            mask = image.Mask.circular(dimensions=(3, 3), pixel_scale=0.1, radius=0.3)

            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [True, True, True]])).all()

        def test__even_x_odd_mask_input_radius_small__correct_mask(self):
            mask = image.Mask.circular(dimensions=(4, 3), pixel_scale=0.1, radius=0.05)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__even_x_odd_mask_input_radius_medium__correct_mask(self):
            mask = image.Mask.circular(dimensions=(4, 3), pixel_scale=0.1, radius=0.150001)

            assert (mask == np.array([[False, True, False],
                                      [True, True, True],
                                      [True, True, True],
                                      [False, True, False]])).all()

        def test__even_x_odd_mask_input_radius_large__correct_mask(self):
            mask = image.Mask.circular(dimensions=(4, 3), pixel_scale=0.1, radius=0.3)

            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [True, True, True],
                                      [True, True, True]])).all()

        def test__even_x_even_mask_input_radius_small__correct_mask(self):
            mask = image.Mask.circular(dimensions=(4, 4), pixel_scale=0.1, radius=0.072)

            assert (mask == np.array([[False, False, False, False],
                                      [False, True, True, False],
                                      [False, True, True, False],
                                      [False, False, False, False]])).all()

        def test__even_x_even_mask_input_radius_medium__correct_mask(self):
            mask = image.Mask.circular(dimensions=(4, 4), pixel_scale=0.1, radius=0.17)

            assert (mask == np.array([[False, True, True, False],
                                      [True, True, True, True],
                                      [True, True, True, True],
                                      [False, True, True, False]])).all()

        def test__even_x_even_mask_input_radius_large__correct_mask(self):
            mask = image.Mask.circular(dimensions=(4, 4), pixel_scale=0.1, radius=0.3)

            assert (mask == np.array([[True, True, True, True],
                                      [True, True, True, True],
                                      [True, True, True, True],
                                      [True, True, True, True]])).all()

    class TestAnnulus(object):
        def test__odd_x_odd_mask_inner_radius_zero_outer_radius_small__correct_mask(self):
            mask = image.Mask.annular(dimensions=(3, 3), pixel_scale=0.1, inner_radius=0.0, outer_radius=0.05)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__odd_x_odd_mask_inner_radius_small_outer_radius_large__correct_mask(self):
            mask = image.Mask.annular(dimensions=(3, 3), pixel_scale=0.1, inner_radius=0.05, outer_radius=0.3)

            assert (mask == np.array([[True, True, True],
                                      [True, False, True],
                                      [True, True, True]])).all()

        def test__even_x_odd_mask_inner_radius_small_outer_radius_medium__correct_mask(self):
            mask = image.Mask.annular(dimensions=(4, 3), pixel_scale=0.1, inner_radius=0.051, outer_radius=0.151)

            assert (mask == np.array([[False, True, False],
                                      [True, False, True],
                                      [True, False, True],
                                      [False, True, False]])).all()

        def test__even_x_odd_mask_inner_radius_medium_outer_radius_large__correct_mask(self):
            mask = image.Mask.annular(dimensions=(4, 3), pixel_scale=0.1, inner_radius=0.151, outer_radius=0.3)

            assert (mask == np.array([[True, False, True],
                                      [False, False, False],
                                      [False, False, False],
                                      [True, False, True]])).all()

        def test__even_x_even_mask_inner_radius_small_outer_radius_medium__correct_mask(self):
            mask = image.Mask.annular(dimensions=(4, 4), pixel_scale=0.1, inner_radius=0.081, outer_radius=0.2)

            assert (mask == np.array([[False, True, True, False],
                                      [True, False, False, True],
                                      [True, False, False, True],
                                      [False, True, True, False]])).all()

        def test__even_x_even_mask_inner_radius_medium_outer_radius_large__correct_mask(self):
            mask = image.Mask.annular(dimensions=(4, 4), pixel_scale=0.1, inner_radius=0.171, outer_radius=0.3)

            assert (mask == np.array([[True, False, False, True],
                                      [False, False, False, False],
                                      [False, False, False, False],
                                      [True, False, False, True]])).all()


# noinspection PyClassHasNoInit
class TestLoadFits:
    def test__input_fits_3x3_ones__loads_data_as_type_numpy_array(self):
        assert type(image.Image.from_fits('3x3_ones.fits', hdu=0, pixel_scale=1, path=test_data_dir).data) == np.ndarray

    def test__input_fits_3x3_ones__loads_correct_data(self):
        assert (
            image.Image.from_fits('3x3_ones.fits', hdu=0, pixel_scale=1, path=test_data_dir).data == np.ones(
                (3, 3))).all()

    def test__input_fits_4x3_ones__loads_correct_data(self):
        assert (
            image.Image.from_fits('4x3_ones.fits', hdu=0, pixel_scale=1, path=test_data_dir).data == np.ones(
                (4, 3))).all()

    def test__input_files_3x3_ones__loads_correct_dimensions(self):
        xy_dim = image.Image.from_fits('3x3_ones.fits', hdu=0, pixel_scale=1, path=test_data_dir).dimensions

        assert xy_dim[0] == 3
        assert xy_dim[1] == 3

    def test__input_files_4x3_ones__loads_correct_dimensions(self):
        xy_dim = image.Image.from_fits('4x3_ones.fits', hdu=0, pixel_scale=1, path=test_data_dir).dimensions

        assert xy_dim[0] == 4
        assert xy_dim[1] == 3

from __future__ import division, print_function
import pytest
import numpy as np
import data_obj
import os

# TODO: I've used a relative path here so it will work on anyone's computer.
test_data_dir = "{}../../data/test_data".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(scope='class')
def image():
    im = data_obj.Image()
    return im


# TODO: These little notes stop PyCharm complaining about stuff. Normally a class should have an init and variables
# TODO: should not have the same name as function names in the scope of the module. However, I think this necessary
# TODO: when testing
# noinspection PyClassHasNoInit,PyShadowingNames
class TestImage:
    def test__init__input_image_3x3__all_attributes_correct(self, image):
        image.load_fits(dir=test_data_dir, file='3x3_ones.fits', hdu=0, pixel_scale=0.1)

        assert (image.image2d == np.ones((3, 3))).all()
        assert image.xy_dim[0] == 3
        assert image.xy_dim[1] == 3
        assert image.xy_arcsec[0] == pytest.approx(0.3)
        assert image.xy_arcsec[1] == pytest.approx(0.3)

    def test__set_sky_via_edges__input_all_ones__sky_bg_level_1(self, image):
        image.image2d = np.ones((3, 3))
        image.xy_dim = 3, 3
        image.set_sky_via_edges(no_edges=1)

        assert image.sky_background_level == 1.0
        assert image.sky_background_noise == 0.0

    def test__set_sky_via_edges__3x3_image_simple_gaussian__answer_ignores_central_pixel(self, image):
        image.image2d = np.array([[1, 1, 1],
                                  [1, 100, 1],
                                  [1, 1, 1]])
        image.xy_dim = 3, 3

        image.set_sky_via_edges(no_edges=1)

        assert image.sky_background_level == 1.0
        assert image.sky_background_noise == 0.0

    def test__set_sky_via_edges__4x3_image_simple_gaussian__ignores_central_pixels(self, image):
        image.image2d = np.array([[1, 1, 1],
                                  [1, 100, 1],
                                  [1, 100, 1],
                                  [1, 1, 1]])
        image.xy_dim = 4, 3

        image.set_sky_via_edges(no_edges=1)

        assert image.sky_background_level == 1.0
        assert image.sky_background_noise == 0.0

    def test__set_sky_via_edges__4x4_image_simple_gaussian__ignores_central_pixels(self, image):
        image.image2d = np.array([[1, 1, 1, 1],
                                  [1, 100, 100, 1],
                                  [1, 100, 100, 1],
                                  [1, 1, 1, 1]])
        image.xy_dim = 4, 4

        image.set_sky_via_edges(no_edges=1)

        assert image.sky_background_level == 1.0
        assert image.sky_background_noise == 0.0

    def test__set_sky_via_edges__5x5_image_simple_gaussian_two_edges__ignores_central_pixel(self, image):
        image.image2d = np.array([[1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 100, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1]])
        image.xy_dim = 5, 5

        image.set_sky_via_edges(no_edges=2)

        assert image.sky_background_level == 1.0
        assert image.sky_background_noise == 0.0

    def test__set_sky_via_edges__6x5_image_two_edges__correct_values(self, image):
        image.image2d = np.array([[0, 1, 2, 3, 4],
                                  [5, 6, 7, 8, 9],
                                  [10, 11, 100, 12, 13],
                                  [14, 15, 100, 16, 17],
                                  [18, 19, 20, 21, 22],
                                  [23, 24, 25, 26, 27]])
        image.xy_dim = 6, 5

        image.set_sky_via_edges(no_edges=2)

        assert image.sky_background_level == np.mean(np.arange(28))
        assert image.sky_background_noise == np.std(np.arange(28))

    def test__set_sky_via_edges__7x7_image_three_edges__correct_values(self, image):
        image.image2d = np.array([[0, 1, 2, 3, 4, 5, 6],
                                  [7, 8, 9, 10, 11, 12, 13],
                                  [14, 15, 16, 17, 18, 19, 20],
                                  [21, 22, 23, 100, 24, 25, 26],
                                  [27, 28, 29, 30, 31, 32, 33],
                                  [34, 35, 36, 37, 38, 39, 40],
                                  [41, 42, 43, 44, 45, 46, 47]])
        image.xy_dim = 7, 7

        image.set_sky_via_edges(no_edges=3)

        assert image.sky_background_level == np.mean(np.arange(48))
        assert image.sky_background_noise == np.std(np.arange(48))


@pytest.fixture(scope='class')
def mask():
    return data_obj.Mask()


# noinspection PyClassHasNoInit,PyShadowingNames
class TestMask:
    def test__set_circle__input_big_mask__correct_mask(self, mask):
        image = data_obj.Image()
        image.pixel_scale = 0.1
        image.xy_dim = [3, 3]

        mask.set_circle(image=image, mask_radius_arcsec=0.5)

        assert (mask.mask2d == np.ones((3, 3))).all()

    def test__set_circle__odd_x_odd_mask_input_radius_1__correct_mask(self, mask):
        image = data_obj.Image()
        image.pixel_scale = 0.1
        image.xy_dim = [3, 3]

        mask.set_circle(image=image, mask_radius_arcsec=0.05)

        assert (mask.mask2d == np.array([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]])).all()

    def test__set_circle__odd_x_odd_mask_input_radius_2__correct_mask(self, mask):
        image = data_obj.Image()
        image.pixel_scale = 0.1
        image.xy_dim = [3, 3]

        mask.set_circle(image=image, mask_radius_arcsec=0.1)

        assert (mask.mask2d == np.array([[0, 1, 0],
                                         [1, 1, 1],
                                         [0, 1, 0]])).all()

    def test__set_circle__odd_x_odd_mask_input_radius_3__correct_mask(self, mask):
        image = data_obj.Image()
        image.pixel_scale = 0.1
        image.xy_dim = [3, 3]

        mask.set_circle(image=image, mask_radius_arcsec=0.3)

        assert (mask.mask2d == np.array([[1, 1, 1],
                                         [1, 1, 1],
                                         [1, 1, 1]])).all()

    def test__set_circle__even_x_odd_mask_input_radius_1__correct_mask(self, mask):
        image = data_obj.Image()
        image.pixel_scale = 0.1
        image.xy_dim = [4, 3]

        mask.set_circle(image=image, mask_radius_arcsec=0.05)

        assert (mask.mask2d == np.array([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]])).all()

    def test__set_circle__even_x_odd_mask_input_radius_2__correct_mask(self, mask):
        image = data_obj.Image()
        image.pixel_scale = 0.1
        image.xy_dim = [4, 3]

        mask.set_circle(image=image, mask_radius_arcsec=0.150001)

        assert (mask.mask2d == np.array([[0, 1, 0],
                                         [1, 1, 1],
                                         [1, 1, 1],
                                         [0, 1, 0]])).all()

    def test__set_circle__even_x_odd_mask_input_radius_3__correct_mask(self, mask):
        image = data_obj.Image()
        image.pixel_scale = 0.1
        image.xy_dim = [4, 3]

        mask.set_circle(image=image, mask_radius_arcsec=0.3)

        assert (mask.mask2d == np.array([[1, 1, 1],
                                         [1, 1, 1],
                                         [1, 1, 1],
                                         [1, 1, 1]])).all()

    def test__set_circle__even_x_even_mask_input_radius_1__correct_mask(self, mask):
        image = data_obj.Image()
        image.pixel_scale = 0.1
        image.xy_dim = [4, 4]

        mask.set_circle(image=image, mask_radius_arcsec=0.072)

        assert (mask.mask2d == np.array([[0, 0, 0, 0],
                                         [0, 1, 1, 0],
                                         [0, 1, 1, 0],
                                         [0, 0, 0, 0]])).all()

    def test__set_circle__even_x_even_mask_input_radius_2__correct_mask(self, mask):
        image = data_obj.Image()
        image.pixel_scale = 0.1
        image.xy_dim = [4, 4]

        mask.set_circle(image=image, mask_radius_arcsec=0.17)

        assert (mask.mask2d == np.array([[0, 1, 1, 0],
                                         [1, 1, 1, 1],
                                         [1, 1, 1, 1],
                                         [0, 1, 1, 0]])).all()

    def test__set_circle__even_x_even_mask_input_radius_3__correct_mask(self, mask):
        image = data_obj.Image()
        image.pixel_scale = 0.1
        image.xy_dim = [4, 4]

        mask.set_circle(image=image, mask_radius_arcsec=0.3)

        assert (mask.mask2d == np.array([[1, 1, 1, 1],
                                         [1, 1, 1, 1],
                                         [1, 1, 1, 1],
                                         [1, 1, 1, 1]])).all()

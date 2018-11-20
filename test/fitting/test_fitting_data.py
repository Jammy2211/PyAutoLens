import numpy as np
import pytest

from autolens.imaging import convolution
from autolens.imaging import image as im
from autolens.imaging.util import grid_util
from autolens.imaging import mask as msk
from autolens.fitting import fitting_data as fit_data


@pytest.fixture(name='image')
def make_image():
    psf = im.PSF(array=np.ones((3, 3)), pixel_scale=3.0, renormalize=False)
    return im.Image(np.ones((4, 4)), pixel_scale=3., psf=psf, noise_map=np.ones((4, 4)),
                    background_noise_map=2.0*np.ones((4,4)), poisson_noise_map=3.0*np.ones((4,4)),
                    exposure_time_map=None, background_sky_map=5.0 * np.ones((4, 4)))

@pytest.fixture(name="mask")
def make_mask():
    return msk.Mask(np.array([[True, True, True, True],
                              [True, False, False, True],
                              [True, False, False, True],
                              [True, True, True, True]]), pixel_scale=3.0)

@pytest.fixture(name="fitting_image")
def make_lensing_image(image, mask):
    return fit_data.FittingImage(image=image, mask=mask)

class TestFittingImage(object):

    def test_attributes(self, image, fitting_image):
        assert image.pixel_scale == fitting_image.pixel_scale
        assert (image.psf == fitting_image.psf).all()
        assert (image.background_noise_map == fitting_image.background_noise_map).all()
        assert (image.poisson_noise_map == fitting_image.poisson_noise_map).all()
        assert image.exposure_time_map == fitting_image.exposure_time_map
        assert (image.background_sky_map == fitting_image.background_sky_map).all()

    def test__image_and_image_mapper(self, fitting_image):
        assert (fitting_image.image == np.ones((4, 4))).all()
        assert (fitting_image.noise_map == np.ones((4, 4))).all()
        assert (fitting_image.background_noise_map == 2.0*np.ones((4,4))).all()
        assert (fitting_image.poisson_noise_map == 3.0*np.ones((4,4))).all()
        assert fitting_image.exposure_time_map == None
        assert (fitting_image.background_sky_map == 5.0*np.ones((4,4))).all()

    def test_masking(self, fitting_image):
        assert (fitting_image.noise_map_ == np.ones(4)).all()
        assert (fitting_image.background_noise_map_ == 2.0*np.ones(4)).all()
        assert (fitting_image.poisson_noise_map_ == 3.0*np.ones(4)).all()
        assert fitting_image.exposure_time_map_ == None
        assert (fitting_image.background_sky_map_ == 5.0*np.ones(4)).all()

    def test_grids(self, fitting_image):
        assert fitting_image.grids.image.shape == (4, 2)

        assert (fitting_image.grids.image == np.array([[1.5, -1.5], [1.5, 1.5],
                                                       [-1.5, -1.5], [-1.5, 1.5]])).all()
        assert (fitting_image.grids.sub == np.array([[2.0, -2.0], [2.0, -1.0], [1.0, -2.0], [1.0, -1.0],
                                                     [2.0, 1.0], [2.0, 2.0], [1.0, 1.0], [1.0, 2.0],
                                                     [-1.0, -2.0], [-1.0, -1.0], [-2.0, -2.0], [-2.0, -1.0],
                                                     [-1.0, 1.0], [-1.0, 2.0], [-2.0, 1.0], [-2.0, 2.0]])).all()
        assert (fitting_image.grids.blurring == np.array([[4.5, -4.5], [4.5, -1.5], [4.5, 1.5], [4.5, 4.5],
                                                          [1.5, -4.5], [1.5, 4.5], [-1.5, -4.5], [-1.5, 4.5],
                                                          [-4.5, -4.5], [-4.5, -1.5], [-4.5, 1.5], [-4.5, 4.5]])).all()

    def test_padded_grids(self, fitting_image):

        padded_image_util = grid_util.image_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=np.full((6, 6), False),
                                                                                             pixel_scales=fitting_image.image.pixel_scales)

        assert (fitting_image.padded_grids.image == padded_image_util).all()
        assert fitting_image.padded_grids.image.image_shape == (4, 4)
        assert fitting_image.padded_grids.image.padded_shape == (6, 6)

        padded_sub_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((6, 6), False), pixel_scales=fitting_image.image.pixel_scales,
            sub_grid_size=fitting_image.grids.sub.sub_grid_size)

        assert fitting_image.padded_grids.sub == pytest.approx(padded_sub_util, 1e-4)
        assert fitting_image.padded_grids.sub.image_shape == (4, 4)
        assert fitting_image.padded_grids.sub.padded_shape == (6, 6)

        assert (fitting_image.padded_grids.blurring == np.array([[0.0, 0.0]])).all()

    def test_border(self, fitting_image):
        assert (fitting_image.border == np.array([0, 1, 2, 3])).all()

    def test_convolver(self, fitting_image):
        assert type(fitting_image.convolver_image) == convolution.ConvolverImage

    def test_subtract(self, fitting_image):
        subtracted_image = fitting_image - np.array([1, 0, 1, 0])
        assert isinstance(subtracted_image, fit_data.FittingImage)
        assert (subtracted_image.psf == fitting_image.psf).all()
        assert subtracted_image.pixel_scale == fitting_image.pixel_scale

        assert subtracted_image == np.array([0, 1, 0, 1])

    def test__constructor_inputs(self):

        psf = im.PSF(np.ones((7, 7)), 1)
        image = im.Image(np.ones((51, 51)), pixel_scale=3., psf=psf, noise_map=np.ones((51, 51)))
        mask = msk.Mask.masked_for_shape_and_pixel_scale(shape=(51, 51), pixel_scale=1.0)
        mask[26, 26] = False

        fitting_image = fit_data.FittingImage(image, mask, sub_grid_size=8, image_psf_shape=(5, 5))

        assert fitting_image.sub_grid_size == 8
        assert fitting_image.convolver_image.psf_shape == (5, 5)


@pytest.fixture(name="fitting_hyper_image")
def make_fitting_hyper_image(image, mask):
    return fit_data.FittingHyperImage(image=image, mask=mask, hyper_model_image=np.ones(4),
                                      hyper_galaxy_images=[np.ones(4), np.ones(4)], hyper_minimum_values=[0.1, 0.2])


class TestFittingHyperImage(object):

    def test_attributes(self, image, fitting_hyper_image):
        assert image.pixel_scale == fitting_hyper_image.pixel_scale
        assert (image.psf == fitting_hyper_image.psf).all()
        assert (image.background_noise_map == fitting_hyper_image.background_noise_map).all()
        assert (fitting_hyper_image.hyper_model_image == np.ones(4)).all()
        assert (fitting_hyper_image.hyper_galaxy_images[0] == np.ones(4)).all()
        assert (fitting_hyper_image.hyper_galaxy_images[1] == np.ones(4)).all()
        assert fitting_hyper_image.hyper_minimum_values == [0.1, 0.2]
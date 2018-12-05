import numpy as np
import pytest

from autolens.data.imaging import image as im, convolution
from autolens.data.array.util import grid_util
from autolens.data.array import mask as msk
from autolens.lensing import lensing_image as li
from autolens.model.inversion import convolution as inversion_convolution


@pytest.fixture(name='image')
def make_image():
    psf = im.PSF(array=np.ones((3, 3)), pixel_scale=3.0, renormalize=False)
    return im.Image(np.ones((4, 4)), pixel_scale=3., psf=psf, noise_map=np.ones((4, 4)),
                    background_noise_map=2.0*np.ones((4,4)), poisson_noise_map=3.0*np.ones((4,4)),
                    exposure_time_map=6.0 * np.ones((4, 4)), background_sky_map=5.0 * np.ones((4, 4)))

@pytest.fixture(name="mask")
def make_mask():
    return msk.Mask(np.array([[True, True, True, True],
                              [True, False, False, True],
                              [True, False, False, True],
                              [True, True, True, True]]), pixel_scale=3.0)

@pytest.fixture(name="lensing_image")
def make_lensing_image(image, mask):
    return li.LensingImage(image=image, mask=mask)


class TestLensingImage(object):

    def test_attributes(self, image, lensing_image):
        assert image.pixel_scale == lensing_image.pixel_scale
        assert (image.psf == lensing_image.psf).all()
        assert (image.background_noise_map == lensing_image.background_noise_map).all()
        assert (image.poisson_noise_map == lensing_image.poisson_noise_map).all()
        assert (image.exposure_time_map == lensing_image.exposure_time_map).all()
        assert (image.background_sky_map == lensing_image.background_sky_map).all()

    def test__image_and_image_mapper(self, lensing_image):
        assert (lensing_image.image == np.ones((4, 4))).all()
        assert (lensing_image.noise_map == np.ones((4, 4))).all()
        assert (lensing_image.background_noise_map == 2.0*np.ones((4,4))).all()
        assert (lensing_image.poisson_noise_map == 3.0*np.ones((4,4))).all()
        assert (lensing_image.exposure_time_map == 6.0*np.ones((4,4))).all()
        assert (lensing_image.background_sky_map == 5.0*np.ones((4,4))).all()

    def test_masking(self, lensing_image):
        assert (lensing_image.noise_map_ == np.ones(4)).all()
        assert (lensing_image.background_noise_map_ == 2.0*np.ones(4)).all()
        assert (lensing_image.poisson_noise_map_ == 3.0*np.ones(4)).all()
        assert (lensing_image.exposure_time_map_ == 6.0*np.ones(4)).all()
        assert (lensing_image.background_sky_map_ == 5.0*np.ones(4)).all()

    def test_grids(self, lensing_image):
        assert lensing_image.grids.regular.shape == (4, 2)

        assert (lensing_image.grids.regular == np.array([[1.5, -1.5], [1.5, 1.5],
                                                       [-1.5, -1.5], [-1.5, 1.5]])).all()
        assert (lensing_image.grids.sub == np.array([[2.0, -2.0], [2.0, -1.0], [1.0, -2.0], [1.0, -1.0],
                                                     [2.0, 1.0], [2.0, 2.0], [1.0, 1.0], [1.0, 2.0],
                                                     [-1.0, -2.0], [-1.0, -1.0], [-2.0, -2.0], [-2.0, -1.0],
                                                     [-1.0, 1.0], [-1.0, 2.0], [-2.0, 1.0], [-2.0, 2.0]])).all()
        assert (lensing_image.grids.blurring == np.array([[4.5, -4.5], [4.5, -1.5], [4.5, 1.5], [4.5, 4.5],
                                                          [1.5, -4.5], [1.5, 4.5], [-1.5, -4.5], [-1.5, 4.5],
                                                          [-4.5, -4.5], [-4.5, -1.5], [-4.5, 1.5], [-4.5, 4.5]])).all()

    def test_padded_grids(self, lensing_image):

        padded_image_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=np.full((6, 6), False),
                                                                                               pixel_scales=lensing_image.image.pixel_scales)

        assert (lensing_image.padded_grids.regular == padded_image_util).all()
        assert lensing_image.padded_grids.regular.image_shape == (4, 4)
        assert lensing_image.padded_grids.regular.padded_shape == (6, 6)

        padded_sub_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((6, 6), False), pixel_scales=lensing_image.image.pixel_scales,
            sub_grid_size=lensing_image.grids.sub.sub_grid_size)

        assert lensing_image.padded_grids.sub == pytest.approx(padded_sub_util, 1e-4)
        assert lensing_image.padded_grids.sub.image_shape == (4, 4)
        assert lensing_image.padded_grids.sub.padded_shape == (6, 6)

        assert (lensing_image.padded_grids.blurring == np.array([[0.0, 0.0]])).all()

    def test_border(self, lensing_image):
        assert (lensing_image.border == np.array([0, 1, 2, 3])).all()

    def test_convolvers(self, lensing_image):
        assert type(lensing_image.convolver_image) == convolution.ConvolverImage
        assert type(lensing_image.convolver_mapping_matrix) == inversion_convolution.ConvolverMappingMatrix

    def test_subtract(self, lensing_image):
        subtracted_image = lensing_image - np.array([1, 0, 1, 0])
        assert isinstance(subtracted_image, li.LensingImage)
        assert (subtracted_image.psf == lensing_image.psf).all()
        assert subtracted_image.pixel_scale == lensing_image.pixel_scale

        assert subtracted_image == np.array([0, 1, 0, 1])

    def test__constructor_inputs(self):
        psf = im.PSF(np.ones((7, 7)), 1)
        image = im.Image(np.ones((51, 51)), pixel_scale=3., psf=psf, noise_map=np.ones((51, 51)))
        mask = msk.Mask.masked_for_shape_and_pixel_scale(shape=(51, 51), pixel_scale=1.0)
        mask[26, 26] = False

        lensing_image = li.LensingImage(image, mask, sub_grid_size=8, image_psf_shape=(5, 5),
                                        mapping_matrix_psf_shape=(3, 3), positions=[np.array([[1.0, 1.0]])])

        assert lensing_image.sub_grid_size == 8
        assert lensing_image.convolver_image.psf_shape == (5, 5)
        assert lensing_image.convolver_mapping_matrix.psf_shape == (3, 3)
        assert (lensing_image.positions[0] == np.array([[1.0, 1.0]])).all()


@pytest.fixture(name="lensing_hyper_image")
def make_lensing_hyper_image(image, mask):
    return li.LensingHyperImage(image=image, mask=mask, hyper_model_image=np.ones(4),
                                hyper_galaxy_images=[np.ones(4), np.ones(4)], hyper_minimum_values=[0.1, 0.2])


class TestLensingHyperImage(object):

    def test_attributes(self, image, lensing_hyper_image):

        assert image.pixel_scale == lensing_hyper_image.pixel_scale
        assert (image.psf == lensing_hyper_image.psf).all()
        assert (image.background_noise_map == lensing_hyper_image.background_noise_map).all()
        assert (lensing_hyper_image.hyper_model_image == np.ones(4)).all()
        assert (lensing_hyper_image.hyper_galaxy_images[0] == np.ones(4)).all()
        assert (lensing_hyper_image.hyper_galaxy_images[1] == np.ones(4)).all()
        assert lensing_hyper_image.hyper_minimum_values == [0.1, 0.2]

    def test__image_and_image_mapper(self, lensing_hyper_image):
        assert (lensing_hyper_image.image == np.ones((4, 4))).all()
        assert (lensing_hyper_image.noise_map == np.ones((4, 4))).all()
        assert (lensing_hyper_image.background_noise_map == 2.0*np.ones((4,4))).all()
        assert (lensing_hyper_image.poisson_noise_map == 3.0*np.ones((4,4))).all()
        assert (lensing_hyper_image.exposure_time_map == 6.0*np.ones((4,4))).all()
        assert (lensing_hyper_image.background_sky_map == 5.0*np.ones((4,4))).all()

    def test_masking(self, lensing_hyper_image):
        assert (lensing_hyper_image.noise_map_ == np.ones(4)).all()
        assert (lensing_hyper_image.background_noise_map_ == 2.0*np.ones(4)).all()
        assert (lensing_hyper_image.poisson_noise_map_ == 3.0*np.ones(4)).all()
        assert (lensing_hyper_image.exposure_time_map_ == 6.0*np.ones(4)).all()
        assert (lensing_hyper_image.background_sky_map_ == 5.0*np.ones(4)).all()
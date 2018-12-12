import numpy as np
import pytest

from autolens.data.imaging import image
from autolens.data.array import mask as mask
from autolens.lensing.fitting import lensing_fitters_multi
from autolens.model.galaxy import galaxy as g
from autolens.lensing import lensing_image
from autolens.model.profiles import light_profiles as lp


@pytest.fixture(name="no_galaxies", scope='function')
def make_no_galaxies():
    return [g.Galaxy()]


@pytest.fixture(name="sersic")
def make_sersic():
    return lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0)


@pytest.fixture(name="galaxy_light", scope='function')
def make_galaxy_light(sersic):
    return g.Galaxy(light_profile=sersic)


@pytest.fixture(name='li_blur')
def make_li_blur():

    im = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
    psf = image.PSF(array=(np.array([[1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0]])), pixel_scale=1.0, renormalize=False)
    im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = mask.Mask(array=ma, pixel_scale=1.0)

    return lensing_image.LensingImage(im, ma, sub_grid_size=2)


@pytest.fixture(name='li_no_blur')
def make_li_no_blur():
    im = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
    psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]])), pixel_scale=1.0, renormalize=False)
    im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = mask.Mask(array=ma, pixel_scale=1.0)

    return lensing_image.LensingImage(im, ma, sub_grid_size=2)

class TestLensingConvolutionFitterMulti:

    def test__multi_image__2x2_unblurred_image_and_blurred_image__computed_correctly(self, li_no_blur, li_blur):

        unblurred_image_1d_0 = np.array([2.0, 1.0, 1.0, 1.0])
        blurring_image_1d_0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        unblurred_image_1d_1 = np.array([1.0, 1.0, 1.0, 1.0])
        blurring_image_1d_1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        fit = lensing_fitters_multi.LensingConvolutionFitterMulti(lensing_images=[li_no_blur, li_blur],
                                                            unblurred_images_1d=[unblurred_image_1d_0, unblurred_image_1d_1],
                                                            blurring_images_1d=[blurring_image_1d_0, blurring_image_1d_1])

        assert (fit.model_images[0] == np.array([[0.0, 0.0, 0.0, 0.0],
                                                 [0.0, 2.0, 1.0, 0.0],
                                                 [0.0, 1.0, 1.0, 0.0],
                                                 [0.0, 0.0, 0.0, 0.0]])).all()

        assert (fit.model_images[1] == np.array([[0.0, 0.0, 0.0, 0.0],
                                                 [0.0, 9.0, 9.0, 0.0],
                                                 [0.0, 9.0, 9.0, 0.0],
                                                 [0.0, 0.0, 0.0, 0.0]])).all()

        noise_term = 4.0 *np.sum(np.log(2 * np.pi * 1.0 ** 2.0))

        assert fit.likelihoods[0] == -0.5 * (1.0 + noise_term)

        chi_squared_term =  4.0 * (8.0 /1.0 )**2.0

        assert fit.likelihoods[1] == pytest.approx(-0.5 * (chi_squared_term + noise_term), 1.0e-4)
        assert fit.likelihood == fit.likelihoods[0] + fit.likelihoods[1]
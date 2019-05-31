import numpy as np
import pytest

from autolens.data import convolution
from autolens.data.array import mask as msk
from autolens.lens.util import lens_fit_util as util
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp


@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask(array=np.array([[True, True, True, True],
                                    [True, False, False, True],
                                    [True, False, False, True],
                                    [True, True, True, True]]), pixel_scale=1.0)


@pytest.fixture(name='blurring_mask')
def make_blurring_mask():
    return msk.Mask(array=np.array([[False, False, False, False],
                                    [False, True, True, False],
                                    [False, True, True, False],
                                    [False, False, False, False]]), pixel_scale=1.0)


@pytest.fixture(name='convolver_no_blur')
def make_convolver_no_blur(mask, blurring_mask):
    psf = np.array([[0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0]])

    return convolution.ConvolverImage(mask=mask, blurring_mask=blurring_mask, psf=psf)


@pytest.fixture(name='convolver_blur')
def make_convolver_blur(mask, blurring_mask):
    psf = np.array([[1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0]])

    return convolution.ConvolverImage(mask=mask, blurring_mask=blurring_mask, psf=psf)


@pytest.fixture(name="galaxy_light")
def make_galaxy_light():
    return g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                      effective_radius=0.6, sersic_index=4.0))


class TestInversionEvidence:

    def test__simple_values(self):
        likelihood_with_regularization_terms = \
            util.likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(
                chi_squared=3.0, regularization_term=6.0, noise_normalization=2.0)

        assert likelihood_with_regularization_terms == -0.5 * (3.0 + 6.0 + 2.0)

        evidences = util.evidence_from_inversion_terms(chi_squared=3.0, regularization_term=6.0,
                                                       log_curvature_regularization_term=9.0,
                                                       log_regularization_term=10.0, noise_normalization=30.0)

        assert evidences == -0.5 * (3.0 + 6.0 + 9.0 - 10.0 + 30.0)


# class TestUnmaskedModelImage:
#
#     def test___3x3_padded_image__asymmetric_psf_blurring__produces_padded_image(self):
#         mask = msk.Mask(array=np.array([[True, True, True],
#                                         [True, False, True],
#                                         [True, True, True]]), pixel_scale=1.0)
#
#         psf = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
#                                       [0.0, 1.0, 2.0],
#                                       [0.0, 0.0, 0.0]])), pixel_scale=1.0)
#
#         padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
#                                                                                                     sub_grid_size=1,
#                                                                                                     psf_shape=(3, 3))
#
#         unmasked_image_1d = np.zeros(25)
#         unmasked_image_1d[12] = 1.0
#
#         unmasked_blurred_image = padded_grid_stack.unmasked_blurred_image_from_psf_and_unmasked_image(
#             psf=psf, unmasked_image_1d=unmasked_image_1d)
#
#         assert (unmasked_blurred_image == np.array([[0.0, 3.0, 0.0],
#                                                     [0.0, 1.0, 2.0],
#                                                     [0.0, 0.0, 0.0]])).all()
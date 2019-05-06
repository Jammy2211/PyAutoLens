import numpy as np
import pytest
from astropy import cosmology as cosmo

from autolens.data import ccd as im, convolution
from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.data.array import scaled_array
from autolens.lens import ray_tracing
from autolens.lens.util import lens_fit_util as util
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy.util import galaxy_util
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.model.profiles import light_profiles as lp
from test.unit.mock.mock_galaxy import MockHyperGalaxy


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
    return g.Galaxy(light_profile=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=1.0, phi=0.0, intensity=1.0,
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


class TestUnmaskedModelImage:

    def test___3x3_padded_image__no_psf_blurring__produces_padded_image(self):
        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        psf = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        unmasked_blurred_image = padded_grid_stack.unmasked_blurred_image_from_psf_and_unmasked_image(
            psf=psf, unmasked_image_1d=np.ones(25))

        assert (unmasked_blurred_image == np.ones((3, 3))).all()

    def test___3x3_padded_image__simple_psf_blurring__produces_padded_image(self):
        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        psf = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                      [0.0, 1.0, 2.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        unmasked_blurred_image = padded_grid_stack.unmasked_blurred_image_from_psf_and_unmasked_image(
            psf=psf, unmasked_image_1d=np.ones(25))

        assert (unmasked_blurred_image == 3.0 * np.ones((3, 3))).all()

    def test___3x3_padded_image__asymmetric_psf_blurring__produces_padded_image(self):
        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        psf = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                      [0.0, 1.0, 2.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        unmasked_image_1d = np.zeros(25)
        unmasked_image_1d[12] = 1.0

        unmasked_blurred_image = padded_grid_stack.unmasked_blurred_image_from_psf_and_unmasked_image(
            psf=psf, unmasked_image_1d=unmasked_image_1d)

        assert (unmasked_blurred_image == np.array([[0.0, 3.0, 0.0],
                                                    [0.0, 1.0, 2.0],
                                                    [0.0, 0.0, 0.0]])).all()


class TestUnmaskedModelImageOfPlanes:

    def test___3x3_padded_image__no_psf_blurring(self, galaxy_light):
        psf = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grid_stack=padded_grid_stack)

        manual_blurred_image_0 = tracer.image_plane.image_plane_image_1d
        manual_blurred_image_0 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_0)
        manual_blurred_image_0 = psf.convolve(array=manual_blurred_image_0)

        unmasked_blurred_image_of_planes = \
            util.unmasked_blurred_image_of_planes_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                 padded_grid_stack=padded_grid_stack,
                                                                                 psf=psf)

        assert (unmasked_blurred_image_of_planes[0] == manual_blurred_image_0[1:4, 1:4]).all()

    def test___x1_galaxy__3x3_padded_image__asymetric_psf_blurring(self, galaxy_light):
        psf = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                      [0.0, 1.0, 2.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(
                                                                                                        3, 3))

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grid_stack=padded_grid_stack)

        manual_blurred_image_0 = tracer.image_plane.image_plane_image_1d
        manual_blurred_image_0 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_0)
        manual_blurred_image_0 = psf.convolve(array=manual_blurred_image_0)

        unmasked_blurred_image_of_planes = \
            util.unmasked_blurred_image_of_planes_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                 padded_grid_stack=padded_grid_stack,
                                                                                 psf=psf)

        assert (unmasked_blurred_image_of_planes[0] == manual_blurred_image_0[1:4, 1:4]).all()

    def test___x2_galaxies__3x3_padded_image__asymetric_psf_blurring(self):
        psf = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                      [0.0, 1.0, 2.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1], image_plane_grid_stack=padded_grid_stack)

        manual_blurred_image_0 = tracer.image_plane.image_plane_image_1d
        manual_blurred_image_0 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_0)
        manual_blurred_image_0 = psf.convolve(array=manual_blurred_image_0)

        unmasked_blurred_image_of_planes = \
            util.unmasked_blurred_image_of_planes_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                 padded_grid_stack=padded_grid_stack,
                                                                                 psf=psf)

        assert unmasked_blurred_image_of_planes[0] == \
               pytest.approx(manual_blurred_image_0[1:4, 1:4], 1.0e-4)

    def test___same_as_above_but_image_and_souce_plane(self):
        psf = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                      [0.0, 1.0, 2.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))
        g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.3))
        g3 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.4))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2, g3],
                                                     image_plane_grid_stack=padded_grid_stack)

        manual_blurred_image_0 = tracer.image_plane.image_plane_image_1d
        manual_blurred_image_0 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_0)
        manual_blurred_image_0 = psf.convolve(array=manual_blurred_image_0)

        manual_blurred_image_1 = tracer.source_plane.image_plane_image_1d
        manual_blurred_image_1 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_1)
        manual_blurred_image_1 = psf.convolve(array=manual_blurred_image_1)

        unmasked_blurred_image_of_planes = \
            util.unmasked_blurred_image_of_planes_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                 padded_grid_stack=padded_grid_stack,
                                                                                 psf=psf)

        assert unmasked_blurred_image_of_planes[0] == pytest.approx(manual_blurred_image_0[1:4, 1:4], 1.0e-4)
        assert unmasked_blurred_image_of_planes[1] == pytest.approx(manual_blurred_image_1[1:4, 1:4], 1.0e-4)

    def test___same_as_above_but_image_and_souce_plane__compare_to_images_of_galaxies(self):
        psf = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                      [0.0, 1.0, 2.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))
        g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.3))
        g3 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.4))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2, g3],
                                                     image_plane_grid_stack=padded_grid_stack)

        manual_blurred_image_0 = tracer.image_plane.image_plane_image_1d_of_galaxies[0]
        manual_blurred_image_0 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_0)
        manual_blurred_image_0 = psf.convolve(array=manual_blurred_image_0)

        manual_blurred_image_1 = tracer.image_plane.image_plane_image_1d_of_galaxies[1]
        manual_blurred_image_1 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_1)
        manual_blurred_image_1 = psf.convolve(array=manual_blurred_image_1)

        manual_blurred_image_2 = tracer.source_plane.image_plane_image_1d_of_galaxies[0]
        manual_blurred_image_2 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_2)
        manual_blurred_image_2 = psf.convolve(array=manual_blurred_image_2)

        manual_blurred_image_3 = tracer.source_plane.image_plane_image_1d_of_galaxies[1]
        manual_blurred_image_3 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_3)
        manual_blurred_image_3 = psf.convolve(array=manual_blurred_image_3)

        unmasked_blurred_image_of_planes = \
            util.unmasked_blurred_image_of_planes_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                 padded_grid_stack=padded_grid_stack,
                                                                                 psf=psf)

        assert unmasked_blurred_image_of_planes[0] == \
               pytest.approx(manual_blurred_image_0[1:4, 1:4] + manual_blurred_image_1[1:4, 1:4], 1.0e-4)
        assert unmasked_blurred_image_of_planes[1] == \
               pytest.approx(manual_blurred_image_2[1:4, 1:4] + manual_blurred_image_3[1:4, 1:4], 1.0e-4)

    def test__if_plane_has_pixelization__unmasked_image_returns_none(self):
        psf = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                      [0.0, 1.0, 2.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))
        g_pix = g.Galaxy(pixelization=pix.Rectangular(), regularization=reg.Constant())

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1, g_pix],
                                                     image_plane_grid_stack=padded_grid_stack)

        unmasked_blurred_image_of_planes = \
            util.unmasked_blurred_image_of_planes_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                 padded_grid_stack=padded_grid_stack,
                                                                                 psf=psf)

        assert type(unmasked_blurred_image_of_planes[0]) == scaled_array.ScaledSquarePixelArray
        assert unmasked_blurred_image_of_planes[1] == None

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g_pix], source_galaxies=[g1],
                                                     image_plane_grid_stack=padded_grid_stack)

        unmasked_blurred_image_of_planes = \
            util.unmasked_blurred_image_of_planes_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                 padded_grid_stack=padded_grid_stack,
                                                                                 psf=psf)

        assert unmasked_blurred_image_of_planes[0] == None
        assert type(unmasked_blurred_image_of_planes[1]) == scaled_array.ScaledSquarePixelArray

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g_pix], source_galaxies=[g_pix],
                                                     image_plane_grid_stack=padded_grid_stack)

        unmasked_blurred_image_of_planes = \
            util.unmasked_blurred_image_of_planes_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                 padded_grid_stack=padded_grid_stack,
                                                                                 psf=psf)

        assert unmasked_blurred_image_of_planes[0] == None
        assert unmasked_blurred_image_of_planes[1] == None


class TestUnmaskedModelImageOfPlanesAndGalaxies:

    def test___3x3_padded_image__no_psf_blurring(self, galaxy_light):
        psf = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grid_stack=padded_grid_stack)

        manual_blurred_image_0 = tracer.image_plane.image_plane_image_1d_of_galaxies[0]
        manual_blurred_image_0 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_0)
        manual_blurred_image_0 = psf.convolve(array=manual_blurred_image_0)

        unmasked_blurred_image_of_planes_and_galaxies = \
            util.unmasked_blurred_image_of_planes_and_galaxies_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                              padded_grid_stack=padded_grid_stack,
                                                                                              psf=psf)

        assert (unmasked_blurred_image_of_planes_and_galaxies[0][0] == manual_blurred_image_0[1:4, 1:4]).all()

    def test___x1_galaxy__3x3_padded_image__asymetric_psf_blurring(self, galaxy_light):
        psf = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                      [0.0, 1.0, 2.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(
                                                                                                        3, 3))

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grid_stack=padded_grid_stack)

        manual_blurred_image_0 = tracer.image_plane.image_plane_image_1d_of_galaxies[0]
        manual_blurred_image_0 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_0)
        manual_blurred_image_0 = psf.convolve(array=manual_blurred_image_0)

        unmasked_blurred_image_of_planes_and_galaxies = \
            util.unmasked_blurred_image_of_planes_and_galaxies_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                              padded_grid_stack=padded_grid_stack,
                                                                                              psf=psf)

        assert (unmasked_blurred_image_of_planes_and_galaxies[0][0] == manual_blurred_image_0[1:4, 1:4]).all()

    def test___x2_galaxies__3x3_padded_image__asymetric_psf_blurring(self):
        psf = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                      [0.0, 1.0, 2.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1], image_plane_grid_stack=padded_grid_stack)

        manual_blurred_image_0 = tracer.image_plane.image_plane_image_1d_of_galaxies[0]
        manual_blurred_image_0 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_0)
        manual_blurred_image_0 = psf.convolve(array=manual_blurred_image_0)

        manual_blurred_image_1 = tracer.image_plane.image_plane_image_1d_of_galaxies[1]
        manual_blurred_image_1 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_1)
        manual_blurred_image_1 = psf.convolve(array=manual_blurred_image_1)

        unmasked_blurred_image_of_planes_and_galaxies = \
            util.unmasked_blurred_image_of_planes_and_galaxies_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                              padded_grid_stack=padded_grid_stack,
                                                                                              psf=psf)

        assert (unmasked_blurred_image_of_planes_and_galaxies[0][0] == manual_blurred_image_0[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_planes_and_galaxies[0][1] == manual_blurred_image_1[1:4, 1:4]).all()

    def test___same_as_above_but_image_and_souce_plane(self):
        psf = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                      [0.0, 1.0, 2.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))
        g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.3))
        g3 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.4))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2, g3],
                                                     image_plane_grid_stack=padded_grid_stack)

        manual_blurred_image_0 = tracer.image_plane.image_plane_image_1d_of_galaxies[0]
        manual_blurred_image_0 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_0)
        manual_blurred_image_0 = psf.convolve(array=manual_blurred_image_0)

        manual_blurred_image_1 = tracer.image_plane.image_plane_image_1d_of_galaxies[1]
        manual_blurred_image_1 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_1)
        manual_blurred_image_1 = psf.convolve(array=manual_blurred_image_1)

        manual_blurred_image_2 = tracer.source_plane.image_plane_image_1d_of_galaxies[0]
        manual_blurred_image_2 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_2)
        manual_blurred_image_2 = psf.convolve(array=manual_blurred_image_2)

        manual_blurred_image_3 = tracer.source_plane.image_plane_image_1d_of_galaxies[1]
        manual_blurred_image_3 = padded_grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=manual_blurred_image_3)
        manual_blurred_image_3 = psf.convolve(array=manual_blurred_image_3)

        unmasked_blurred_image_of_planes_and_galaxies = \
            util.unmasked_blurred_image_of_planes_and_galaxies_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                              padded_grid_stack=padded_grid_stack,
                                                                                              psf=psf)

        assert (unmasked_blurred_image_of_planes_and_galaxies[0][0] == manual_blurred_image_0[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_planes_and_galaxies[0][1] == manual_blurred_image_1[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_planes_and_galaxies[1][0] == manual_blurred_image_2[1:4, 1:4]).all()
        assert (unmasked_blurred_image_of_planes_and_galaxies[1][1] == manual_blurred_image_3[1:4, 1:4]).all()

    def test___if_galaxy_has_pixelization__unmasked_image_is_none(self):
        psf = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                      [0.0, 1.0, 2.0],
                                      [0.0, 0.0, 0.0]])), pixel_scale=1.0)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]]), pixel_scale=1.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1),
                      pixelization=pix.Rectangular(), regularization=reg.Constant())
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))
        g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.3))
        g3 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.4))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2, g3],
                                                     image_plane_grid_stack=padded_grid_stack)

        unmasked_blurred_image_of_planes_and_galaxies = \
            util.unmasked_blurred_image_of_planes_and_galaxies_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                              padded_grid_stack=padded_grid_stack,
                                                                                              psf=psf)

        assert unmasked_blurred_image_of_planes_and_galaxies[0][0] == None
        assert type(unmasked_blurred_image_of_planes_and_galaxies[0][1]) == scaled_array.ScaledSquarePixelArray
        assert type(unmasked_blurred_image_of_planes_and_galaxies[1][0]) == scaled_array.ScaledSquarePixelArray
        assert type(unmasked_blurred_image_of_planes_and_galaxies[1][1]) == scaled_array.ScaledSquarePixelArray

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2),
                      pixelization=pix.Rectangular(), regularization=reg.Constant())
        g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.3))
        g3 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.4),
                      pixelization=pix.Rectangular(), regularization=reg.Constant())

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2, g3],
                                                     image_plane_grid_stack=padded_grid_stack)

        unmasked_blurred_image_of_planes_and_galaxies = \
            util.unmasked_blurred_image_of_planes_and_galaxies_from_padded_grid_stack_and_psf(planes=tracer.planes,
                                                                                              padded_grid_stack=padded_grid_stack,
                                                                                              psf=psf)

        assert type(unmasked_blurred_image_of_planes_and_galaxies[0][0]) == scaled_array.ScaledSquarePixelArray
        assert unmasked_blurred_image_of_planes_and_galaxies[0][1] == None
        assert type(unmasked_blurred_image_of_planes_and_galaxies[1][0]) == scaled_array.ScaledSquarePixelArray
        assert unmasked_blurred_image_of_planes_and_galaxies[1][1] == None

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


class TestBlurImages:

    def test__2x2_image_all_1s__3x3__psf_central_1__no_blurring(self, convolver_no_blur):
        image_1d = np.array([1.0, 1.0, 1.0, 1.0])
        blurring_image_1d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        blurred_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(unblurred_image_1d=image_1d,
                                                                                       blurring_image_1d=blurring_image_1d,
                                                                                       convolver=convolver_no_blur)

        assert (blurred_image_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__2x2_image_all_1s__3x3_psf_all_1s__image_blurs_to_4s(self, convolver_blur):
        image_1d = np.array([1.0, 1.0, 1.0, 1.0])
        blurring_image_1d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        blurred_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(unblurred_image_1d=image_1d,
                                                                                       blurring_image_1d=blurring_image_1d,
                                                                                       convolver=convolver_blur)
        assert (blurred_image_1d == np.array([4.0, 4.0, 4.0, 4.0])).all()

    def test__2x2_unblurred_image_all_1s__3x3_psf_no_blurring__image_blurs_to_itself(self, convolver_no_blur):
        unblurred_image_1d = np.array([1.0, 1.0, 1.0, 1.0])
        blurring_image_1d = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        blurred_image = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=unblurred_image_1d, blurring_image_1d=blurring_image_1d, convolver=convolver_no_blur)

        assert (blurred_image == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__2x2_unblurred_image_all_1s__3x3_psf_all_1s__image_blurs_to_9s(self, convolver_blur):
        unblurred_image_1d = np.array([1.0, 1.0, 1.0, 1.0])
        blurring_image_1d = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        blurred_image = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=unblurred_image_1d, blurring_image_1d=blurring_image_1d, convolver=convolver_blur)

        assert (blurred_image == np.array([9.0, 9.0, 9.0, 9.0])).all()

    def test__2x2_image__psf_is_non_symmetric_producing_l_shape(self, mask, blurring_mask):
        psf = np.array([[0.0, 3.0, 0.0],
                        [0.0, 2.0, 1.0],
                        [0.0, 0.0, 0.0]])

        convolver = convolution.ConvolverImage(mask=mask, blurring_mask=blurring_mask, psf=psf)

        unblurred_image_1d = np.array([1.0, 2.0, 3.0, 4.0])
        blurring_image_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

        blurred_image = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=unblurred_image_1d, blurring_image_1d=blurring_image_1d, convolver=convolver)

        # Manually compute result of convolution, which is each central value *2.0 plus its 2 appropriate neighbors

        blurred_image_manual_0 = 2.0 * unblurred_image_1d[0] + 3.0 * unblurred_image_1d[2] + blurring_image_1d[4]
        blurred_image_manual_1 = 2.0 * unblurred_image_1d[1] + 3.0 * unblurred_image_1d[3] + unblurred_image_1d[0]
        blurred_image_manual_2 = 2.0 * unblurred_image_1d[2] + 3.0 * blurring_image_1d[9] + blurring_image_1d[6]
        blurred_image_manual_3 = 2.0 * unblurred_image_1d[3] + 3.0 * blurring_image_1d[10] + unblurred_image_1d[2]

        assert blurred_image_manual_0 == pytest.approx(blurred_image[0], 1e-6)
        assert blurred_image_manual_1 == pytest.approx(blurred_image[1], 1e-6)
        assert blurred_image_manual_2 == pytest.approx(blurred_image[2], 1e-6)
        assert blurred_image_manual_3 == pytest.approx(blurred_image[3], 1e-6)


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


class TestBlurredImageOfPlanes:

    def test__blurred_image_of_planes__real_tracer__2x2_image__psf_is_non_symmetric_producing_l_shape(self, mask,
                                                                                                      convolver_blur):
        data_grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask, sub_grid_size=1,
                                                                                           psf_shape=(3, 3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                     image_plane_grid_stack=data_grid_stack)

        blurred_lens_image = convolver_blur.convolve_image(image_array=tracer.image_plane.image_plane_image_1d,
                                                           blurring_array=tracer.image_plane.image_plane_blurring_image_1d)
        blurred_lens_image = data_grid_stack.regular.scaled_array_2d_from_array_1d(array_1d=blurred_lens_image)

        blurred_source_image = convolver_blur.convolve_image(image_array=tracer.source_plane.image_plane_image_1d,
                                                             blurring_array=tracer.source_plane.image_plane_blurring_image_1d)
        blurred_source_image = data_grid_stack.regular.scaled_array_2d_from_array_1d(array_1d=blurred_source_image)

        blurred_image_of_planes = util.blurred_image_of_planes_from_1d_images_and_convolver(
            total_planes=tracer.total_planes, image_plane_image_1d_of_planes=tracer.image_plane_image_1d_of_planes,
            image_plane_blurring_image_1d_of_planes=tracer.image_plane_blurring_image_1d_of_planes,
            convolver=convolver_blur, map_to_scaled_array=data_grid_stack.regular.array_2d_from_array_1d)

        assert (blurred_image_of_planes[0] == blurred_lens_image).all()
        assert (blurred_image_of_planes[1] == blurred_source_image).all()

    def test__same_as_above_but_multi_tracer(self, mask, convolver_blur):
        data_grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                           sub_grid_size=1,
                                                                                           psf_shape=(3, 3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0), redshift=0.1)
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0), redshift=0.2)
        g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0), redshift=0.3)

        tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2], image_plane_grid_stack=data_grid_stack,
                                               cosmology=cosmo.Planck15)

        blurred_plane_image_0 = convolver_blur.convolve_image(tracer.planes[0].image_plane_image_1d,
                                                              tracer.planes[0].image_plane_blurring_image_1d)

        blurred_plane_image_1 = convolver_blur.convolve_image(tracer.planes[1].image_plane_image_1d,
                                                              tracer.planes[1].image_plane_blurring_image_1d)

        blurred_plane_image_2 = convolver_blur.convolve_image(tracer.planes[2].image_plane_image_1d,
                                                              tracer.planes[2].image_plane_blurring_image_1d)

        blurred_plane_image_0 = data_grid_stack.regular.scaled_array_2d_from_array_1d(blurred_plane_image_0)
        blurred_plane_image_1 = data_grid_stack.regular.scaled_array_2d_from_array_1d(blurred_plane_image_1)
        blurred_plane_image_2 = data_grid_stack.regular.scaled_array_2d_from_array_1d(blurred_plane_image_2)

        blurred_image_of_planes = util.blurred_image_of_planes_from_1d_images_and_convolver(
            total_planes=tracer.total_planes, image_plane_image_1d_of_planes=tracer.image_plane_image_1d_of_planes,
            image_plane_blurring_image_1d_of_planes=tracer.image_plane_blurring_image_1d_of_planes,
            convolver=convolver_blur, map_to_scaled_array=data_grid_stack.regular.array_2d_from_array_1d)

        assert (blurred_image_of_planes[0] == blurred_plane_image_0).all()
        assert (blurred_image_of_planes[1] == blurred_plane_image_1).all()
        assert (blurred_image_of_planes[2] == blurred_plane_image_2).all()

    def test__blurred_images_of_planes__if_galaxy_has_no_light_profile__replace_with_none(self, mask, convolver_blur):
        data_grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                           sub_grid_size=1,
                                                                                           psf_shape=(3, 3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
        g0_image = galaxy_util.intensities_of_galaxies_from_grid(grid=data_grid_stack.sub, galaxies=[g0])
        g0_blurring_image = galaxy_util.intensities_of_galaxies_from_grid(grid=data_grid_stack.blurring, galaxies=[g0])
        g0_blurred_image_1d = convolver_blur.convolve_image(image_array=g0_image, blurring_array=g0_blurring_image)
        g0_blurred_image = data_grid_stack.regular.scaled_array_2d_from_array_1d(array_1d=g0_blurred_image_1d)

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                     image_plane_grid_stack=data_grid_stack)

        blurred_image_of_planes = util.blurred_image_of_planes_from_1d_images_and_convolver(
            total_planes=tracer.total_planes, image_plane_image_1d_of_planes=tracer.image_plane_image_1d_of_planes,
            image_plane_blurring_image_1d_of_planes=tracer.image_plane_blurring_image_1d_of_planes,
            convolver=convolver_blur, map_to_scaled_array=data_grid_stack.regular.array_2d_from_array_1d)

        assert (blurred_image_of_planes[0] == g0_blurred_image).all()
        assert (blurred_image_of_planes[1] == g0_blurred_image).all()

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g.Galaxy()],
                                                     image_plane_grid_stack=data_grid_stack)

        blurred_image_of_planes = util.blurred_image_of_planes_from_1d_images_and_convolver(
            total_planes=tracer.total_planes, image_plane_image_1d_of_planes=tracer.image_plane_image_1d_of_planes,
            image_plane_blurring_image_1d_of_planes=tracer.image_plane_blurring_image_1d_of_planes,
            convolver=convolver_blur, map_to_scaled_array=data_grid_stack.regular.array_2d_from_array_1d)

        assert (blurred_image_of_planes[0] == g0_blurred_image).all()
        assert blurred_image_of_planes[1] == None

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[g0],
                                                     image_plane_grid_stack=data_grid_stack)

        blurred_image_of_planes = util.blurred_image_of_planes_from_1d_images_and_convolver(
            total_planes=tracer.total_planes, image_plane_image_1d_of_planes=tracer.image_plane_image_1d_of_planes,
            image_plane_blurring_image_1d_of_planes=tracer.image_plane_blurring_image_1d_of_planes,
            convolver=convolver_blur, map_to_scaled_array=data_grid_stack.regular.array_2d_from_array_1d)

        assert blurred_image_of_planes[0] == None
        assert (blurred_image_of_planes[1] == g0_blurred_image).all()

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[g.Galaxy()],
                                                     image_plane_grid_stack=data_grid_stack)

        blurred_image_of_planes = util.blurred_image_of_planes_from_1d_images_and_convolver(
            total_planes=tracer.total_planes, image_plane_image_1d_of_planes=tracer.image_plane_image_1d_of_planes,
            image_plane_blurring_image_1d_of_planes=tracer.image_plane_blurring_image_1d_of_planes,
            convolver=convolver_blur, map_to_scaled_array=data_grid_stack.regular.array_2d_from_array_1d)

        assert blurred_image_of_planes[0] == None
        assert blurred_image_of_planes[1] == None


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


class TestContributionsFromHypers:

    def test__x1_hyper_galaxy__model_is_galaxy_image__contributions_all_1(self):
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image_1d = np.array([1.0, 1.0, 1.0, 1.0])

        hyper_galaxy_images_1d = [np.array([1.0, 1.0, 1.0, 1.0])]

        minimum_values = [0.0]

        contribution_maps_1d = util.contribution_maps_1d_from_hyper_images_and_galaxies(
            hyper_model_image_1d=hyper_model_image_1d, hyper_galaxy_images_1d=hyper_galaxy_images_1d,
            hyper_galaxies=hyper_galaxies, hyper_minimum_values=minimum_values)

        assert (contribution_maps_1d[0] == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__x1_hyper_galaxy__model_and_galaxy_image_different_contributions_change(self):
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image_1d = np.array([0.5, 1.0, 1.5, 0.5])

        hyper_galaxy_images_1d = [np.array([0.5, 1.0, 1.5, 0.5])]

        minimum_values = [0.6]

        contribution_maps_1d = util.contribution_maps_1d_from_hyper_images_and_galaxies(
            hyper_model_image_1d=hyper_model_image_1d, hyper_galaxy_images_1d=hyper_galaxy_images_1d,
            hyper_galaxies=hyper_galaxies,
            hyper_minimum_values=minimum_values)

        assert (contribution_maps_1d[0] == np.array([0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0, 0.0])).all()

    def test__x2_hyper_galaxy__model_and_galaxy_image_different_contributions_change(self):
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
                          MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image_1d = np.array([0.5, 1.0, 1.5])

        hyper_galaxy_images_1d = [np.array([0.5, 1.0, 1.5]), np.array([0.5, 1.0, 1.5])]

        minimum_values = [0.5, 0.6]

        contribution_maps_1d = util.contribution_maps_1d_from_hyper_images_and_galaxies(
            hyper_model_image_1d=hyper_model_image_1d, hyper_galaxy_images_1d=hyper_galaxy_images_1d,
            hyper_galaxies=hyper_galaxies,
            hyper_minimum_values=minimum_values)

        assert (contribution_maps_1d[0] == np.array([1.0, 1.0, 1.0])).all()
        assert (contribution_maps_1d[1] == np.array([0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0])).all()

    def test__x2_hyper_galaxy__same_as_above_use_real_hyper_galaxy(self):
        hyper_galaxies = [g.HyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
                          g.HyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image_1d = np.array([0.5, 1.0, 1.5, 0.5])

        hyper_galaxy_images_1d = [np.array([0.5, 1.0, 1.5, 0.5]), np.array([0.5, 1.0, 1.5, 0.5])]

        minimum_values = [0.5, 0.6]

        contribution_maps_1d = util.contribution_maps_1d_from_hyper_images_and_galaxies(
            hyper_model_image_1d=hyper_model_image_1d, hyper_galaxy_images_1d=hyper_galaxy_images_1d,
            hyper_galaxies=hyper_galaxies,
            hyper_minimum_values=minimum_values)

        assert (contribution_maps_1d[0] == np.array([1.0, 1.0, 1.0, 1.0])).all()
        assert (contribution_maps_1d[1] == np.array([0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0, 0.0])).all()


class TestScaledNoiseFromContributions:

    def test__x1_hyper_galaxy__noise_factor_is_0__scaled_noise_is_input_noise(self):
        contributions_1d = [np.array([1.0, 1.0, 2.0])]
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]
        noise_map_ = np.array([1.0, 1.0, 1.0])

        scaled_noise_map_ = util.scaled_noise_map_from_hyper_galaxies_and_contribution_maps(
            contribution_maps=contributions_1d, hyper_galaxies=hyper_galaxies, noise_map=noise_map_)

        assert (scaled_noise_map_ == noise_map_).all()

    def test__x1_hyper_galaxy__noise_factor_and_power_are_1__scaled_noise_added_to_input_noise(self):
        contributions_1d = [np.array([1.0, 1.0, 0.5])]
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=1.0)]
        noise_map_ = np.array([1.0, 1.0, 1.0])

        scaled_noise_map_ = util.scaled_noise_map_from_hyper_galaxies_and_contribution_maps(
            contribution_maps=contributions_1d, hyper_galaxies=hyper_galaxies, noise_map=noise_map_)

        assert (scaled_noise_map_ == np.array([2.0, 2.0, 1.5])).all()

    def test__x1_hyper_galaxy__noise_factor_1_and_power_is_2__scaled_noise_added_to_input_noise(self):
        contributions_1d = [np.array([1.0, 1.0, 0.5])]
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0)]
        noise_map_ = np.array([1.0, 1.0, 1.0])

        scaled_noise_map_ = util.scaled_noise_map_from_hyper_galaxies_and_contribution_maps(
            contribution_maps=contributions_1d, hyper_galaxies=hyper_galaxies, noise_map=noise_map_)

        assert (scaled_noise_map_ == np.array([2.0, 2.0, 1.25])).all()

    def test__x2_hyper_galaxy__noise_factor_1_and_power_is_2__scaled_noise_added_to_input_noise(self):
        contributions_1d = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
                          MockHyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]
        noise_map_ = np.array([1.0, 1.0, 1.0])

        scaled_noise_map_ = util.scaled_noise_map_from_hyper_galaxies_and_contribution_maps(
            contribution_maps=contributions_1d, hyper_galaxies=hyper_galaxies, noise_map=noise_map_)

        assert (scaled_noise_map_ == np.array([2.5, 2.5, 1.75])).all()

    def test__x2_hyper_galaxy__same_as_above_but_use_real_hyper_galaxy(self):
        contributions_1d = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
        hyper_galaxies = [g.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
                          g.HyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]
        noise_map_ = np.array([1.0, 1.0, 1.0])

        scaled_noise_map_ = util.scaled_noise_map_from_hyper_galaxies_and_contribution_maps(
            contribution_maps=contributions_1d, hyper_galaxies=hyper_galaxies, noise_map=noise_map_)

        assert (scaled_noise_map_ == np.array([2.5, 2.5, 1.75])).all()

    # def test__same_as_above_but_using_fiting_hyper_image(self, image, masks):
    #
    #     fitting_hyper_image = fit_data.FitDataHyper(datas=image, masks=masks, hyper_model_image=None,
    #                                                 hyper_galaxy_images=None, hyper_minimum_values=None)
    #
    #     contribution_maps_1d = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
    #     hyper_galaxies = [g.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
    #                       g.HyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]
    #     scaled_noises = lens_fit_util.scaled_noise_maps_from_fitting_hyper_images_contributions_and_hyper_galaxies(
    #         fitting_hyper_images=[fitting_hyper_image], contributions_1d=[contribution_maps_1d], hyper_galaxies=hyper_galaxies)
    #
    #     assert (scaled_noises[0] == np.array([2.5, 2.5, 1.75])).all()
    #
    # def test__same_as_above_but_using_x2_fiting_hyper_image(self, image, masks):
    #
    #     fitting_hyper_image_0 = fit_data.FitDataHyper(datas=image, masks=masks, hyper_model_image=None,
    #                                                   hyper_galaxy_images=None, hyper_minimum_values=None)
    #     fitting_hyper_image_1 = fit_data.FitDataHyper(datas=image, masks=masks, hyper_model_image=None,
    #                                                   hyper_galaxy_images=None, hyper_minimum_values=None)
    #
    #     contributions_0 = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
    #     contributions_1 = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
    #     hyper_galaxies = [g.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
    #                       g.HyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]
    #     scaled_noises = lens_fit_util.scaled_noise_maps_from_fitting_hyper_images_contributions_and_hyper_galaxies(
    #         fitting_hyper_images=[fitting_hyper_image_0, fitting_hyper_image_1],
    #         contributions_1d=[contributions_0, contributions_1], hyper_galaxies=hyper_galaxies)
    #
    #     assert (scaled_noises[0] == np.array([2.5, 2.5, 1.75])).all()
    #     assert (scaled_noises[1] == np.array([2.5, 2.5, 1.75])).all()

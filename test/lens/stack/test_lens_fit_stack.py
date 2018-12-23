import numpy as np
import pytest

from autofit.core import fitting_util
from autolens.data.imaging import ccd as im
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy as g
from autolens.lens.util import lens_fit_util
from autolens.lens.stack import ray_tracing_stack
from autolens.lens.stack import lens_fit_stack
from autolens.lens.stack import lens_image_stack as lis
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from test.mock.mock_profiles import MockLightProfile

@pytest.fixture(name='li_blur_stack')
def make_li_blur_stack():

    image_0 = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
    psf_0 = im.PSF(array=(np.array([[1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0]])), pixel_scale=1.0, renormalize=False)
    image_0 = im.CCD(image_0, pixel_scale=1.0, psf=psf_0, noise_map=np.ones((4, 4)))

    mask_0 = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    mask_0 = msk.Mask(array=mask_0, pixel_scale=1.0)
    
    image_1 = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
    psf_1 = im.PSF(array=(np.array([[1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0]])), pixel_scale=1.0, renormalize=False)
    image_1 = im.CCD(image_1, pixel_scale=1.0, psf=psf_1, noise_map=np.ones((4, 4)))

    mask_1 = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    mask_1 = msk.Mask(array=mask_1, pixel_scale=1.0)

    return lis.LensImageStack(images=[image_0, image_1], masks=[mask_0, mask_1], sub_grid_size=1)

@pytest.fixture(name='li_manual_stack')
def make_li_manual_stack():
    
    image_0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 2.0, 3.0, 0.0],
                   [0.0, 4.0, 5.0, 6.0, 0.0],
                   [0.0, 7.0, 8.0, 9.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0]])
    psf_0 = im.PSF(array=(np.array([[1.0, 5.0, 9.0],
                                     [2.0, 5.0, 1.0],
                                     [3.0, 4.0, 0.0]])), pixel_scale=1.0)
    image_0 = im.CCD(image_0, pixel_scale=1.0, psf=psf_0, noise_map=np.ones((5, 5)))
    mask_0 = msk.Mask(array=np.array([[True, True, True, True, True],
                                   [True, False, False, False, True],
                                   [True, False, False, False, True],
                                   [True, False, False, False, True],
                                   [True, True, True, True, True]]), pixel_scale=1.0)

    image_1 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 2.0, 3.0, 0.0],
                   [0.0, 4.0, 6.0, 6.0, 0.0],
                   [0.0, 7.0, 8.0, 9.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0]])
    psf_1 = im.PSF(array=(np.array([[1.0, 1.0, 1.0],
                                     [2.0, 1.0, 1.0],
                                     [3.0, 1.0, 0.0]])), pixel_scale=1.0)
    image_1 = im.CCD(image_1, pixel_scale=1.0, psf=psf_1, noise_map=np.ones((5, 5)))
    mask_1 = msk.Mask(array=np.array([[True, True, True, True, True],
                                   [True, False, False, False, True],
                                   [True, False, False, False, True],
                                   [True, False, False, True, True],
                                   [True, True, True, True, True]]), pixel_scale=1.0)

    return lis.LensImageStack(images=[image_0, image_1], masks=[mask_0, mask_1], sub_grid_size=1)


class TestLensProfileFit:

    class TestLikelihood:

        def test__image__tracing_fits_data_perfectly__no_psf_blurring__lh_is_noise_normalization(self):

            psf_0 = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])), pixel_scale=1.0)
            image_0 = im.CCD(np.ones((3, 3)), pixel_scale=1.0, psf=psf_0, noise_map=np.ones((3, 3)))
            mask_0 = msk.Mask(array=np.array([[True, True, True],
                                           [True, False, True],
                                           [True, True, True]]), pixel_scale=1.0)

            psf_1 = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])), pixel_scale=1.0)
            image_1 = im.CCD(np.ones((3, 3)), pixel_scale=1.0, psf=psf_1, noise_map=np.ones((3, 3)))
            mask_1 = msk.Mask(array=np.array([[True, True, True],
                                           [True, False, True],
                                           [True, True, True]]), pixel_scale=1.0)

            li_stack = lis.LensImageStack(images=[image_0, image_1], masks=[mask_0, mask_1], sub_grid_size=1)

            g0 = g.Galaxy(light_profile=MockLightProfile(value=1.0))
            tracer = ray_tracing_stack.TracerImagePlaneStack(lens_galaxies=[g0],
                                                             image_plane_grid_stacks=li_stack.grid_stacks)

            fit = lens_fit_stack.LensProfileFitStack(lens_image_stack=li_stack, tracer=tracer)
            assert fit.likelihood == 2.0 * -0.5 * np.log(2 * np.pi * 1.0)

        def test__1x2_image__tracing_fits_data_with_chi_sq_5_and_chi_sq_4(self):

            psf_0 = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])), pixel_scale=1.0)
            image_0 = im.CCD(5.0 * np.ones((3, 4)), pixel_scale=1.0, psf=psf_0, noise_map=np.ones((3, 4)))
            image_0[1,2]  = 4.0
            mask_0 = msk.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)

            psf_1 = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])), pixel_scale=1.0)
            image_1 = im.CCD(5.0 * np.ones((3, 4)), pixel_scale=1.0, psf=psf_1, noise_map=np.ones((3, 4)))
            mask_1 = msk.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)

            li_stack = lis.LensImageStack(images=[image_0, image_1], masks=[mask_0, mask_1], sub_grid_size=1)

            # Setup as a ray trace instance, using a light profile for the lens

            g0 = g.Galaxy(light_profile=MockLightProfile(value=1.0, size=2))
            tracer = ray_tracing_stack.TracerImagePlaneStack(lens_galaxies=[g0],
                                                             image_plane_grid_stacks=li_stack.grid_stacks)

            fit = lens_fit_stack.LensProfileFitStack(lens_image_stack=li_stack, tracer=tracer)

            assert fit.chi_squared == 25.0 + 32.0
            assert fit.reduced_chi_squared == (25.0 + 32.0) / 2.0
            assert fit.likelihood == -0.5 * ((25.0 + 2.0*np.log(2 * np.pi * 1.0)) + (32.0 + 2.0*np.log(2 * np.pi * 1.0)))

    class TestCompareToManual:

        def test___manual_image_and_psf(self, li_manual_stack):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            tracer = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[g0, g1], source_galaxies=[g0],
                     image_plane_grid_stacks=li_manual_stack.grid_stacks)

            padded_tracer = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[g0, g1], source_galaxies=[g0],
                                                         image_plane_grid_stacks=li_manual_stack.padded_grid_stacks)

            fit = lens_fit_stack.fit_lens_image_stack_with_tracer(lens_image_stack=li_manual_stack,
                                                                  tracer=tracer, padded_tracer=padded_tracer)

            assert li_manual_stack.noise_maps[0] == pytest.approx(fit.noise_maps[0], 1e-4)
            assert li_manual_stack.noise_maps[1] == pytest.approx(fit.noise_maps[1], 1e-4)

            model_image_1d_0 = lens_fit_util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
                               unblurred_image_1d=tracer.image_plane_images_1d[0],
                               blurring_image_1d=tracer.image_plane_blurring_images_1d[0],
                               convolver=li_manual_stack.convolvers_image[0])

            model_image_1d_1 = lens_fit_util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
                               unblurred_image_1d=tracer.image_plane_images_1d[1],
                               blurring_image_1d=tracer.image_plane_blurring_images_1d[1],
                               convolver=li_manual_stack.convolvers_image[1])

            model_image_0 = li_manual_stack.map_to_scaled_arrays[0](array_1d=model_image_1d_0)
            model_image_1 = li_manual_stack.map_to_scaled_arrays[1](array_1d=model_image_1d_1)

            assert model_image_0 == pytest.approx(fit.model_images[0], 1e-4)
            assert model_image_1 == pytest.approx(fit.model_images[1], 1e-4)

            residual_map_0 = fitting_util.residual_map_from_data_mask_and_model_data(data=li_manual_stack.images[0],
                                                            mask=li_manual_stack.masks[0], model_data=model_image_0)
            residual_map_1 = fitting_util.residual_map_from_data_mask_and_model_data(data=li_manual_stack.images[1],
                                                            mask=li_manual_stack.masks[1], model_data=model_image_1)

            assert residual_map_0 == pytest.approx(fit.residual_maps[0], 1e-4)
            assert residual_map_1 == pytest.approx(fit.residual_maps[1], 1e-4)

            chi_squared_map_0 = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(
                residual_map=residual_map_0, mask=li_manual_stack.masks[0], noise_map=li_manual_stack.noise_maps[0])
            chi_squared_map_1 = fitting_util.chi_squared_map_from_residual_map_noise_map_and_mask(
                residual_map=residual_map_1, mask=li_manual_stack.masks[1], noise_map=li_manual_stack.noise_maps[1])

            assert chi_squared_map_0 == pytest.approx(fit.chi_squared_maps[0], 1e-4)
            assert chi_squared_map_1 == pytest.approx(fit.chi_squared_maps[1], 1e-4)

            chi_squared_0 = fitting_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map_0,
                                                                                 mask=li_manual_stack.masks[0])
            noise_normalization_0 = fitting_util.noise_normalization_from_noise_map_and_mask(noise_map=li_manual_stack.noise_maps[0],
                                                                                           mask=li_manual_stack.masks[0])
            likelihood_0 = fitting_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared_0,
                                                                                          noise_normalization=noise_normalization_0)

            chi_squared_1 = fitting_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map_1,
                                                                                 mask=li_manual_stack.masks[1])
            noise_normalization_1 = fitting_util.noise_normalization_from_noise_map_and_mask(noise_map=li_manual_stack.noise_maps[1],
                                                                                           mask=li_manual_stack.masks[1])
            likelihood_1 = fitting_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared_1,
                                                                                          noise_normalization=noise_normalization_1)

            assert fit.chi_squareds == [chi_squared_0, chi_squared_1]
            assert fit.noise_normalizations == [noise_normalization_0, noise_normalization_1]
            assert fit.likelihoods == [likelihood_0, likelihood_1]

            assert likelihood_0 + likelihood_1 == pytest.approx(fit.likelihood, 1e-4)
            assert likelihood_0 + likelihood_1 == fit.figure_of_merit
            blurred_image_of_planes_0 = lens_fit_util.blurred_image_of_planes_from_1d_images_and_convolver(
                total_planes=tracer.total_planes,
                image_plane_image_1d_of_planes=tracer.image_plane_images_1d_of_planes[0],
                image_plane_blurring_image_1d_of_planes=tracer.image_plane_blurring_images_1d_of_planes[0],
                convolver=li_manual_stack.convolvers_image[0],
                map_to_scaled_array=li_manual_stack.map_to_scaled_arrays[0])

            blurred_image_of_planes_1 = lens_fit_util.blurred_image_of_planes_from_1d_images_and_convolver(
                total_planes=tracer.total_planes,
                image_plane_image_1d_of_planes=tracer.image_plane_images_1d_of_planes[1],
                image_plane_blurring_image_1d_of_planes=tracer.image_plane_blurring_images_1d_of_planes[1],
                convolver=li_manual_stack.convolvers_image[1],
                map_to_scaled_array=li_manual_stack.map_to_scaled_arrays[1])

            assert (blurred_image_of_planes_0[0] == fit.model_images_of_planes[0][0]).all()
            assert (blurred_image_of_planes_0[1] == fit.model_images_of_planes[0][1]).all()
            assert (blurred_image_of_planes_1[0] == fit.model_images_of_planes[1][0]).all()
            assert (blurred_image_of_planes_1[1] == fit.model_images_of_planes[1][1]).all()

            unmasked_blurred_image_0 = \
                lens_fit_util.unmasked_blurred_image_from_padded_grid_stack_psf_and_unmasked_image(
                padded_grid_stack=li_manual_stack.padded_grid_stacks[0], psf=li_manual_stack.psfs[0],
                    unmasked_image_1d=padded_tracer.image_plane_images_1d[0])

            unmasked_blurred_image_1 = \
                lens_fit_util.unmasked_blurred_image_from_padded_grid_stack_psf_and_unmasked_image(
                padded_grid_stack=li_manual_stack.padded_grid_stacks[1], psf=li_manual_stack.psfs[1],
                    unmasked_image_1d=padded_tracer.image_plane_images_1d[1])

            assert (unmasked_blurred_image_0 == fit.unmasked_model_images[0]).all()
            assert (unmasked_blurred_image_1 == fit.unmasked_model_images[1]).all()

            unmasked_blurred_image_of_galaxies_i0 = \
                lens_fit_util.unmasked_blurred_images_of_galaxies_from_psf_and_unmasked_1d_galaxy_images(
                    galaxies=padded_tracer.image_plane.galaxies,
                    image_plane_image_1d_of_galaxies=padded_tracer.image_plane.image_plane_images_1d_of_galaxies[0],
                    padded_grid_stack=li_manual_stack.padded_grid_stacks[0], psf=li_manual_stack.psfs[0])

            unmasked_blurred_image_of_galaxies_s0 = \
                lens_fit_util.unmasked_blurred_images_of_galaxies_from_psf_and_unmasked_1d_galaxy_images(
                    galaxies=padded_tracer.source_plane.galaxies,
                    image_plane_image_1d_of_galaxies=padded_tracer.source_plane.image_plane_images_1d_of_galaxies[0],
                    padded_grid_stack=li_manual_stack.padded_grid_stacks[0], psf=li_manual_stack.psfs[0])

            unmasked_blurred_image_of_galaxies_i1 = \
                lens_fit_util.unmasked_blurred_images_of_galaxies_from_psf_and_unmasked_1d_galaxy_images(
                    galaxies=padded_tracer.image_plane.galaxies,
                    image_plane_image_1d_of_galaxies=padded_tracer.image_plane.image_plane_images_1d_of_galaxies[0],
                    padded_grid_stack=li_manual_stack.padded_grid_stacks[1], psf=li_manual_stack.psfs[1])

            unmasked_blurred_image_of_galaxies_s1 = \
                lens_fit_util.unmasked_blurred_images_of_galaxies_from_psf_and_unmasked_1d_galaxy_images(
                    galaxies=padded_tracer.source_plane.galaxies,
                    image_plane_image_1d_of_galaxies=padded_tracer.source_plane.image_plane_images_1d_of_galaxies[0],
                    padded_grid_stack=li_manual_stack.padded_grid_stacks[1], psf=li_manual_stack.psfs[1])

            assert (unmasked_blurred_image_of_galaxies_i0[0] ==
                    fit.unmasked_model_images_of_planes_and_galaxies[0][0][0]).all()
            assert (unmasked_blurred_image_of_galaxies_s0[0] ==
                    fit.unmasked_model_images_of_planes_and_galaxies[0][1][0]).all()
            assert (unmasked_blurred_image_of_galaxies_i1[0] ==
                    fit.unmasked_model_images_of_planes_and_galaxies[1][0][0]).all()
            assert (unmasked_blurred_image_of_galaxies_s1[0] ==
                    fit.unmasked_model_images_of_planes_and_galaxies[1][1][0]).all()
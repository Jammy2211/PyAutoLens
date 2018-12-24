import numpy as np
import pytest

from autofit.core import fit_util
from autolens.data.ccd import ccd as im
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy as g
from autolens.lens.util import lens_fit_util as util
from autolens.lens import ray_tracing, lens_fit
from autolens.lens import lens_image as li
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations
from autolens.model.inversion import regularization
from autolens.model.inversion import inversions
from test.mock.mock_profiles import MockLightProfile
from test.mock.mock_lens import MockTracer
from test.mock.mock_galaxy import MockHyperGalaxy

@pytest.fixture(name='li_blur')
def make_li_blur():

    image = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
    psf = im.PSF(array=(np.array([[1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0]])), pixel_scale=1.0, renormalize=False)
    ccd = im.CCD(image, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

    mask = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    mask = msk.Mask(array=mask, pixel_scale=1.0)

    return li.LensImage(ccd, mask, sub_grid_size=1)

@pytest.fixture(name='li_manual')
def make_li_manual():
    image = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 2.0, 3.0, 0.0],
                   [0.0, 4.0, 5.0, 6.0, 0.0],
                   [0.0, 7.0, 8.0, 9.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0]])
    psf = im.PSF(array=(np.array([[1.0, 5.0, 9.0],
                                     [2.0, 5.0, 1.0],
                                     [3.0, 4.0, 0.0]])), pixel_scale=1.0)
    image = im.CCD(image, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
    mask = msk.Mask(array=np.array([[True, True, True, True, True],
                                   [True, False, False, False, True],
                                   [True, False, False, False, True],
                                   [True, False, False, False, True],
                                   [True, True, True, True, True]]), pixel_scale=1.0)

    return li.LensImage(image, mask, sub_grid_size=1)

@pytest.fixture(name='hyper')
def make_hyper():
    class Hyper(object):

        def __init__(self):
            pass

    hyper = Hyper()

    hyper.hyper_model_image = np.array([[1.0, 3.0, 5.0, 7.0],
                                        [7.0, 9.0, 8.0, 1.0],
                                        [6.0, 4.0, 0.0, 9.0],
                                        [3.0, 4.0, 5.0, 6.0]])
    hyper.hyper_galaxy_images = [np.array([[1.0, 3.0, 5.0, 4.0],
                                           [7.0, 9.0, 8.0, 9.0],
                                           [6.0, 4.0, 0.0, 3.0],
                                           [6.0, 2.0, 3.0, 2.0]]),
                                 np.array([[1.0, 3.0, 5.0, 1.0],
                                           [7.0, 9.0, 8.0, 2.0],
                                           [6.0, 4.0, 0.0, 3.0],
                                           [1.0, 3.0, 4.0, 1.0]])]
    hyper.hyper_minimum_values = [0.2, 0.8]
    return hyper

@pytest.fixture(name='li_hyper_no_blur')
def make_li_hyper_no_blur(hyper):

    image = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])

    psf = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]])), pixel_scale=1.0, renormalize=False)
    ccd = im.CCD(image, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

    mask = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    mask = msk.Mask(array=mask, pixel_scale=1.0)

    return li.LensHyperImage(ccd, mask, hyper_model_image=hyper.hyper_model_image,
                        hyper_galaxy_images=hyper.hyper_galaxy_images,
                        hyper_minimum_values=hyper.hyper_minimum_values, sub_grid_size=1)

@pytest.fixture(name='li_hyper_manual')
def make_li_hyper_manual(hyper):
    image = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 2.0, 3.0, 0.0],
                   [0.0, 4.0, 5.0, 6.0, 0.0],
                   [0.0, 7.0, 8.0, 9.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0]])
    psf = im.PSF(array=(np.array([[1.0, 5.0, 9.0],
                                     [2.0, 5.0, 1.0],
                                     [3.0, 4.0, 0.0]])), pixel_scale=1.0)
    ccd = im.CCD(image, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
    mask = msk.Mask(array=np.array([[True, True, True, True, True],
                                   [True, False, False, False, True],
                                   [True, False, False, False, True],
                                   [True, False, False, False, True],
                                   [True, True, True, True, True]]), pixel_scale=1.0)

    return li.LensHyperImage(ccd, mask, hyper_model_image=hyper.hyper_model_image,
                        hyper_galaxy_images=hyper.hyper_galaxy_images,
                        hyper_minimum_values=hyper.hyper_minimum_values, sub_grid_size=1)


class TestAbstractLensFit:

    class TestAbstractLogic:

        def test__logic_in_abstract_fit(self, li_manual):

            galaxy_light = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light],
                                                  image_plane_grid_stack=li_manual.grid_stack)

            fit = lens_fit.AbstractLensFit(tracer=tracer, padded_tracer=None,
                                           map_to_scaled_array=li_manual.map_to_scaled_array)

            assert fit.total_inversions == 0

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g.Galaxy(pixelization=pixelizations.Rectangular(),
                                                                          regularization=regularization.Constant())],
                                                  image_plane_grid_stack=li_manual.grid_stack)

            fit = lens_fit.AbstractLensFit(tracer=tracer, padded_tracer=None,
                                           map_to_scaled_array=li_manual.map_to_scaled_array)

            assert fit.total_inversions == 1


class TestAbstractLensProfileFit:

    class TestBlurredImage:

        def test__mock_tracer__2x2_image_all_1s__3x3_psf_all_1s__blurring_region__image_blurs_to_9s(self, li_blur):

            tracer = MockTracer(unblurred_image_1d=li_blur.mask.map_2d_array_to_masked_1d_array(li_blur.image),
                                blurring_image_1d=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                                has_light_profile=True, has_hyper_galaxy=False, has_pixelization=False)

            fit = lens_fit.AbstractLensProfileFit(lens_image=li_blur, tracer=tracer, padded_tracer=None)

            assert (fit.blurred_profile_image == np.array([[0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 9.0, 9.0, 0.0],
                                                           [0.0, 9.0, 9.0, 0.0],
                                                           [0.0, 0.0, 0.0, 0.0]])).all()

    class TestBlurredImageOfGalaxies:

        def test__padded_tracer_is_none__mode_profie_images_return_none(self, li_blur):

            g0 = g.Galaxy(light_profile=MockLightProfile(value=1.0))
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0], image_plane_grid_stack=li_blur.grid_stack)

            fit = lens_fit.AbstractLensProfileFit(lens_image=li_blur, tracer=tracer, padded_tracer=None)

            assert fit.unmasked_model_image == None
            assert fit.unmasked_model_image_of_planes_and_galaxies == None


class TestAbstractLensInversionFit:

    class TestModelImageOfPlanes:

        def test__model_images_are_none_and_an_image(self):

            image = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]])

            mask = np.array([[True, True, True, True, True],
                           [True, False, False, False, True],
                           [True, False, False, False, True],
                           [True, False, False, False, True],
                           [True, True, True, True, True]])

            mask = msk.Mask(mask, pixel_scale=1.0)

            psf = im.PSF(array=np.array([[0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0]]), pixel_scale=1.0)
            ccd = im.CCD(image=image, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
            lens_image = li.LensImage(ccd=ccd, mask=mask, sub_grid_size=2)

            galaxy_pix = g.Galaxy(pixelization=pixelizations.Rectangular(shape=(3, 3)),
                                  regularization=regularization.Constant(coefficients=(1.0,)))
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[galaxy_pix],
                                                         image_plane_grid_stack=lens_image.grid_stack, border=None)

            fit = lens_fit.AbstractLensInversionFit(lens_image=lens_image, noise_map_1d=lens_image.noise_map_1d,
                                                    tracer=tracer)

            assert fit.model_images_of_planes[0] == None
            assert fit.model_images_of_planes[1] == pytest.approx(np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                                            [0.0, 1.0, 1.0, 1.0, 0.0],
                                                                            [0.0, 1.0, 1.0, 1.0, 0.0],
                                                                            [0.0, 1.0, 1.0, 1.0, 0.0],
                                                                            [0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-2)


class TestAbstractLensProfileInversionFit:

    class TestModelImagesOfPlanes:

        def test___model_images_of_planes_are_profile_and_inversion_images(self, li_manual):

            galaxy_light = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))

            pix = pixelizations.Rectangular(shape=(3, 3))
            reg = regularization.Constant(coefficients=(1.0,))
            galaxy_pix = g.Galaxy(pixelization=pix, regularization=reg)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_light], source_galaxies=[galaxy_pix],
                                                         image_plane_grid_stack=li_manual.grid_stack, border=None)

            fit = lens_fit.AbstractLensProfileInversionFit(lens_image=li_manual,
                                                           noise_map_1d=li_manual.noise_map_1d,
                                                           tracer=tracer, padded_tracer=None)

            blurred_profile_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
                unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
                convolver=li_manual.convolver_image)

            blurred_profile_image = li_manual.map_to_scaled_array(array_1d=blurred_profile_image_1d)

            assert (fit.model_image_of_planes[0] == blurred_profile_image).all()

            blurred_profile_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
                unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
                convolver=li_manual.convolver_image)

            profile_subtracted_image_1d = li_manual.image_1d - blurred_profile_image_1d

            mapper = pix.mapper_from_grid_stack_and_border(grid_stack=li_manual.grid_stack, border=None)

            inversion = inversions.inversion_from_image_mapper_and_regularization(
                image_1d=profile_subtracted_image_1d, noise_map_1d=li_manual.noise_map_1d,
                convolver=li_manual.convolver_mapping_matrix, mapper=mapper, regularization=reg)

            assert (fit.model_image_of_planes[1] == inversion.reconstructed_data).all()


class TestLensProfileFit:

    class TestLikelihood:

        def test__image__tracing_fits_data_perfectly__no_psf_blurring__lh_is_noise_normalization(self):

            psf = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])), pixel_scale=1.0)

            ccd = im.CCD(image=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

            mask = msk.Mask(array=np.array([[True, True, True],
                                           [True, False, True],
                                           [True, True, True]]), pixel_scale=1.0)
            lens_image = li.LensImage(ccd=ccd, mask=mask, sub_grid_size=1)

            g0 = g.Galaxy(light_profile=MockLightProfile(value=1.0))
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0], image_plane_grid_stack=lens_image.grid_stack)

            fit = lens_fit.LensProfileFit(lens_image=lens_image, tracer=tracer)
            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        def test__1x2_image__tracing_fits_data_with_chi_sq_5(self):

            psf = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])), pixel_scale=1.0)

            ccd = im.CCD(5.0 * np.ones((3, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 4)))
            ccd.image[1,2]  = 4.0

            mask = msk.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)

            lens_image = li.LensImage(ccd=ccd, mask=mask, sub_grid_size=1)

            # Setup as a ray trace instance, using a light profile for the lens

            g0 = g.Galaxy(light_profile=MockLightProfile(value=1.0, size=2))
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0], image_plane_grid_stack=lens_image.grid_stack)

            fit = lens_fit.LensProfileFit(lens_image=lens_image, tracer=tracer)

            assert fit.chi_squared == 25.0
            assert fit.reduced_chi_squared == 25.0 / 2.0
            assert fit.likelihood == -0.5 * (25.0 + 2.0*np.log(2 * np.pi * 1.0))

    class TestCompareToManual:

        def test___manual_image_and_psf(self, li_manual):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g0],
                                                         image_plane_grid_stack=li_manual.grid_stack)

            padded_tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g0],
                                                                image_plane_grid_stack=li_manual.padded_grid_stack)

            fit = lens_fit.fit_lens_image_with_tracer(lens_image=li_manual, tracer=tracer,
                                                      padded_tracer=padded_tracer)

            assert li_manual.noise_map == pytest.approx(fit.noise_map, 1e-4)

            model_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
                unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
                convolver=li_manual.convolver_image)

            model_image = li_manual.map_to_scaled_array(array_1d=model_image_1d)

            assert model_image == pytest.approx(fit.model_image, 1e-4)

            residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=li_manual.image,
                           mask=li_manual.mask, model_data=model_image)

            assert residual_map == pytest.approx(fit.residual_map, 1e-4)

            chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                    mask=li_manual.mask, noise_map=li_manual.noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map,
                                                                                 mask=li_manual.mask)
            noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(noise_map=li_manual.noise_map,
                                                                                           mask=li_manual.mask,)
            likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                          noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)
            assert likelihood == fit.figure_of_merit

            blurred_image_of_planes = util.blurred_image_of_planes_from_1d_images_and_convolver(
            total_planes=tracer.total_planes, image_plane_image_1d_of_planes=tracer.image_plane_image_1d_of_planes,
            image_plane_blurring_image_1d_of_planes=tracer.image_plane_blurring_image_1d_of_planes,
            convolver=li_manual.convolver_image, map_to_scaled_array=li_manual.map_to_scaled_array)

            assert (blurred_image_of_planes[0] == fit.model_image_of_planes[0]).all()
            assert (blurred_image_of_planes[1] == fit.model_image_of_planes[1]).all()

            unmasked_blurred_image = \
                util.unmasked_blurred_image_from_padded_grid_stack_psf_and_unmasked_image(
                padded_grid_stack=li_manual.padded_grid_stack, psf=li_manual.psf,
                    unmasked_image_1d=padded_tracer.image_plane_image_1d)

            assert (unmasked_blurred_image == fit.unmasked_model_image).all()

            unmasked_blurred_image_of_galaxies = \
                util.unmasked_blurred_image_of_planes_and_galaxies_from_padded_grid_stack_and_psf(
                    planes=padded_tracer.planes, padded_grid_stack=li_manual.padded_grid_stack, psf=li_manual.psf)

            assert (unmasked_blurred_image_of_galaxies[0][0] == fit.unmasked_model_image_of_planes_and_galaxies[0][0]).all()
            assert (unmasked_blurred_image_of_galaxies[1][0] == fit.unmasked_model_image_of_planes_and_galaxies[1][0]).all()


class TestLensInversionFit:

    class TestCompareToManual:

        def test___manual_image_and_psf(self, li_manual):

            pix = pixelizations.Rectangular(shape=(3, 3))
            reg = regularization.Constant(coefficients=(1.0,))

            g0 = g.Galaxy(pixelization=pix, regularization=reg)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[g0],
                                                         image_plane_grid_stack=li_manual.grid_stack, border=None)

            fit = lens_fit.fit_lens_image_with_tracer(lens_image=li_manual, tracer=tracer)

            mapper = pix.mapper_from_grid_stack_and_border(grid_stack=li_manual.grid_stack, border=None)
            inversion = inversions.inversion_from_image_mapper_and_regularization(mapper=mapper,
                                                                                  regularization=reg, image_1d=li_manual.image_1d, noise_map_1d=li_manual.noise_map_1d,
                                                                                  convolver=li_manual.convolver_mapping_matrix)

            assert inversion.reconstructed_data == pytest.approx(fit.model_image, 1e-4)

            residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=li_manual.image,
                           mask=li_manual.mask, model_data=inversion.reconstructed_data)

            assert residual_map == pytest.approx(fit.residual_map, 1e-4)

            chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                mask=li_manual.mask, noise_map=li_manual.noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map,
                                                                                 mask=li_manual.mask)

            noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(mask=li_manual.mask,
                                                                                  noise_map=li_manual.noise_map)

            likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                          noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

            likelihood_with_regularization = \
                util.likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(
                    chi_squared=chi_squared, regularization_term=inversion.regularization_term,
                    noise_normalization=noise_normalization)

            assert likelihood_with_regularization == pytest.approx(fit.likelihood_with_regularization, 1e-4)

            evidence = util.evidence_from_inversion_terms(chi_squared=chi_squared,
          regularization_term=inversion.regularization_term,
          log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
          log_regularization_term=inversion.log_det_regularization_matrix_term, noise_normalization=noise_normalization)

            assert evidence == fit.evidence
            assert evidence == fit.figure_of_merit


class TestLensProfileInversionFit:


    class TestCompareToManual:

        def test___manual_image_and_psf(self, li_manual):

            galaxy_light = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))

            pix = pixelizations.Rectangular(shape=(3, 3))
            reg = regularization.Constant(coefficients=(1.0,))
            galaxy_pix = g.Galaxy(pixelization=pix, regularization=reg)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_light], source_galaxies=[galaxy_pix],
                                                         image_plane_grid_stack=li_manual.grid_stack, border=None)

            fit = lens_fit.fit_lens_image_with_tracer(lens_image=li_manual, tracer=tracer)

            blurred_profile_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
                unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
                convolver=li_manual.convolver_image)

            blurred_profile_image = li_manual.map_to_scaled_array(array_1d=blurred_profile_image_1d)

            assert blurred_profile_image == pytest.approx(fit.blurred_profile_image, 1e-4)

            profile_subtracted_image = li_manual.image - blurred_profile_image

            assert profile_subtracted_image == pytest.approx(fit.profile_subtracted_image)

            blurred_profile_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
                unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
                convolver=li_manual.convolver_image)

            profile_subtracted_image_1d = li_manual.image_1d - blurred_profile_image_1d

            mapper = pix.mapper_from_grid_stack_and_border(grid_stack=li_manual.grid_stack, border=None)

            inversion = inversions.inversion_from_image_mapper_and_regularization(
                image_1d=profile_subtracted_image_1d, noise_map_1d=li_manual.noise_map_1d,
                convolver=li_manual.convolver_mapping_matrix, mapper=mapper, regularization=reg)

            model_image = blurred_profile_image + inversion.reconstructed_data

            assert model_image == pytest.approx(fit.model_image, 1e-4)

            residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=li_manual.image,
                           mask=li_manual.mask, model_data=model_image)

            assert residual_map == pytest.approx(fit.residual_map, 1e-4)

            chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                                                mask=li_manual.mask, noise_map=li_manual.noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map,
                                                                                 mask=li_manual.mask)
            noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(mask=li_manual.mask,
                                                                                  noise_map=li_manual.noise_map)

            likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                          noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

            likelihood_with_regularization = \
                util.likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(
                    chi_squared=chi_squared, regularization_term=inversion.regularization_term,
                    noise_normalization=noise_normalization)

            assert likelihood_with_regularization == pytest.approx(fit.likelihood_with_regularization, 1e-4)

            evidence = util.evidence_from_inversion_terms(chi_squared=chi_squared,
                                                                  regularization_term=inversion.regularization_term,
                                                                  log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
                                                                  log_regularization_term=inversion.log_det_regularization_matrix_term, noise_normalization=noise_normalization)

            assert evidence == fit.evidence
            assert evidence == fit.figure_of_merit


class TestLensProfileHyperFit:

    class TestLikelihood:

        def test__hyper_galaxy_adds_to_noise_normalization__chi_squared_is_0(self, li_hyper_no_blur):
            # Setup as a ray trace instance, using a light profile for the lens

            g0 = g.Galaxy(light_profile=MockLightProfile(value=1.0, size=4))
            g1 = g.Galaxy(light_profile=MockLightProfile(value=0.0, size=4))

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1],
                                                  image_plane_grid_stack=li_hyper_no_blur.grid_stack)

            li_hyper_no_blur.hyper_model_image_1d = np.array([1.0, 1.0, 1.0, 1.0])
            li_hyper_no_blur.hyper_galaxy_images_1d = [np.array([1.0, 1.0, 1.0, 1.0]),
                                                           np.array([1.0, 1.0, 1.0, 1.0])]

            tracer.image_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=1.0,
                                                                          noise_power=1.0)
            tracer.image_plane.galaxies[1].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=2.0,
                                                                          noise_power=1.0)

            fit = lens_fit.LensProfileHyperFit(lens_hyper_image=li_hyper_no_blur, tracer=tracer)

            chi_squared = 0.0
            noise_normalization = 4.0 * np.log(2 * np.pi * 4.0 ** 2.0)

            assert fit.likelihood == -0.5 * (chi_squared + noise_normalization)

        def test__hyper_galaxy_adds_to_noise_normalization_for_scaled_noise__chi_squared_nonzero(self, li_hyper_no_blur):

            li_hyper_no_blur.image[1:3,1:3] = 2.0

            g0 = g.Galaxy(light_profile=MockLightProfile(value=1.0, size=4))
            g1 = g.Galaxy(light_profile=MockLightProfile(value=0.0, size=4))

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1],
                                                  image_plane_grid_stack=li_hyper_no_blur.grid_stack)

            li_hyper_no_blur.hyper_model_image_1d = np.array([1.0, 1.0, 1.0, 1.0])
            li_hyper_no_blur.hyper_galaxy_images_1d = [np.array([1.0, 1.0, 1.0, 1.0]),
                                                           np.array([1.0, 1.0, 1.0, 1.0])]


            tracer.image_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=1.0,
                                                                          noise_power=1.0)
            tracer.image_plane.galaxies[1].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=2.0,
                                                                          noise_power=1.0)

            fit = lens_fit.LensProfileHyperFit(lens_hyper_image=li_hyper_no_blur, tracer=tracer)

            chi_squared = 4.0 * (1.0 / (4.0)) ** 2.0
            noise_normalization = 4.0 * np.log(2 * np.pi * 4.0 ** 2.0)

            assert fit.likelihood == -0.5 * (chi_squared + noise_normalization)

    class TestCompareToManual:

        def test___manual_image_and_psf(self, li_hyper_manual):

            hyper_galaxy = g.HyperGalaxy(contribution_factor=4.0, noise_factor=2.0, noise_power=3.0)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0), hyper_galaxy=hyper_galaxy)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                         image_plane_grid_stack=li_hyper_manual.grid_stack)

            padded_tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                               image_plane_grid_stack=li_hyper_manual.padded_grid_stack)

            fit = lens_fit.fit_lens_image_with_tracer(lens_image=li_hyper_manual, tracer=tracer,
                                                      padded_tracer=padded_tracer)

            contributions_1d = util.contribution_maps_1d_from_hyper_images_and_galaxies(
                hyper_model_image_1d=li_hyper_manual.hyper_model_image_1d,
                hyper_galaxy_images_1d=li_hyper_manual.hyper_galaxy_images_1d,
                hyper_galaxies=tracer.hyper_galaxies, hyper_minimum_values=li_hyper_manual.hyper_minimum_values)

            contribution_maps = list(map(lambda contribution_1d :
                                     li_hyper_manual.map_to_scaled_array(array_1d=contribution_1d),
                                     contributions_1d))

            assert contribution_maps[0] == pytest.approx(fit.contribution_maps[0], 1.0e-4)

            hyper_noise_map_1d = util.scaled_noise_map_from_hyper_galaxies_and_contribution_maps(
                contribution_maps=contributions_1d, hyper_galaxies=tracer.hyper_galaxies,
                noise_map=li_hyper_manual.noise_map_1d)

            hyper_noise_map = li_hyper_manual.map_to_scaled_array(array_1d=hyper_noise_map_1d)

            assert hyper_noise_map == pytest.approx(fit.noise_map, 1.0e-4)

            model_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
                unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
                convolver=li_hyper_manual.convolver_image)

            model_image = li_hyper_manual.map_to_scaled_array(array_1d=model_image_1d)

            assert model_image == pytest.approx(fit.model_image, 1e-4)

            residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=li_hyper_manual.image,
                                                                                   mask=li_hyper_manual.mask,
                                                                                   model_data=model_image)

            assert residual_map == pytest.approx(fit.residual_map, 1e-4)

            chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                                                mask=li_hyper_manual.mask, noise_map=hyper_noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map,
                                                                                 mask=li_hyper_manual.mask)
            noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(mask=li_hyper_manual.mask,
                                                                                  noise_map=li_hyper_manual.noise_map)
            likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                          noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.figure_of_merit, 1e-4)
            assert likelihood == pytest.approx(fit.figure_of_merit, 1e-4)

            blurred_image_of_planes = util.blurred_image_of_planes_from_1d_images_and_convolver(
            total_planes=tracer.total_planes, image_plane_image_1d_of_planes=tracer.image_plane_image_1d_of_planes,
            image_plane_blurring_image_1d_of_planes=tracer.image_plane_blurring_image_1d_of_planes,
            convolver=li_hyper_manual.convolver_image,
            map_to_scaled_array=li_hyper_manual.map_to_scaled_array)

            assert (blurred_image_of_planes[0] == fit.model_image_of_planes[0]).all()
            assert (blurred_image_of_planes[1] == fit.model_image_of_planes[1]).all()

            unmasked_blurred_image = \
                util.unmasked_blurred_image_from_padded_grid_stack_psf_and_unmasked_image(
                padded_grid_stack=li_hyper_manual.padded_grid_stack, psf=li_hyper_manual.psf,
                    unmasked_image_1d=padded_tracer.image_plane_image_1d)

            assert (unmasked_blurred_image == fit.unmasked_model_image).all()

            unmasked_blurred_image_of_galaxies = \
                util.unmasked_blurred_image_of_planes_and_galaxies_from_padded_grid_stack_and_psf(
                    planes=padded_tracer.planes, padded_grid_stack=li_hyper_manual.padded_grid_stack,
                    psf=li_hyper_manual.psf)

            assert (unmasked_blurred_image_of_galaxies[0][0] == fit.unmasked_model_image_of_planes_and_galaxies[0][0]).all()
            assert (unmasked_blurred_image_of_galaxies[1][0] == fit.unmasked_model_image_of_planes_and_galaxies[1][0]).all()


class TestLensInversionHyperFit:

    class TestCompareToManual:

        def test___manual_image_and_psf(self, li_hyper_manual):

            pix = pixelizations.Rectangular(shape=(3, 3))
            mapper = pix.mapper_from_grid_stack_and_border(grid_stack=li_hyper_manual.grid_stack,
                                                           border=li_hyper_manual.border)
            reg = regularization.Constant(coefficients=(1.0,))

            hyper_galaxy = g.HyperGalaxy(contribution_factor=4.0, noise_factor=2.0, noise_power=3.0)
            hyp_galaxy = g.Galaxy(hyper_galaxy=hyper_galaxy)
            inv_galaxy = g.Galaxy(pixelization=pix, regularization=reg)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[hyp_galaxy, hyp_galaxy],
                                                         source_galaxies=[inv_galaxy],
                                                         image_plane_grid_stack=li_hyper_manual.grid_stack,
                                                         border=None)

            fit = lens_fit.fit_lens_image_with_tracer(lens_image=li_hyper_manual, tracer=tracer)

            contributions_1d = util.contribution_maps_1d_from_hyper_images_and_galaxies(
                hyper_model_image_1d=li_hyper_manual.hyper_model_image_1d,
                hyper_galaxy_images_1d=li_hyper_manual.hyper_galaxy_images_1d,
                hyper_galaxies=tracer.hyper_galaxies, hyper_minimum_values=li_hyper_manual.hyper_minimum_values)

            contribution_maps = list(map(lambda contribution_1d:
                                     li_hyper_manual.map_to_scaled_array(array_1d=contribution_1d),
                                     contributions_1d))

            assert contribution_maps[0] == pytest.approx(fit.contribution_maps[0], 1.0e-4)

            hyper_noise_map_1d = util.scaled_noise_map_from_hyper_galaxies_and_contribution_maps(
                contribution_maps=contributions_1d, hyper_galaxies=tracer.hyper_galaxies,
                noise_map=li_hyper_manual.noise_map_1d)

            hyper_noise_map = li_hyper_manual.map_to_scaled_array(array_1d=hyper_noise_map_1d)

            assert hyper_noise_map == pytest.approx(fit.noise_map, 1.0e-4)

            inversion = inversions.inversion_from_image_mapper_and_regularization(mapper=mapper,
                                                                                  regularization=reg, image_1d=li_hyper_manual.image_1d, noise_map_1d=hyper_noise_map_1d,
                                                                                  convolver=li_hyper_manual.convolver_mapping_matrix)

            assert inversion.reconstructed_data == pytest.approx(fit.model_image, 1e-4)

            residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=li_hyper_manual.image,
                           mask=li_hyper_manual.mask, model_data=inversion.reconstructed_data)

            assert residual_map == pytest.approx(fit.residual_map, 1e-4)

            chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                                                mask=li_hyper_manual.mask, noise_map=hyper_noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map,
                                                                                 mask=li_hyper_manual.mask)
            noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(mask=li_hyper_manual.mask,
                                                                                  noise_map=hyper_noise_map)

            likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                          noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

            likelihood_with_regularization = \
                util.likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(
                    chi_squared=chi_squared, regularization_term=inversion.regularization_term,
                    noise_normalization=noise_normalization)

            assert likelihood_with_regularization == pytest.approx(fit.likelihood_with_regularization, 1e-4)

            evidence = util.evidence_from_inversion_terms(chi_squared=chi_squared,
                                                                  regularization_term=inversion.regularization_term,
                                                                  log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
                                                                  log_regularization_term=inversion.log_det_regularization_matrix_term, noise_normalization=noise_normalization)

            assert evidence == fit.evidence
            assert evidence == fit.figure_of_merit


class TestLensProfileInversionHyperFit:

    class TestCompareToManual:

        def test___manual_image_and_psf(self, li_hyper_manual):

            pix = pixelizations.Rectangular(shape=(3, 3))
            mapper = pix.mapper_from_grid_stack_and_border(grid_stack=li_hyper_manual.grid_stack,
                                                           border=li_hyper_manual.border)
            reg = regularization.Constant(coefficients=(1.0,))

            hyper_galaxy = g.HyperGalaxy(contribution_factor=4.0, noise_factor=2.0, noise_power=3.0)
            hyp_galaxy = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0), hyper_galaxy=hyper_galaxy)
            inv_galaxy = g.Galaxy(pixelization=pix, regularization=reg)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[hyp_galaxy, hyp_galaxy],
                                                         source_galaxies=[inv_galaxy],
                                                         image_plane_grid_stack=li_hyper_manual.grid_stack,
                                                         border=None)

            fit = lens_fit.fit_lens_image_with_tracer(lens_image=li_hyper_manual, tracer=tracer)

            contributions_1d = util.contribution_maps_1d_from_hyper_images_and_galaxies(
                hyper_model_image_1d=li_hyper_manual.hyper_model_image_1d,
                hyper_galaxy_images_1d=li_hyper_manual.hyper_galaxy_images_1d,
                hyper_galaxies=tracer.hyper_galaxies, hyper_minimum_values=li_hyper_manual.hyper_minimum_values)

            contribution_maps = list(map(lambda contribution_1d:
                                     li_hyper_manual.map_to_scaled_array(array_1d=contribution_1d),
                                     contributions_1d))

            assert contribution_maps[0] == pytest.approx(fit.contribution_maps[0], 1.0e-4)

            hyper_noise_map_1d = util.scaled_noise_map_from_hyper_galaxies_and_contribution_maps(
                contribution_maps=contributions_1d, hyper_galaxies=tracer.hyper_galaxies,
                noise_map=li_hyper_manual.noise_map_1d)

            hyper_noise_map = li_hyper_manual.map_to_scaled_array(array_1d=hyper_noise_map_1d)

            assert hyper_noise_map == pytest.approx(fit.noise_map, 1.0e-4)

            blurred_profile_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
                unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
                convolver=li_hyper_manual.convolver_image)

            blurred_profile_image = li_hyper_manual.map_to_scaled_array(array_1d=blurred_profile_image_1d)

            assert blurred_profile_image == pytest.approx(fit.blurred_profile_image, 1e-4)

            profile_subtracted_image = li_hyper_manual.image - blurred_profile_image

            assert profile_subtracted_image == pytest.approx(fit.profile_subtracted_image)

            blurred_profile_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
                unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
                convolver=li_hyper_manual.convolver_image)

            profile_subtracted_image_1d = li_hyper_manual.image_1d - blurred_profile_image_1d

            inversion = inversions.inversion_from_image_mapper_and_regularization(
                image_1d=profile_subtracted_image_1d, noise_map_1d=hyper_noise_map_1d,
                convolver=li_hyper_manual.convolver_mapping_matrix, mapper=mapper, regularization=reg)

            model_image = blurred_profile_image + inversion.reconstructed_data

            assert model_image == pytest.approx(fit.model_image, 1e-4)

            residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=li_hyper_manual.image,
                                                                                   mask=li_hyper_manual.mask,
                                                                                   model_data=model_image)

            assert residual_map == pytest.approx(fit.residual_map, 1e-4)

            chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                                                mask=li_hyper_manual.mask, noise_map=hyper_noise_map)

            assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

            chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map,
                                                                                 mask=li_hyper_manual.mask)
            noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(noise_map=hyper_noise_map,
                                                                                           mask=li_hyper_manual.mask)

            likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                         noise_normalization=noise_normalization)

            assert likelihood == pytest.approx(fit.likelihood, 1e-4)

            likelihood_with_regularization = \
                util.likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(
                    chi_squared=chi_squared, regularization_term=inversion.regularization_term,
                    noise_normalization=noise_normalization)

            assert likelihood_with_regularization == pytest.approx(fit.likelihood_with_regularization, 1e-4)

            evidence = util.evidence_from_inversion_terms(chi_squared=chi_squared,
                                      regularization_term=inversion.regularization_term,
                                      log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
                                      log_regularization_term=inversion.log_det_regularization_matrix_term,
                                      noise_normalization=noise_normalization)

            assert evidence == fit.evidence
            assert evidence == fit.figure_of_merit


class MockTracerPositions:

    def __init__(self, positions, noise=None):
        self.positions = positions
        self.noise = noise


class TestPositionFit:

    def test__x1_positions__mock_position_tracer__maximum_separation_is_correct(self):
        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0]])])
        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == 1.0

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [1.0, 1.0]])])
        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(2)

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [1.0, 3.0]])])
        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(1.0) + np.square(3.0))

        tracer = MockTracerPositions(positions=[np.array([[-2.0, -4.0], [1.0, 3.0]])])
        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(3.0) + np.square(7.0))

        tracer = MockTracerPositions(positions=[np.array([[8.0, 4.0], [-9.0, -4.0]])])
        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(17.0) + np.square(8.0))

    def test_multiple_positions__mock_position_tracer__maximum_separation_is_correct(self):
        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.5]])])
        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == 1.0

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]])])
        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(18)

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])])
        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(18)

        tracer = MockTracerPositions(positions=[np.array([[-2.0, -4.0], [1.0, 3.0], [0.1, 0.1], [-0.1, -0.1],
                                                          [0.3, 0.4], [-0.6, 0.5]])])
        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(3.0) + np.square(7.0))

        tracer = MockTracerPositions(positions=[np.array([[8.0, 4.0], [8.0, 4.0], [-9.0, -4.0]])])
        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.maximum_separations[0] == np.sqrt(np.square(17.0) + np.square(8.0))

    def test_multiple_sets_of_positions__multiple_sets_of_max_distances(self):
        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.5]]),
                                                np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]]),
                                                np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])])

        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=1.0)

        assert fit.maximum_separations[0] == 1.0
        assert fit.maximum_separations[1] == np.sqrt(18)
        assert fit.maximum_separations[2] == np.sqrt(18)

    def test__likelihood__is_sum_of_separations_divided_by_noise(self):
        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.5]]),
                                                np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]]),
                                                np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])])

        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=1.0)
        assert fit.chi_squared_map[0] == 1.0
        assert fit.chi_squared_map[1] == pytest.approx(18.0, 1e-4)
        assert fit.chi_squared_map[2] == pytest.approx(18.0, 1e-4)
        assert fit.figure_of_merit == pytest.approx(-0.5 * (1.0 + 18 + 18), 1e-4)

        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=2.0)
        assert fit.chi_squared_map[0] == (1.0 / 2.0) ** 2.0
        assert fit.chi_squared_map[1] == pytest.approx((np.sqrt(18.0) / 2.0) ** 2.0, 1e-4)
        assert fit.chi_squared_map[2] == pytest.approx((np.sqrt(18.0) / 2.0) ** 2.0, 1e-4)
        assert fit.figure_of_merit == pytest.approx(-0.5 * ((1.0 / 2.0) ** 2.0 + (np.sqrt(18.0) / 2.0) ** 2.0 +
                                                            (np.sqrt(18.0) / 2.0) ** 2.0), 1e-4)

    def test__threshold__if_not_met_returns_ray_tracing_exception(self):
        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0]])])
        fit = lens_fit.LensPositionFit(positions=tracer.positions, noise_map=1.0)

        assert fit.maximum_separation_within_threshold(threshold=100.0) == True
        assert fit.maximum_separation_within_threshold(threshold=0.1) == False

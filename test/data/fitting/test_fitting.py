import numpy as np
import pytest

from autolens.data.imaging import image as im
from autolens.data.array import mask as msk
from autolens.data.fitting import fitting_data
from autolens.data.fitting import fitting
from autolens.model.galaxy import galaxy as g
from test.mock.mock_galaxy import MockHyperGalaxy

@pytest.fixture(name='fi_no_blur')
def make_fi_no_blur():

    image = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0, 0.0],
                      [0.0, 1.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]])

    psf = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]])), pixel_scale=1.0, renormalize=False)
    image = im.Image(image, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = msk.Mask(array=ma, pixel_scale=1.0)

    return fitting_data.FittingImage(image=image, mask=ma, sub_grid_size=2)


@pytest.fixture(name='fi_blur')
def make_fi_blur():

    image = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])

    psf = im.PSF(array=(np.array([[1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0]])), pixel_scale=1.0, renormalize=False)
    image = im.Image(image, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = msk.Mask(array=ma, pixel_scale=1.0)

    return lensing_image.LensingImage(image, ma, sub_grid_size=2)


class TestAbstractFit:

    def test__image_and_model_are_identical__check_values(self, fi_no_blur):

        fit = fitting.AbstractFit(fitting_datas=[fi_no_blur], model_datas_=[np.ones((4))])

        assert (fit.model_datas_[0] == np.ones((4))).all()
        assert (fit.residuals_[0] == np.zeros((4))).all()
        assert (fit.chi_squareds_[0] == np.zeros((4))).all()

    def test___manual_image_and_psf(self, li_manual):
        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                     image_plane_grids=[li_manual.grids])

        padded_tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                            image_plane_grids=[li_manual.padded_grids])

        fit = lensing_fitting.fit_lensing_image_with_tracer(lensing_image=li_manual, tracer=tracer,
                                                            padded_tracer=padded_tracer)

        image_im = tracer.image_plane_images_
        blurring_im = tracer.image_plane_blurring_images_
        model_images = fitting_util.blur_image_including_blurring_region(image_=image_im,
                                                                         blurring_image_=blurring_im,
                                                                         convolver=[li_manual.convolver_image])
        residuals = fitting_util.residual_from_data_and_model_data(data=[li_manual], model_data=model_images)
        chi_squareds = fitting_util.chi_squared_from_residual_and_noise_map(residual=residuals,
                                                                            noise_map=[li_manual.noise_map_])

        assert li_manual.grids.regular.scaled_array_from_array_1d(li_manual.noise_map_) == \
               pytest.approx(fit.noise_map, 1e-4)
        assert li_manual.grids.regular.scaled_array_from_array_1d(model_images[0]) == \
               pytest.approx(fit.model_data, 1e-4)
        assert li_manual.grids.regular.scaled_array_from_array_1d(residuals[0]) == \
               pytest.approx(fit.residual, 1e-4)
        assert li_manual.grids.regular.scaled_array_from_array_1d(chi_squareds[0]) == \
               pytest.approx(fit.chi_squared, 1e-4)

        chi_squared_terms = fitting_util.chi_squared_term_from_chi_squared(chi_squared=chi_squareds)
        noise_terms = fitting_util.noise_term_from_noise_map(noise_map=[li_manual.noise_map_])
        likelihoods = fitting_util.likelihood_from_chi_squared_term_and_noise_term(
            chi_squared_term=chi_squared_terms,
            noise_term=noise_terms)

        assert likelihoods[0] == pytest.approx(fit.likelihood, 1e-4)

        fast_likelihood = lensing_fitting.fast_likelihood_from_lensing_image_and_tracer(lensing_image=li_manual,
                                                                                        tracer=tracer)
        assert fast_likelihood == pytest.approx(fit.likelihood)

        padded_model_image = fitting_util.unmasked_blurred_images_from_fitting_images(fitting_images=[li_manual],
                                                                                      unmasked_images_=padded_tracer.image_plane_images_)

        assert (padded_model_image == fit.unmasked_model_profile_image).all()

        padded_model_image_of_galaxies = \
            lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(lensing_image=li_manual,
                                                                                            tracer=padded_tracer,
                                                                                            image_index=0)

        assert (padded_model_image_of_galaxies[0][0] == fit.unmasked_model_profile_images_of_galaxies[0][0][
            0]).all()
        assert (padded_model_image_of_galaxies[1][0] == fit.unmasked_model_profile_images_of_galaxies[0][1][
            0]).all()

    def test___manual_image_and_psf__x2_image(self, li_manual, li_manual_1):
        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                     image_plane_grids=[li_manual.grids, li_manual_1.grids])

        padded_tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                            image_plane_grids=[li_manual.padded_grids,
                                                                               li_manual_1.padded_grids])

        fit = lensing_fitting.fit_multiple_lensing_images_with_tracer(lensing_images=[li_manual, li_manual_1],
                                                                      tracer=tracer, padded_tracer=padded_tracer)

        image_im = tracer.image_plane_images_
        blurring_im = tracer.image_plane_blurring_images_
        model_images = fitting_util.blur_image_including_blurring_region(image_=image_im,
                                                                         blurring_image_=blurring_im,
                                                                         convolver=[li_manual.convolver_image,
                                                                                    li_manual_1.convolver_image])
        residuals = fitting_util.residual_from_data_and_model_data(data=[li_manual, li_manual_1],
                                                                   model_data=model_images)
        chi_squareds = fitting_util.chi_squared_from_residual_and_noise_map(residual=residuals,
                                                                            noise_map=[li_manual.noise_map_,
                                                                                       li_manual_1.noise_map_])

        assert li_manual.grids.regular.scaled_array_from_array_1d(li_manual.noise_map_) == \
               pytest.approx(fit.noise_maps[0], 1e-4)
        assert li_manual.grids.regular.scaled_array_from_array_1d(model_images[0]) == \
               pytest.approx(fit.model_datas[0], 1e-4)
        assert li_manual.grids.regular.scaled_array_from_array_1d(residuals[0]) == \
               pytest.approx(fit.residuals[0], 1e-4)
        assert li_manual.grids.regular.scaled_array_from_array_1d(chi_squareds[0]) == \
               pytest.approx(fit.chi_squareds[0], 1e-4)

        assert li_manual_1.grids.regular.scaled_array_from_array_1d(li_manual_1.noise_map_) == \
               pytest.approx(fit.noise_maps[1], 1e-4)
        assert li_manual_1.grids.regular.scaled_array_from_array_1d(model_images[1]) == \
               pytest.approx(fit.model_datas[1], 1e-4)
        assert li_manual_1.grids.regular.scaled_array_from_array_1d(residuals[1]) == \
               pytest.approx(fit.residuals[1], 1e-4)
        assert li_manual_1.grids.regular.scaled_array_from_array_1d(chi_squareds[1]) == \
               pytest.approx(fit.chi_squareds[1], 1e-4)

        chi_squared_terms = fitting_util.chi_squared_term_from_chi_squared(chi_squared=chi_squareds)
        noise_terms = fitting_util.noise_term_from_noise_map(
            noise_map=[li_manual.noise_map_, li_manual_1.noise_map_])
        likelihoods = fitting_util.likelihood_from_chi_squared_term_and_noise_term(
            chi_squared_term=chi_squared_terms,
            noise_term=noise_terms)

        assert likelihoods[0] == pytest.approx(fit.likelihoods[0], 1e-4)
        assert likelihoods[1] == pytest.approx(fit.likelihoods[1], 1e-4)
        assert likelihoods[0] + likelihoods[1] == pytest.approx(fit.likelihood, 1e-4)

        fast_likelihood = lensing_fitting.fast_likelihood_from_multiple_lensing_images_and_tracer(
            lensing_images=[li_manual, li_manual_1], tracer=tracer)
        assert fast_likelihood == pytest.approx(fit.likelihood)

        padded_model_images = fitting_util.unmasked_blurred_images_from_fitting_images(
            fitting_images=[li_manual, li_manual_1],
            unmasked_images_=padded_tracer.image_plane_images_)

        assert (padded_model_images[0] == fit.unmasked_model_profile_images[0]).all()
        assert (padded_model_images[1] == fit.unmasked_model_profile_images[1]).all()

        padded_model_image_of_galaxies_0 = \
            lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(lensing_image=li_manual,
                                                                                            tracer=padded_tracer,
                                                                                            image_index=0)

        assert (padded_model_image_of_galaxies_0[0][0] == fit.unmasked_model_profile_images_of_galaxies[0][0][
            0]).all()
        assert (padded_model_image_of_galaxies_0[1][0] == fit.unmasked_model_profile_images_of_galaxies[0][1][
            0]).all()

        padded_model_image_of_galaxies_1 = \
            lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(
                lensing_image=li_manual_1,
                tracer=padded_tracer,
                image_index=1)

        assert (padded_model_image_of_galaxies_1[0][0] == fit.unmasked_model_profile_images_of_galaxies[1][0][
            0]).all()
        assert (padded_model_image_of_galaxies_1[1][0] == fit.unmasked_model_profile_images_of_galaxies[1][1][
            0]).all()
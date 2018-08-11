import numpy as np
import pytest

from autolens.analysis import fitting, ray_tracing, galaxy
from autolens.imaging import mask as mask
from autolens.imaging import masked_image
from autolens.imaging import image
from autolens.imaging import convolution
from autolens.profiles import light_profiles
from autolens.pixelization import reconstruction


class TestResiduals:

    def test__model_mathces_data__residuals_all_0s(self):
        image = 10.0 * np.ones((2, 2))
        model = 10.0 * np.ones((2, 2))

        residuals = fitting.residuals_from_image_and_model(image, model)

        assert (residuals == np.zeros((2, 2))).all()

    def test__model_data_mismatch__residuals_non_0(self):
        image = 10.0 * np.ones((2, 2))
        model = np.array([[11, 10],
                          [9, 8]])

        residuals = fitting.residuals_from_image_and_model(image, model)

        assert (residuals == np.array([[-1, 0],
                                       [1, 2]])).all()


class TestChiSquareds:

    def test__model_mathces_data__chi_sq_all_0s(self):
        image = 10.0 * np.ones((2, 2))
        noise = 4.0 * np.ones((2, 2))
        model = 10.0 * np.ones((2, 2))

        residuals = fitting.residuals_from_image_and_model(image, model)
        chi_squared = fitting.chi_squareds_from_residuals_and_noise(residuals, noise)

        assert (chi_squared == np.zeros((2, 2))).all()

    def test__model_data_mismatch__chi_sq_non_0(self):
        image = 10.0 * np.ones((2, 2))
        noise = 2.0 * np.ones((2, 2))
        model = np.array([[11, 10],
                          [9, 8]])

        residuals = fitting.residuals_from_image_and_model(image, model)
        chi_squared = fitting.chi_squareds_from_residuals_and_noise(residuals, noise)

        assert (chi_squared == (np.array([[1 / 4, 0],
                                          [1 / 4, 1]]))).all()


class TestLikelihood:

    def test__model_matches_data__noise_all_2s__lh_is_noise_term(self):
        im = np.array([10.0, 10.0, 10.0, 10.0])
        noise = np.array([2.0, 2.0, 2.0, 2.0])
        model = np.array([10.0, 10.0, 10.0, 10.0])

        residuals = fitting.residuals_from_image_and_model(im, model)
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, noise)
        chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_term = fitting.noise_term_from_data(noise)
        likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

        chi_sq_term = 0
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(
            2 * np.pi * 4.0)

        assert likelihood == -0.5 * (chi_sq_term + noise_term)

    def test__model_data_mismatch__chi_sq_term_contributes_to_lh(self):
        im = np.array([10.0, 10.0, 10.0, 10.0])
        noise = np.array([2.0, 2.0, 2.0, 2.0])
        model = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting.residuals_from_image_and_model(im, model)
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, noise)
        chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_term = fitting.noise_term_from_data(noise)
        likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

        # chi squared = 0.25, 0, 0.25, 1.0
        # likelihood = -0.5*(0.25+0+0.25+1.0)

        chi_sq_term = 1.5
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(
            2 * np.pi * 4.0)

        assert likelihood == -0.5 * (chi_sq_term + noise_term)

    def test__same_as_above_but_different_noise_in_each_pixel(self):
        im = np.array([10.0, 10.0, 10.0, 10.0])
        noise = np.array([1.0, 2.0, 3.0, 4.0])
        model = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting.residuals_from_image_and_model(im, model)
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, noise)
        chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_term = fitting.noise_term_from_data(noise)
        likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

        # chi squared = (1.0/1.0)**2, (0.0), (-1.0/3.0)**2.0, (2.0/4.0)**2.0

        chi_sq_term = 1.0 + (1.0 / 9.0) + 0.25
        noise_term = np.log(2 * np.pi * 1.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 9.0) + np.log(
            2 * np.pi * 16.0)

        assert likelihood == pytest.approx(-0.5 * (chi_sq_term + noise_term), 1e-4)


class TestContributionsFromHypers:

    def test__x1_hyper_galaxy__model_is_galaxy_image__contributions_all_1(self):
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image = np.array([[1.0, 1.0, 1.0]])

        hyper_galaxy_images = [np.array([[1.0, 1.0, 1.0]])]

        minimum_values = [0.0]

        contributions = fitting.contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images,
                                                                             hyper_galaxies, minimum_values)

        assert (contributions[0] == np.array([[1.0, 1.0, 1.0]])).all()

    def test__x1_hyper_galaxy__model_and_galaxy_image_different_contributions_change(self):
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image = np.array([[0.5, 1.0, 1.5]])

        hyper_galaxy_images = [np.array([[0.5, 1.0, 1.5]])]

        minimum_values = [0.6]

        contributions = fitting.contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images, hyper_galaxies,
                                                                             minimum_values)

        assert (contributions[0] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()

    def test__x2_hyper_galaxy__model_and_galaxy_image_different_contributions_change(self):
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
                          MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image = np.array([[0.5, 1.0, 1.5]])

        hyper_galaxy_images = [np.array([[0.5, 1.0, 1.5]]), np.array([[0.5, 1.0, 1.5]])]

        minimum_values = [0.5, 0.6]

        contributions = fitting.contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images, hyper_galaxies,
                                                                             minimum_values)

        assert (contributions[0] == np.array([[1.0, 1.0, 1.0]])).all()
        assert (contributions[1] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()

    def test__x2_hyper_galaxy__same_as_above_use_real_hyper_galaxy(self):
        hyper_galaxies = [galaxy.HyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
                          galaxy.HyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image = np.array([[0.5, 1.0, 1.5]])

        hyper_galaxy_images = [np.array([[0.5, 1.0, 1.5]]), np.array([[0.5, 1.0, 1.5]])]

        minimum_values = [0.5, 0.6]

        contributions = fitting.contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images, hyper_galaxies,
                                                                             minimum_values)

        assert (contributions[0] == np.array([[1.0, 1.0, 1.0]])).all()
        assert (contributions[1] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()


class TestScaledNoiseFromContributions:

    def test__x1_hyper_galaxy__noise_factor_is_0__scaled_noise_is_input_noise(self):
        contributions = [np.array([1.0, 1.0, 2.0])]
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]
        noise = np.array([1.0, 1.0, 1.0])

        scaled_noise = fitting.scaled_noise_from_hyper_galaxies_and_contributions(contributions, hyper_galaxies, noise)

        assert (scaled_noise == noise).all()

    def test__x1_hyper_galaxy__noise_factor_and_power_are_1__scaled_noise_added_to_input_noise(self):
        contributions = [np.array([1.0, 1.0, 0.5])]
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=1.0)]
        noise = np.array([1.0, 1.0, 1.0])

        scaled_noise = fitting.scaled_noise_from_hyper_galaxies_and_contributions(contributions, hyper_galaxies, noise)

        assert (scaled_noise == np.array([2.0, 2.0, 1.5])).all()

    def test__x1_hyper_galaxy__noise_factor_1_and_power_is_2__scaled_noise_added_to_input_noise(self):
        contributions = [np.array([1.0, 1.0, 0.5])]
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0)]
        noise = np.array([1.0, 1.0, 1.0])

        scaled_noise = fitting.scaled_noise_from_hyper_galaxies_and_contributions(contributions, hyper_galaxies, noise)

        assert (scaled_noise == np.array([2.0, 2.0, 1.25])).all()

    def test__x2_hyper_galaxy__noise_factor_1_and_power_is_2__scaled_noise_added_to_input_noise(self):
        contributions = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
                          MockHyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]
        noise = np.array([1.0, 1.0, 1.0])

        scaled_noise = fitting.scaled_noise_from_hyper_galaxies_and_contributions(contributions, hyper_galaxies, noise)

        assert (scaled_noise == np.array([2.5, 2.5, 1.75])).all()

    def test__x2_hyper_galaxy__same_as_above_but_use_real_hyper_galaxy(self):
        contributions = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
        hyper_galaxies = [galaxy.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
                          galaxy.HyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]
        noise = np.array([1.0, 1.0, 1.0])

        scaled_noise = fitting.scaled_noise_from_hyper_galaxies_and_contributions(contributions, hyper_galaxies, noise)

        assert (scaled_noise == np.array([2.5, 2.5, 1.75])).all()


class TestPixelizationEvidence:

    def test__simple_values(self):

        im = np.array([10.0, 10.0, 10.0, 10.0])
        noise = np.array([2.0, 2.0, 2.0, 2.0])
        model = np.array([10.0, 10.0, 10.0, 10.0])

        solution = np.array([1.0, 1.0, 1.0])

        cov_reg_matrix = np.array([[2.0, -1.0, 0.0],
                                   [-1.0, 2.0, -1.0],
                                   [0.0, -1.0, 2.0]])

        reg_matrix = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])

        pix_fit = reconstruction.Reconstruction(data_vector=None, blurred_mapping=None,
                                                regularization=reg_matrix, covariance=None,
                                                covariance_regularization=cov_reg_matrix, reconstruction=solution)

        evidence = fitting.pixelization_evidence_from_data_model_and_pix(im, noise, model, pix_fit)

        chi_sq_term = 0
        reg_term = 3.0
        log_det_cov_reg = np.log(np.linalg.det(cov_reg_matrix))
        log_det_reg = np.log(np.linalg.det(reg_matrix))
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(
            2 * np.pi * 4.0)

        assert evidence == pytest.approx(-0.5 * (chi_sq_term + reg_term + log_det_cov_reg - log_det_reg + noise_term),
                                         1e-4)

    def test__complicated_values(self):
        im = np.array([10.0, 10.0, 10.0, 10.0])
        noise = np.array([1.0, 2.0, 3.0, 4.0])
        model = np.array([11.0, 10.0, 9.0, 8.0])

        solution = np.array([2.0, 3.0, 5.0])

        cov_reg_matrix = np.array([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]])

        reg_matrix = np.array([[2.0, -1.0, 0.0],
                               [-1.0, 2.0, -1.0],
                               [0.0, -1.0, 2.0]])

        pix_fit = reconstruction.Reconstruction(data_vector=None, blurred_mapping=None,
                                                regularization=reg_matrix, covariance=None,
                                                covariance_regularization=cov_reg_matrix, reconstruction=solution)

        evidence = fitting.pixelization_evidence_from_data_model_and_pix(im, noise, model, pix_fit)

        chi_sq_term = 1.0 + (1.0 / 9.0) + 0.25
        reg_term = 34.0
        log_det_cov_reg = np.log(np.linalg.det(cov_reg_matrix))
        log_det_reg = np.log(np.linalg.det(reg_matrix))
        noise_term = np.log(2 * np.pi * 1.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 9.0) + np.log(
            2 * np.pi * 16.0)

        assert evidence == pytest.approx(-0.5 * (chi_sq_term + reg_term + log_det_cov_reg - log_det_reg + noise_term),
                                         1e-4)

    def test__use_fitting_functions_to_compute_terms(self):
        im = np.array([10.0, 100.0, 0.0, 10.0])
        noise = np.array([1.0, 2.0, 77.0, 4.0])
        model = np.array([11.0, 13.0, 9.0, 8.0])

        solution = np.array([8.0, 7.0, 3.0])

        cov_reg_matrix = np.array([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]])

        reg_matrix = np.array([[2.0, -1.0, 0.0],
                               [-1.0, 2.0, -1.0],
                               [0.0, -1.0, 2.0]])

        pix_fit = reconstruction.Reconstruction(data_vector=None, blurred_mapping=None,
                                                regularization=reg_matrix, covariance=None,
                                                covariance_regularization=cov_reg_matrix, reconstruction=solution)

        evidence = fitting.pixelization_evidence_from_data_model_and_pix(im, noise, model, pix_fit)

        chi_sq_term = fitting.chi_squared_term_from_chi_squareds(im, noise, model)
        reg_term = pix_fit.regularization_term_from_reconstruction()
        log_det_cov_reg = pix_fit.log_determinant_of_matrix_cholesky(pix_fit.covariance_regularization)
        log_det_reg = pix_fit.log_determinant_of_matrix_cholesky(pix_fit.regularization)
        noise_term = fitting.noise_term_from_data(noise)

        assert evidence == pytest.approx(-0.5 * (chi_sq_term + reg_term + log_det_cov_reg - log_det_reg + noise_term),
                                         1e-4)


@pytest.fixture(name="no_galaxies", scope='function')
def make_no_galaxies():
    return [galaxy.Galaxy()]

@pytest.fixture(name="galaxy_light_sersic", scope='function')
def make_galaxy_light_sersic():
    sersic = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                             sersic_index=4.0)
    return galaxy.Galaxy(light_profile=sersic)


@pytest.fixture(name='mi_no_blur')
def make_mi_no_blur():
    im = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
    psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]])))
    im = image.Image(im, pixel_scale=1.0, psf=psf, noise=np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = mask.Mask(array=ma, pixel_scale=1.0)

    return masked_image.MaskedImage(im, ma, sub_grid_size=2)


@pytest.fixture(name='mi_blur')
def make_mi_blur():
    im = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
    psf = image.PSF(array=(np.array([[1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0]])))
    im = image.Image(im, pixel_scale=1.0, psf=psf, noise=np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = mask.Mask(array=ma, pixel_scale=1.0)

    return masked_image.MaskedImage(im, ma, sub_grid_size=2)

@pytest.fixture(name='mi_no_blur_1x1')
def make_mi_no_blur_1x1():
    psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]])))

    im = image.Image(np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise=np.ones((3, 3)))

    ma = mask.Mask(array=np.array([[True, True, True],
                                   [True, False, True],
                                   [True, True, True]]), pixel_scale=1.0)

    return masked_image.MaskedImage(im, ma, sub_grid_size=1)


class MockTracer(object):

    def __init__(self, image, blurring_image):
        self.image = image
        self.blurring_image = blurring_image

    def generate_image_of_galaxy_light_profiles(self):
        return self.image

    def generate_blurring_image_of_galaxy_light_profiles(self):
        return self.blurring_image


# noinspection PyUnusedLocal
class MockLightProfile(light_profiles.LightProfile):

    def intensity_from_grid(self, grid):
        return np.array([self.value])

    def __init__(self, value):
        self.value = value

    def intensity_at_radius(self, radius):
        return self.value

    def intensity_at_coordinates(self, coordinates):
        return self.value


class MockHyperGalaxy(object):

    def __init__(self, contribution_factor=0.0, noise_factor=0.0, noise_power=1.0):
        self.contribution_factor = contribution_factor
        self.noise_factor = noise_factor
        self.noise_power = noise_power

    def contributions_from_preload_images(self, model, galaxy_image, minimum_value):
        contributions = galaxy_image / (model + self.contribution_factor)
        contributions = contributions / np.max(contributions)
        contributions[contributions < minimum_value] = 0.0
        return contributions

    def scaled_noise_from_contributions(self, noise, contributions):
        return self.noise_factor * (noise * contributions) ** self.noise_power


class TestFitter:


    class TestBlurredLightProfileImages:

        def test__mock_tracer__2x2_image_all_1s__3x3__psf_central_1__no_blurring(self, mi_no_blur):

            tracer = MockTracer(image=mi_no_blur.mask.map_to_1d(mi_no_blur.image),
                                blurring_image=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

            fitter = fitting.ProfileFitter(masked_image=mi_no_blur, tracer=tracer)

            assert (fitter.blurred_image == np.array([1.0, 1.0, 1.0, 1.0])).all()

        def test__mock_tracer__2x2_image_all_1s__3x3_psf_all_1s__image_blurs_to_4s(self, mi_blur):

            tracer = MockTracer(image=mi_blur.mask.map_to_1d(mi_blur.image),
                                blurring_image=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

            fitter = fitting.ProfileFitter(masked_image=mi_blur, tracer=tracer)

            assert (fitter.blurred_image == np.array([4.0, 4.0, 4.0, 4.0])).all()

        def test__mock_tracer__2x2_image_all_1s__3x3_psf_central_1__include_blurring_region__still_no_blurring(self,
                                                                                                               mi_no_blur):

            tracer = MockTracer(image=mi_no_blur.mask.map_to_1d(mi_no_blur.image),
                                blurring_image=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

            fitter = fitting.ProfileFitter(masked_image=mi_no_blur, tracer=tracer)

            assert (fitter.blurred_image == np.array([1.0, 1.0, 1.0, 1.0])).all()

        def test__mock_tracer__2x2_image_all_1s__3x3_psf_all_1s__include_blurring_region_image_blur_to_9s(self, mi_blur):

            tracer = MockTracer(image=mi_blur.mask.map_to_1d(mi_blur.image),
                                blurring_image=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

            fitter = fitting.ProfileFitter(masked_image=mi_blur, tracer=tracer)

            assert (fitter.blurred_image == np.array([9.0, 9.0, 9.0, 9.0])).all()

        def test__real_tracer__2x2_image__no_psf_blurring(self, mi_no_blur, galaxy_light_sersic, no_galaxies):

            tracer = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic], source_galaxies=no_galaxies,
                                           image_plane_grids=mi_no_blur.grids)

            fitter = fitting.ProfileFitter(masked_image=mi_no_blur, tracer=tracer)

            tracer_non_blurred_image = tracer.generate_image_of_galaxy_light_profiles()

            assert (tracer_non_blurred_image == fitter.image).all()
            assert (tracer_non_blurred_image == fitter.blurred_image).all()

        def test__real_tracer__2x2_image__psf_is_non_symmetric_producing_l_shape(self, galaxy_light_sersic,
                                                                                 no_galaxies):

            psf = image.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                             [0.0, 2.0, 1.0],
                                             [0.0, 0.0, 0.0]])))
            im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=psf, noise=np.ones((4, 4)))

            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)
            mi = masked_image.MaskedImage(im, ma, sub_grid_size=1)

            tracer = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic], source_galaxies=no_galaxies,
                                           image_plane_grids=mi.grids)

            fitter = fitting.ProfileFitter(masked_image=mi, tracer=tracer)

            # Manually compute result of convolution, which is each central value *2.0 plus its 2 appropriate neighbors

            central_values = tracer.generate_image_of_galaxy_light_profiles()
            blurring_values = tracer.generate_blurring_image_of_galaxy_light_profiles()

            tracer_blurred_image_manual_0 = 2.0 * central_values[0] + 3.0 * central_values[2] + blurring_values[4]
            tracer_blurred_image_manual_1 = 2.0 * central_values[1] + 3.0 * central_values[3] + central_values[0]
            tracer_blurred_image_manual_2 = 2.0 * central_values[2] + 3.0 * blurring_values[9] + blurring_values[6]
            tracer_blurred_image_manual_3 = 2.0 * central_values[3] + 3.0 * blurring_values[10] + central_values[2]

            assert tracer_blurred_image_manual_0 == pytest.approx(fitter.blurred_image[0], 1e-6)
            assert tracer_blurred_image_manual_1 == pytest.approx(fitter.blurred_image[1], 1e-6)
            assert tracer_blurred_image_manual_2 == pytest.approx(fitter.blurred_image[2], 1e-6)
            assert tracer_blurred_image_manual_3 == pytest.approx(fitter.blurred_image[3], 1e-6)


    class TestLikelihood:

        def test__1x1_image__tracing_fits_data_perfectly__no_psf_blurring__lh_is_noise_term(self, no_galaxies):

            psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])))

            im = image.Image(np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise=np.ones((3, 3)))

            ma = mask.Mask(array=np.array([[True, True, True],
                                           [True, False, True],
                                           [True, True, True]]), pixel_scale=1.0)
            mi = masked_image.MaskedImage(im, ma, sub_grid_size=1)

            mock_galaxy = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))
            tracer = ray_tracing.Tracer(lens_galaxies=[mock_galaxy], source_galaxies=no_galaxies,
                                        image_plane_grids=mi.grids)

            fitter = fitting.ProfileFitter(masked_image=mi, tracer=tracer)

            assert fitter.blurred_image_likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        def test___1x1_image__tracing_fits_data_perfectly__psf_blurs_model_to_5__lh_is_chi_sq_plus_noise(self,
                                                                                                         no_galaxies):

            psf = image.PSF(array=(np.array([[0.0, 1.0, 0.0],
                                             [1.0, 1.0, 1.0],
                                             [0.0, 1.0, 0.0]])))

            im = image.Image(np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise=np.ones((3, 3)))

            ma = mask.Mask(array=np.array([[True, True, True],
                                           [True, False, True],
                                           [True, True, True]]), pixel_scale=1.0)

            mi = masked_image.MaskedImage(im, ma, sub_grid_size=1)

            # Setup as a ray trace instance, using a light profile for the lens

            mock_galaxy = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))
            tracer = ray_tracing.Tracer(lens_galaxies=[mock_galaxy], source_galaxies=no_galaxies,
                                           image_plane_grids=mi.grids)

            fitter = fitting.ProfileFitter(masked_image=mi, tracer=tracer)

            assert fitter.blurred_image_likelihood == -0.5 * (16.0 + np.log(2 * np.pi * 1.0))


    class TestCompareToManual:

        def test___random_image_and_psf(self):

            im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 2.0, 3.0, 0.0],
                           [0.0, 4.0, 5.0, 6.0, 0.0],
                           [0.0, 7.0, 8.0, 9.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]])
            psf = image.PSF(array=(np.array([[1.0, 5.0, 9.0],
                                             [2.0, 5.0, 1.0],
                                             [3.0, 4.0, 0.0]])))
            im = image.Image(im, pixel_scale=1.0, psf=psf, noise=np.ones((5, 5)))
            ma = mask.Mask(array=np.array([[True, True,  True,  True,  True],
                                           [True, False, False, False, True],
                                           [True, False, False, False, True],
                                           [True, False, False, False, True],
                                           [True, True,  True,  True,  True]]), pixel_scale=1.0)
            mi = masked_image.MaskedImage(im, ma, sub_grid_size=1)

            mock_galaxy = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersic(intensity=1.0))
            tracer = ray_tracing.Tracer(lens_galaxies=[mock_galaxy], source_galaxies=[mock_galaxy],
                                        image_plane_grids=mi.grids)

            fitter = fitting.ProfileFitter(masked_image=mi, tracer=tracer)

            image_im = tracer.generate_image_of_galaxy_light_profiles()
            blurring_im = tracer.generate_blurring_image_of_galaxy_light_profiles()
            blurred_im = mi.convolver_image.convolve_image_jit(image_im, blurring_im)
            residuals = fitting.residuals_from_image_and_model(mi, blurred_im)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, mi.noise)
            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_data(mi.noise)
            likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

            assert image_im == pytest.approx(fitter.image, 1e-4)
            assert blurring_im == pytest.approx(fitter.blurring_region_image, 1e-4)
            assert blurred_im == pytest.approx(fitter.blurred_image, 1e-4)
            assert residuals == pytest.approx(fitter.blurred_image_residuals, 1e-4)
            assert chi_squareds == pytest.approx(fitter.blurred_image_chi_squareds, 1e-4)
            assert chi_squared_term == pytest.approx(fitter.blurred_image_chi_squared_term, 1e-4)
            assert noise_term == pytest.approx(fitter.noise_term, 1e-4)
            assert likelihood == pytest.approx(fitter.blurred_image_likelihood, 1e-4)


class TestHyperFitter:

    class TestScaledLikelihood:

        def test__hyper_galaxy_adds_to_noise_term_for_scaled_noise__chi_squared_is_0(self, no_galaxies,
                                                                                     mi_no_blur_1x1):
    
            # Setup as a ray trace instance, using a light profile for the lens
    
            mock_galaxy = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))
    
            tracer = ray_tracing.Tracer(lens_galaxies=[mock_galaxy], source_galaxies=no_galaxies,
                                        image_plane_grids=mi_no_blur_1x1.grids)
    
            model = np.array([1.0])
            hyper_galaxy_images = [np.array([1.0]), np.array([1.0])]
    
            tracer.image_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=1.0,
                                                                             noise_power=1.0)
            tracer.source_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=2.0,
                                                                              noise_power=1.0)
    
            fitter = fitting.HyperProfileFitter(masked_image=mi_no_blur_1x1, tracer=tracer, hyper_model_image=model,
                                                hyper_galaxy_images=hyper_galaxy_images, hyper_minimum_values=[0.0, 0.0])
    
            chi_squared_term = 0.0
            scaled_noise_term = np.log(2 * np.pi * 4.0 ** 2.0)
    
            assert fitter.blurred_image_scaled_likelihood == -0.5 * (chi_squared_term + scaled_noise_term)
    
        def test__hyper_galaxy_adds_to_noise_term_for_scaled_noise__chi_squared_nonzero(self, no_galaxies, mi_no_blur_1x1):
    
            mi_no_blur_1x1[0] = 2.0
    
            mock_galaxy = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))
    
            tracer = ray_tracing.Tracer(lens_galaxies=[mock_galaxy], source_galaxies=no_galaxies,
                                        image_plane_grids=mi_no_blur_1x1.grids)
    
            model = np.array([1.0])
            hyper_galaxy_images = [np.array([1.0]), np.array([1.0])]
    
            tracer.image_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=1.0,
                                                                             noise_power=1.0)
            tracer.source_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=2.0,
                                                                              noise_power=1.0)
    
            fitter = fitting.HyperProfileFitter(masked_image=mi_no_blur_1x1, tracer=tracer, hyper_model_image=model,
                                                hyper_galaxy_images=hyper_galaxy_images, hyper_minimum_values=[0.0, 0.0])
    
            scaled_chi_squared_term = (1.0/(4.0))**2.0
            scaled_noise_term = np.log(2 * np.pi * 4.0 ** 2.0)
    
            assert fitter.blurred_image_scaled_likelihood == -0.5 * (scaled_chi_squared_term + scaled_noise_term)

    class TestCompareToManual:

        def test___random_image_and_psf(self):

            im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 2.0, 3.0, 0.0],
                           [0.0, 4.0, 5.0, 6.0, 0.0],
                           [0.0, 7.0, 8.0, 9.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]])
            psf = image.PSF(array=(np.array([[1.0, 5.0, 9.0],
                                             [2.0, 5.0, 1.0],
                                             [3.0, 4.0, 0.0]])))
            im = image.Image(im, pixel_scale=1.0, psf=psf, noise=np.ones((5, 5)))
            ma = mask.Mask(array=np.array([[True, True,  True,  True,  True],
                                           [True, False, False, False, True],
                                           [True, False, False, False, True],
                                           [True, False, False, False, True],
                                           [True, True,  True,  True,  True]]), pixel_scale=1.0)
            mi = masked_image.MaskedImage(im, ma, sub_grid_size=1)

            hyper_model_image = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 8.0, 6.0, 4.0, 0.0])
            hyper_galaxy_images = [np.array([1.0, 3.0, 5.0, 7.0, 9.0, 8.0, 6.0, 4.0, 0.0]),
                                   np.array([1.0, 3.0, 5.0, 7.0, 9.0, 8.0, 6.0, 4.0, 0.0])]
    
            hyper_galaxy = galaxy.HyperGalaxy(contribution_factor=4.0, noise_factor=2.0, noise_power=3.0)
            mock_galaxy = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersic(intensity=1.0), 
                                        hyper_galaxy=hyper_galaxy)
            tracer = ray_tracing.Tracer(lens_galaxies=[mock_galaxy], source_galaxies=[mock_galaxy],
                                        image_plane_grids=mi.grids)

            fitter = fitting.HyperProfileFitter(masked_image=mi, tracer=tracer, hyper_model_image=hyper_model_image, 
                                                hyper_galaxy_images=hyper_galaxy_images, hyper_minimum_values=[0.2, 0.8])

            image_im = tracer.generate_image_of_galaxy_light_profiles()
            blurring_im = tracer.generate_blurring_image_of_galaxy_light_profiles()
            blurred_im = mi.convolver_image.convolve_image_jit(image_im, blurring_im)
            residuals = fitting.residuals_from_image_and_model(mi, blurred_im)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, mi.noise)
            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_data(mi.noise)
            likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)
            contributions = fitting.contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images,
                                                            [hyper_galaxy, hyper_galaxy], minimum_values=[0.2, 0.8])
            scaled_noise = fitting.scaled_noise_from_hyper_galaxies_and_contributions(contributions,
                                                                                      [hyper_galaxy, hyper_galaxy],
                                                                                      mi.noise)
            scaled_chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, scaled_noise)
            scaled_chi_squared_term = fitting.chi_squared_term_from_chi_squareds(scaled_chi_squareds)
            scaled_noise_term = fitting.noise_term_from_data(scaled_noise)
            scaled_likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(scaled_chi_squared_term,
                                                                                    scaled_noise_term)

            assert image_im == pytest.approx(fitter.image, 1e-4)
            assert blurring_im == pytest.approx(fitter.blurring_region_image, 1e-4)
            assert blurred_im == pytest.approx(fitter.blurred_image, 1e-4)
            assert residuals == pytest.approx(fitter.blurred_image_residuals, 1e-4)
            assert chi_squareds == pytest.approx(fitter.blurred_image_chi_squareds, 1e-4)
            assert chi_squared_term == pytest.approx(fitter.blurred_image_chi_squared_term, 1e-4)
            assert noise_term == pytest.approx(fitter.noise_term, 1e-4)
            assert likelihood == pytest.approx(fitter.blurred_image_likelihood, 1e-4)
            assert contributions[0] == pytest.approx(fitter.contributions[0], 1e-4)
            assert contributions[1] == pytest.approx(fitter.contributions[1], 1e-4)
            assert scaled_noise == pytest.approx(fitter.scaled_noise, 1e-4)
            assert scaled_chi_squareds == pytest.approx(fitter.blurred_image_scaled_chi_squareds, 1e-4)
            assert scaled_chi_squared_term == pytest.approx(fitter.blurred_image_scaled_chi_squared_term, 1e-4)
            assert scaled_noise_term == pytest.approx(fitter.scaled_noise_term, 1e-4)
            assert scaled_likelihood == pytest.approx(fitter.blurred_image_scaled_likelihood, 1e-4)
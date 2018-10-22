import numpy as np
import pytest

from autolens.imaging import image
from autolens.imaging import mask as mask
from autolens.inversion import inversions
from autolens.inversion import pixelizations
from autolens.inversion import regularization
from autolens.fitting import fitting
from autolens.fitting import fitting_data as fit_data
from autolens.galaxy import galaxy as g
from autolens.lensing import lensing_image
from autolens.lensing import plane as pl
from autolens.lensing import ray_tracing
from autolens.profiles import light_profiles as lp
from test.mock.mock_galaxy import MockHyperGalaxy

@pytest.fixture(name='fi_no_blur')
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


@pytest.fixture(name='fi_blur')
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


class TestBlurImages:

    def test__2x2_image_all_1s__3x3__psf_central_1__no_blurring(self, fi_no_blur):

        _blurring_image = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        blurred_image = fitting.blur_image_including_blurring_region(image=fi_no_blur[:],
                                                                     blurring_image=_blurring_image,
                                                                   convolver=fi_no_blur.convolver_image)
        assert (blurred_image == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__2x2_image_all_1s__3x3_psf_all_1s__image_blurs_to_4s(self, fi_blur):

        _blurring_image = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        blurred_image = fitting.blur_image_including_blurring_region(image=fi_blur[:],
                                                                     blurring_image=_blurring_image,
                                                                   convolver=fi_blur.convolver_image)
        assert (blurred_image == np.array([4.0, 4.0, 4.0, 4.0])).all()

    def test__2x2_image_all_1s__3x3_psf_all_1s__blurring_region__image_blurs_to_9s(self, fi_blur):

        _blurring_image = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        blurred_image = fitting.blur_image_including_blurring_region(image=fi_blur[:],
                                                                     blurring_image=_blurring_image,
                                                                   convolver=fi_blur.convolver_image)
        
        assert (blurred_image == np.array([9.0, 9.0, 9.0, 9.0])).all()


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
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, noise)

        assert (chi_squareds == np.zeros((2, 2))).all()

    def test__model_data_mismatch__chi_sq_non_0(self):
        image = 10.0 * np.ones((2, 2))
        noise = 2.0 * np.ones((2, 2))
        model = np.array([[11, 10],
                          [9, 8]])

        residuals = fitting.residuals_from_image_and_model(image, model)
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, noise)

        assert (chi_squareds == (np.array([[1 / 4, 0],
                                           [1 / 4, 1]]))).all()


class TestLikelihood:

    def test__model_matches_data__noise_all_2s__lh_is_noise_term(self):
        im = np.array([10.0, 10.0, 10.0, 10.0])
        noise = np.array([2.0, 2.0, 2.0, 2.0])
        model = np.array([10.0, 10.0, 10.0, 10.0])

        residuals = fitting.residuals_from_image_and_model(im, model)
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, noise)
        chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_term = fitting.noise_term_from_noise_map(noise)
        likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

        chi_squared_term = 0
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(
            2 * np.pi * 4.0)

        assert likelihood == -0.5 * (chi_squared_term + noise_term)

    def test__model_data_mismatch__chi_squared_term_contributes_to_lh(self):
        im = np.array([10.0, 10.0, 10.0, 10.0])
        noise = np.array([2.0, 2.0, 2.0, 2.0])
        model = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting.residuals_from_image_and_model(im, model)
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, noise)
        chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_term = fitting.noise_term_from_noise_map(noise)
        likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

        # chi squared = 0.25, 0, 0.25, 1.0
        # likelihood = -0.5*(0.25+0+0.25+1.0)

        chi_squared_term = 1.5
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(
            2 * np.pi * 4.0)

        assert likelihood == -0.5 * (chi_squared_term + noise_term)

    def test__same_as_above_but_different_noise_in_each_pixel(self):

        im = np.array([10.0, 10.0, 10.0, 10.0])
        noise = np.array([1.0, 2.0, 3.0, 4.0])
        model = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting.residuals_from_image_and_model(im, model)
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, noise)
        chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_term = fitting.noise_term_from_noise_map(noise)
        likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

        # chi squared = (1.0/1.0)**2, (0.0), (-1.0/3.0)**2.0, (2.0/4.0)**2.0

        chi_squared_term = 1.0 + (1.0 / 9.0) + 0.25
        noise_term = np.log(2 * np.pi * 1.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 9.0) + np.log(
            2 * np.pi * 16.0)

        assert likelihood == pytest.approx(-0.5 * (chi_squared_term + noise_term), 1e-4)
        
        
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

        contributions = fitting.contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images,
                                                                                     hyper_galaxies,
                                                                                     minimum_values)

        assert (contributions[0] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()

    def test__x2_hyper_galaxy__model_and_galaxy_image_different_contributions_change(self):
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
                          MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image = np.array([[0.5, 1.0, 1.5]])

        hyper_galaxy_images = [np.array([[0.5, 1.0, 1.5]]), np.array([[0.5, 1.0, 1.5]])]

        minimum_values = [0.5, 0.6]

        contributions = fitting.contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images,
                                                                                     hyper_galaxies,
                                                                                     minimum_values)

        assert (contributions[0] == np.array([[1.0, 1.0, 1.0]])).all()
        assert (contributions[1] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()

    def test__x2_hyper_galaxy__same_as_above_use_real_hyper_galaxy(self):
        hyper_galaxies = [g.HyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
                          g.HyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image = np.array([[0.5, 1.0, 1.5]])

        hyper_galaxy_images = [np.array([[0.5, 1.0, 1.5]]), np.array([[0.5, 1.0, 1.5]])]

        minimum_values = [0.5, 0.6]

        contributions = fitting.contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images,
                                                                                     hyper_galaxies,
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
        hyper_galaxies = [g.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
                          g.HyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]
        noise = np.array([1.0, 1.0, 1.0])

        scaled_noise = fitting.scaled_noise_from_hyper_galaxies_and_contributions(contributions, hyper_galaxies, noise)

        assert (scaled_noise == np.array([2.5, 2.5, 1.75])).all()


class TestInversionEvidence:

    def test__simple_values(self):

        likelihood_with_regularization_term = \
            fitting.likelihood_with_regularization_from_chi_squared_regularization_and_noise_terms(chi_squared_term=3.0,
                                                                                                           regularization_term=6.0,
                                                                                                           noise_term=2.0)

        assert likelihood_with_regularization_term == -0.5 * (3.0 + 6.0 + 2.0)

        evidence = fitting.evidence_from_reconstruction_terms(chi_squared_term=3.0, regularization_term=6.0,
                                                                      log_covariance_regularization_term=9.0,
                                                                      log_regularization_term=10.0, noise_term=30.0)

        assert evidence == -0.5 * (3.0 + 6.0 + 9.0 - 10.0 + 30.0)


class TestPaddedModelImages:

    def test___3x3_padded_image__no_psf_blurring__produces_padded_image(self):

        psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])), pixel_scale=1.0)
        im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)

        fitting_image = fit_data.FittingImage(im, ma, sub_grid_size=1)

        padded_model_image = fitting.unmasked_model_image_from_fitting_image(fitting_image=fitting_image,
                                                                             _unmasked_image=np.ones(25))

        assert (padded_model_image == np.ones((3,3))).all()

    def test___3x3_padded_image__simple_psf_blurring__produces_padded_image(self):

        psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 2.0],
                                         [0.0, 0.0, 0.0]])), pixel_scale=1.0)
        im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)

        fitting_image = fit_data.FittingImage(im, ma, sub_grid_size=1)

        padded_model_image = fitting.unmasked_model_image_from_fitting_image(fitting_image=fitting_image,
                                                                             _unmasked_image=np.ones(25))

        assert (padded_model_image == 3.0*np.ones((3, 3))).all()

    def test___3x3_padded_image__asymmetric_psf_blurring__produces_padded_image(self):

        psf = image.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                         [0.0, 1.0, 2.0],
                                         [0.0, 0.0, 0.0]])), pixel_scale=1.0)
        im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)

        fitting_image = fit_data.FittingImage(im, ma, sub_grid_size=1)

        _unmasked_image = np.zeros(25)
        _unmasked_image[12] = 1.0

        padded_model_image = fitting.unmasked_model_image_from_fitting_image(fitting_image=fitting_image,
                                                                             _unmasked_image=_unmasked_image)

        assert (padded_model_image == np.array([[0.0, 3.0, 0.0],
                                                [0.0, 1.0, 2.0],
                                                [0.0, 0.0, 0.0]])).all()
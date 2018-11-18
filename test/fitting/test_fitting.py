import numpy as np
import pytest

from autolens.imaging import image as im
from autolens.imaging import mask as msk
from autolens.fitting import fitting
from autolens.fitting import fitting_data as fit_data
from autolens.model.galaxy import galaxy as g
from autolens.lensing import lensing_image
from test.mock.mock_galaxy import MockHyperGalaxy

@pytest.fixture(name='fi_no_blur')
def make_li_no_blur():

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

    return lensing_image.LensingImage(image, ma, sub_grid_size=2)


@pytest.fixture(name='fi_blur')
def make_li_blur():

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


class TestBlurImages:

    def test__2x2_image_all_1s__3x3__psf_central_1__no_blurring(self, fi_no_blur):

        _blurring_image = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        blurred_images = fitting.blur_images_including_blurring_regions(images_=[fi_no_blur[:]],
                                                                        blurring_images_=[_blurring_image],
                                                                        convolvers=[fi_no_blur.convolver_image])
        assert (blurred_images[0] == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__2x2_image_all_1s__3x3_psf_all_1s__image_blurs_to_4s(self, fi_blur):

        _blurring_image = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        blurred_images = fitting.blur_images_including_blurring_regions(images_=[fi_blur[:]],
                                                                        blurring_images_=[_blurring_image],
                                                                        convolvers=[fi_blur.convolver_image])
        assert (blurred_images[0] == np.array([4.0, 4.0, 4.0, 4.0])).all()

    def test__2x2_image_all_1s__3x3_psf_all_1s__blurring_region__image_blurs_to_9s(self, fi_blur):

        _blurring_image = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        blurred_images = fitting.blur_images_including_blurring_regions(images_=[fi_blur[:], fi_blur[:]],
                                                                        blurring_images_=[_blurring_image, _blurring_image],
                                                                        convolvers=[fi_blur.convolver_image,
                                                                                   fi_blur.convolver_image])
        
        assert (blurred_images[0] == np.array([9.0, 9.0, 9.0, 9.0])).all()
        assert (blurred_images[1] == np.array([9.0, 9.0, 9.0, 9.0])).all()


class TestResiduals:

    def test__model_mathces_data__residuals_all_0s(self):

        data = 10.0 * np.ones((2, 2))
        model_data = 10.0 * np.ones((2, 2))

        residuals = fitting.residuals_from_datas_and_model_datas(datas_=[data], model_datas_=[model_data])

        assert (residuals[0] == np.zeros((2, 2))).all()

    def test__model_data_mismatch__x2_datas__residuals_non_0(self):

        data_0 = 10.0 * np.ones((2, 2))
        model_data_0 = np.array([[11, 10],
                              [9, 8]])

        data_1 = 10.0 * np.ones((2, 2))
        model_data_1 = np.array([[10, 10],
                              [9, 8]])

        residuals = fitting.residuals_from_datas_and_model_datas(datas_=[data_0, data_1],
                                                                 model_datas_=[model_data_0, model_data_1])

        assert (residuals[0] == np.array([[-1, 0],
                                          [1, 2]])).all()

        assert (residuals[1] == np.array([[0, 0],
                                          [1, 2]])).all()


class TestChiSquareds:

    def test__model_mathces_data__chi_sq_all_0s(self):

        data = 10.0 * np.ones((2, 2))
        noise_map = 4.0 * np.ones((2, 2))
        model_data = 10.0 * np.ones((2, 2))

        residuals = fitting.residuals_from_datas_and_model_datas(datas_=[data], model_datas_=[model_data])
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(residuals_=residuals, noise_maps_=[noise_map])

        assert (chi_squareds[0] == np.zeros((2, 2))).all()

    def test__model_data_mismatch__x2_images__chi_sq_non_0(self):

        data_0 = 10.0 * np.ones((2, 2))
        data_1 = 10.0 * np.ones((2, 2))

        noise_map_0 = 2.0 * np.ones((2, 2))
        noise_map_1 = 2.0 * np.ones((2, 2))

        model_data_0 = np.array([[11, 10],
                                  [9, 8]])
        model_data_1 = np.array([[10, 10],
                                  [9, 8]])

        residuals = fitting.residuals_from_datas_and_model_datas(datas_=[data_0, data_1],
                                                                 model_datas_=[model_data_0, model_data_1])
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(residuals,
                                                                          noise_maps_=[noise_map_0, noise_map_1])

        assert (chi_squareds[0] == (np.array([[1 / 4, 0],
                                              [1 / 4, 1]]))).all()

        assert (chi_squareds[1] == (np.array([[0    , 0],
                                             [1 / 4, 1]]))).all()


class TestLikelihood:

    def test__model_matches_data__noise_all_2s__lh_is_noise_term(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        model_data = np.array([10.0, 10.0, 10.0, 10.0])

        residuals = fitting.residuals_from_datas_and_model_datas(datas_=[data], model_datas_=[model_data])
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(residuals_=residuals, noise_maps_=[noise_map])
        chi_squared_terms = fitting.chi_squared_terms_from_chi_squareds(chi_squareds_=chi_squareds)
        noise_terms = fitting.noise_terms_from_noise_maps(noise_maps_=[noise_map])
        likelihoods = fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms=chi_squared_terms,
                                                                           noise_terms=noise_terms)

        chi_squared_term = 0
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(
            2 * np.pi * 4.0)

        assert likelihoods[0] == -0.5 * (chi_squared_term + noise_term)

    def test__model_data_mismatch__chi_squared_term_contributes_to_lh(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting.residuals_from_datas_and_model_datas(datas_=[data], model_datas_=[model_data])
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(residuals_=residuals, noise_maps_=[noise_map])
        chi_squared_terms = fitting.chi_squared_terms_from_chi_squareds(chi_squareds_=chi_squareds)
        noise_terms = fitting.noise_terms_from_noise_maps(noise_maps_=[noise_map])
        likelihoods = fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms=chi_squared_terms,
                                                                           noise_terms=noise_terms)

        # chi squared = 0.25, 0, 0.25, 1.0
        # likelihood = -0.5*(0.25+0+0.25+1.0)

        chi_squared_term = 1.5
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(
            2 * np.pi * 4.0)

        assert likelihoods[0] == -0.5 * (chi_squared_term + noise_term)

    def test__same_as_above_but_different_noise_in_each_pixel(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        noise_map = np.array([1.0, 2.0, 3.0, 4.0])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting.residuals_from_datas_and_model_datas(datas_=[data], model_datas_=[model_data])
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(residuals_=residuals, noise_maps_=[noise_map])
        chi_squared_terms = fitting.chi_squared_terms_from_chi_squareds(chi_squareds_=chi_squareds)
        noise_terms = fitting.noise_terms_from_noise_maps(noise_maps_=[noise_map])
        likelihoods = fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms=chi_squared_terms,
                                                                           noise_terms=noise_terms)

        # chi squared = (1.0/1.0)**2, (0.0), (-1.0/3.0)**2.0, (2.0/4.0)**2.0

        chi_squared_term = 1.0 + (1.0 / 9.0) + 0.25
        noise_term = np.log(2 * np.pi * 1.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 9.0) + np.log(
            2 * np.pi * 16.0)

        assert likelihoods[0] == pytest.approx(-0.5 * (chi_squared_term + noise_term), 1e-4)

    def test__same_as_above__x2_images(self):

        data_0 = np.array([10.0, 10.0, 10.0, 10.0])
        noise_map_0 = np.array([2.0, 2.0, 2.0, 2.0])
        model_data_0 = np.array([11.0, 10.0, 9.0, 8.0])

        data_1 = np.array([10.0, 10.0, 10.0, 10.0])
        noise_map_1 = np.array([1.0, 2.0, 3.0, 4.0])
        model_data_1 = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting.residuals_from_datas_and_model_datas(datas_=[data_0, data_1],
                                                                 model_datas_=[model_data_0, model_data_1])
        chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(residuals_=residuals,
                                                                          noise_maps_=[noise_map_0, noise_map_1])
        chi_squared_terms = fitting.chi_squared_terms_from_chi_squareds(chi_squareds_=chi_squareds)
        noise_terms = fitting.noise_terms_from_noise_maps(noise_maps_=[noise_map_0, noise_map_1])
        likelihoods = fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms=chi_squared_terms,
                                                                           noise_terms=noise_terms)

        # chi squared = 0.25, 0, 0.25, 1.0
        # likelihood = -0.5*(0.25+0+0.25+1.0)

        chi_squared_term = 1.5
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(
            2 * np.pi * 4.0)

        likelihood_0 = -0.5 * (chi_squared_term + noise_term)

        assert likelihoods[0] == pytest.approx(likelihood_0, 1e-4)

        # chi squared = (1.0/1.0)**2, (0.0), (-1.0/3.0)**2.0, (2.0/4.0)**2.0

        chi_squared_term = 1.0 + (1.0 / 9.0) + 0.25
        noise_term = np.log(2 * np.pi * 1.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 9.0) + np.log(
            2 * np.pi * 16.0)

        likelihood_1 = -0.5 * (chi_squared_term + noise_term)

        assert likelihoods[1] == pytest.approx(likelihood_1, 1e-4)
        

class TestInversionEvidence:

    def test__simple_values(self):

        likelihood_with_regularization_terms = \
            fitting.likelihoods_with_regularization_from_chi_squared_regularization_and_noise_terms(
                chi_squared_terms=[3.0], regularization_terms=[6.0],  noise_terms=[2.0])

        assert likelihood_with_regularization_terms[0] == -0.5 * (3.0 + 6.0 + 2.0)

        evidences = fitting.evidences_from_reconstruction_terms(chi_squared_terms=[3.0], regularization_terms=[6.0],
                                                                log_covariance_regularization_terms=[9.0],
                                                                log_regularization_terms=[10.0], noise_terms=[30.0])

        assert evidences[0] == -0.5 * (3.0 + 6.0 + 9.0 - 10.0 + 30.0)

    def test__x2_images_simple_values(self):

        likelihood_with_regularization_terms = \
            fitting.likelihoods_with_regularization_from_chi_squared_regularization_and_noise_terms(
                chi_squared_terms=[3.0, 4.0], regularization_terms=[6.0, 7.0], noise_terms=[2.0, 3.0])

        assert likelihood_with_regularization_terms[0] == -0.5 * (3.0 + 6.0 + 2.0)
        assert likelihood_with_regularization_terms[1] == -0.5 * (4.0 + 7.0 + 3.0)

        evidences = fitting.evidences_from_reconstruction_terms(chi_squared_terms=[3.0, 4.0],
                                                               regularization_terms=[6.0, 7.0],
                                                               log_covariance_regularization_terms=[9.0, 10.0],
                                                               log_regularization_terms=[10.0, 11.0],
                                                               noise_terms=[30.0, 31.0])

        assert evidences[0] == -0.5 * (3.0 + 6.0 + 9.0 - 10.0 + 30.0)
        assert evidences[1] == -0.5 * (4.0 + 7.0 + 10.0 - 11.0 + 31.0)


class TestPaddedModelImages:

    def test___3x3_padded_image__no_psf_blurring__produces_padded_image(self):

        psf = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])), pixel_scale=1.0)
        image = im.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

        mask = msk.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)

        fitting_image = fit_data.FittingImage(image, mask, sub_grid_size=1)

        padded_model_image = fitting.unmasked_blurred_images_from_fitting_images(fitting_images=[fitting_image],
                                                                                 unmasked_images_=[np.ones(25)])

        assert (padded_model_image[0] == np.ones((3,3))).all()

    def test___3x3_padded_image__simple_psf_blurring__produces_padded_image(self):

        psf = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 2.0],
                                         [0.0, 0.0, 0.0]])), pixel_scale=1.0)
        image = im.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

        mask = msk.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)

        fitting_image = fit_data.FittingImage(image, mask, sub_grid_size=1)

        padded_model_image = fitting.unmasked_blurred_images_from_fitting_images(fitting_images=[fitting_image],
                                                                                 unmasked_images_=[np.ones(25)])

        assert (padded_model_image == 3.0*np.ones((3, 3))).all()

    def test___3x3_padded_image__asymmetric_psf_blurring__produces_padded_image(self):

        psf = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                         [0.0, 1.0, 2.0],
                                         [0.0, 0.0, 0.0]])), pixel_scale=1.0)
        image = im.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

        mask = msk.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)

        fitting_image = fit_data.FittingImage(image, mask, sub_grid_size=1)

        _unmasked_image = np.zeros(25)
        _unmasked_image[12] = 1.0

        padded_model_image = fitting.unmasked_blurred_images_from_fitting_images(fitting_images=[fitting_image],
                                                                                 unmasked_images_=[_unmasked_image])

        assert (padded_model_image == np.array([[0.0, 3.0, 0.0],
                                                [0.0, 1.0, 2.0],
                                                [0.0, 0.0, 0.0]])).all()


@pytest.fixture(name='image')
def make_image():
    psf = im.PSF(array=np.ones((3, 3)), pixel_scale=3.0, renormalize=False)
    return im.Image(np.ones((4, 4)), pixel_scale=3., psf=psf, noise_map=np.ones((4, 4)))

@pytest.fixture(name="mask")
def make_mask():
    return msk.Mask(np.array([[True, True, True, True],
                              [True, False, False, True],
                              [True, False, True, True],
                              [True, True, True, True]]), pixel_scale=3.0)

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

    def test__same_as_above__contributions_from_fitting_hyper_images(self, image, mask):

        hyper_galaxies = [g.HyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
                          g.HyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image = np.array([[0.5, 1.0, 1.5]])

        hyper_galaxy_images = [np.array([[0.5, 1.0, 1.5]]), np.array([[0.5, 1.0, 1.5]])]

        minimum_values = [0.5, 0.6]

        fitting_hyper_image = fit_data.FittingHyperImage(image=image, mask=mask, hyper_model_image=hyper_model_image,
                                   hyper_galaxy_images=hyper_galaxy_images, hyper_minimum_values=minimum_values)

        contributions = fitting.contributions_from_fitting_hyper_images_and_hyper_galaxies(
            fitting_hyper_images=[fitting_hyper_image], hyper_galaxies=hyper_galaxies)

        assert (contributions[0][0] == np.array([[1.0, 1.0, 1.0]])).all()
        assert (contributions[0][1] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()

    def test__same_as_above__x2_images(self, image, mask):

        hyper_galaxies = [g.HyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
                          g.HyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image = np.array([[0.5, 1.0, 1.5]])
        hyper_galaxy_images = [np.array([[0.5, 1.0, 1.5]]), np.array([[0.5, 1.0, 1.5]])]
        minimum_values = [0.5, 0.6]

        fitting_hyper_image_0 = fit_data.FittingHyperImage(image=image, mask=mask, hyper_model_image=hyper_model_image,
                                   hyper_galaxy_images=hyper_galaxy_images, hyper_minimum_values=minimum_values)

        fitting_hyper_image_1 = fit_data.FittingHyperImage(image=image, mask=mask, hyper_model_image=hyper_model_image,
                                   hyper_galaxy_images=hyper_galaxy_images, hyper_minimum_values=minimum_values)

        contributions = fitting.contributions_from_fitting_hyper_images_and_hyper_galaxies(
            fitting_hyper_images=[fitting_hyper_image_0, fitting_hyper_image_1], hyper_galaxies=hyper_galaxies)

        assert (contributions[0][0] == np.array([[1.0, 1.0, 1.0]])).all()
        assert (contributions[0][1] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()
        assert (contributions[1][0] == np.array([[1.0, 1.0, 1.0]])).all()
        assert (contributions[1][1] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()


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

    def test__same_as_above_but_using_fiting_hyper_image(self, image, mask):

        fitting_hyper_image = fit_data.FittingHyperImage(image=image, mask=mask, hyper_model_image=None,
                                   hyper_galaxy_images=None, hyper_minimum_values=None)

        contributions = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
        hyper_galaxies = [g.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
                          g.HyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]
        scaled_noises = fitting.scaled_noises_from_fitting_hyper_images_contributions_and_hyper_galaxies(
            fitting_hyper_images=[fitting_hyper_image], contributions_=[contributions], hyper_galaxies=hyper_galaxies)

        assert (scaled_noises[0] == np.array([2.5, 2.5, 1.75])).all()

    def test__same_as_above_but_using_x2_fiting_hyper_image(self, image, mask):

        fitting_hyper_image_0 = fit_data.FittingHyperImage(image=image, mask=mask, hyper_model_image=None,
                                   hyper_galaxy_images=None, hyper_minimum_values=None)
        fitting_hyper_image_1 = fit_data.FittingHyperImage(image=image, mask=mask, hyper_model_image=None,
                                   hyper_galaxy_images=None, hyper_minimum_values=None)

        contributions_0 = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
        contributions_1 = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
        hyper_galaxies = [g.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
                          g.HyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]
        scaled_noises = fitting.scaled_noises_from_fitting_hyper_images_contributions_and_hyper_galaxies(
            fitting_hyper_images=[fitting_hyper_image_0, fitting_hyper_image_1],
            contributions_=[contributions_0, contributions_1], hyper_galaxies=hyper_galaxies)

        assert (scaled_noises[0] == np.array([2.5, 2.5, 1.75])).all()
        assert (scaled_noises[1] == np.array([2.5, 2.5, 1.75])).all()
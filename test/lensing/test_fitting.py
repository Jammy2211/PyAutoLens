import numpy as np
import pytest

from autolens.lensing import fitting, ray_tracing, galaxy, lensing_image
from autolens.imaging import mask as mask
from autolens.imaging import image
from autolens.profiles import light_profiles as lp
from autolens.pixelization import pixelization


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
        noise_term = fitting.noise_term_from_noise(noise)
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
        noise_term = fitting.noise_term_from_noise(noise)
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
        noise_term = fitting.noise_term_from_noise(noise)
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

        evidence = fitting.evidence_from_reconstruction_terms(chi_squared_term=3.0, regularization_term=6.0,
                                                              log_covariance_regularization_term=9.0,
                                                              log_regularization_term=10.0, noise_term=30.0)

        assert evidence == -0.5*(3.0 + 6.0 + 9.0 - 10.0 + 30.0)


@pytest.fixture(name="no_galaxies", scope='function')
def make_no_galaxies():
    return [galaxy.Galaxy()]

@pytest.fixture(name="sersic")
def make_sersic():
    return lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0)

@pytest.fixture(name="galaxy_light", scope='function')
def make_galaxy_light(sersic):
    return galaxy.Galaxy(light_profile=sersic)

@pytest.fixture(name='li_no_blur')
def make_li_no_blur():
    im = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
    psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]])))
    im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = mask.Mask(array=ma, pixel_scale=1.0)

    return lensing_image.LensingImage(im, ma, sub_grid_size=2)

@pytest.fixture(name='li_blur')
def make_li_blur():
    im = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
    psf = image.PSF(array=(np.array([[1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0]])))
    im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = mask.Mask(array=ma, pixel_scale=1.0)

    return lensing_image.LensingImage(im, ma, sub_grid_size=2)

@pytest.fixture(name='li_no_blur_1x1')
def make_li_no_blur_1x1():
    psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]])))

    im = image.Image(np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

    ma = mask.Mask(array=np.array([[True, True, True],
                                   [True, False, True],
                                   [True, True, True]]), pixel_scale=1.0)

    return lensing_image.LensingImage(im, ma, sub_grid_size=1)


class MockTracer(object):

    def __init__(self, image, blurring_image, has_grid_mappers=False):
        self.image = image
        self.blurring_image = blurring_image
        self.has_grid_mappers = has_grid_mappers

    @property
    def image_plane_image(self):
        return self.image

    @property
    def image_plane_images_of_planes(self):
        return [self.image]

    @property
    def image_plane_images_of_galaxies(self):
        return [self.image]

    @property
    def image_plane_blurring_image(self):
        return self.blurring_image

    @property
    def image_plane_blurring_images_of_planes(self):
        return [self.blurring_image]

    @property
    def image_plane_blurring_images_of_galaxies(self):
        return [self.blurring_image]

    def reconstructors_from_source_plane(self, borders):
        return MockReconstructor()

    def plane_images_of_planes(self, shape=(30, 30)):
        image = np.zeros(shape)
        k=0
        for i in range(shape[0]):
            for j in range(shape[1]):
                image[i,j] = k
                k += 1

        return [image]*2

    @property
    def hyper_galaxies(self):
        return [MockHyperGalaxy(), MockHyperGalaxy()]


class MockReconstructor(object):

    def __init__(self):
        pass

    def reconstruction_from_reconstructor_and_data(self, lensing_image, noise, convolver_mapping_matrix):
        return MockReconstruction()


class MockReconstruction(object):

    def __init__(self):

        self.blurred_mapping_matrix = np.zeros((1, 1))
        self.regularization_matrix = np.zeros((1, 1))
        self.curvature_matrix = np.zeros((1, 1))
        self.curvature_reg_matrix = np.zeros((1, 1))
        self.solution_vector = np.zeros((1))

    @property
    def reconstructed_image(self):
        return np.zeros((1,1))


# noinspection PyUnusedLocal
class MockLightProfile(lp.LightProfile):

    def intensities_from_grid(self, grid):
        return np.array([self.value])

    def __init__(self, value):
        self.value = value

    def intensities_from_grid_radii(self, grid_radii):
        return self.value

    def intensity_at_coordinates(self, coordinates):
        return self.value


class MockHyperGalaxy(object):

    def __init__(self, contribution_factor=0.0, noise_factor=0.0, noise_power=1.0):
        self.contribution_factor = contribution_factor
        self.noise_factor = noise_factor
        self.noise_power = noise_power

    def contributions_from_hyper_images(self, hyper_model_image, hyper_galaxy_images, minimum_value):
        contributions = hyper_galaxy_images / (hyper_model_image + self.contribution_factor)
        contributions = contributions / np.max(contributions)
        contributions[contributions < minimum_value] = 0.0
        return contributions

    def scaled_noise_from_contributions(self, noise, contributions):
        return self.noise_factor * (noise * contributions) ** self.noise_power


class TestProfileFitter:


    class TestBlurredImagePlaneImages:

        def test__mock_tracer__2x2_image_all_1s__3x3__psf_central_1__no_blurring(self, li_no_blur):

            tracer = MockTracer(image=li_no_blur.mask.map_2d_array_to_masked_1d_array(li_no_blur.image),
                                blurring_image=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

            fitter = fitting.ProfileFitter(lensing_image=li_no_blur, tracer=tracer)

            assert (fitter._model_image == np.array([1.0, 1.0, 1.0, 1.0])).all()

        def test__mock_tracer__2x2_image_all_1s__3x3_psf_all_1s__image_blurs_to_4s(self, li_blur):

            tracer = MockTracer(image=li_blur.mask.map_2d_array_to_masked_1d_array(li_blur.image),
                                blurring_image=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

            fitter = fitting.ProfileFitter(lensing_image=li_blur, tracer=tracer)

            assert (fitter._model_image == np.array([4.0, 4.0, 4.0, 4.0])).all()

        def test__mock_tracer__2x2_image_all_1s__3x3_psf_central_1__include_blurring_region__still_no_blurring(self,
                                                                                                               li_no_blur):

            tracer = MockTracer(image=li_no_blur.mask.map_2d_array_to_masked_1d_array(li_no_blur.image),
                                blurring_image=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

            fitter = fitting.ProfileFitter(lensing_image=li_no_blur, tracer=tracer)

            assert (fitter._model_image == np.array([1.0, 1.0, 1.0, 1.0])).all()

        def test__mock_tracer__2x2_image_all_1s__3x3_psf_all_1s__include_blurring_region_image_blur_to_9s(self, li_blur):

            tracer = MockTracer(image=li_blur.mask.map_2d_array_to_masked_1d_array(li_blur.image),
                                blurring_image=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

            fitter = fitting.ProfileFitter(lensing_image=li_blur, tracer=tracer)

            assert (fitter._model_image == np.array([9.0, 9.0, 9.0, 9.0])).all()

        def test__real_tracer__2x2_image__no_psf_blurring(self, li_no_blur, galaxy_light):

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light],
                                                  image_grids=li_no_blur.grids)

            fitter = fitting.ProfileFitter(lensing_image=li_no_blur, tracer=tracer)

            tracer_non_blurred_image = tracer.image_plane_image

            assert (tracer_non_blurred_image == fitter._model_image).all()

        def test__real_tracer__2x2_image__psf_is_non_symmetric_producing_l_shape(self, galaxy_light):

            psf = image.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                             [0.0, 2.0, 1.0],
                                             [0.0, 0.0, 0.0]])))
            im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_grids=li.grids)

            fitter = fitting.ProfileFitter(lensing_image=li, tracer=tracer)

            # Manually compute result of convolution, which is each central value *2.0 plus its 2 appropriate neighbors

            central_values = tracer.image_plane_image
            blurring_values = tracer.image_plane_blurring_image

            tracer_blurred_image_manual_0 = 2.0 * central_values[0] + 3.0 * central_values[2] + blurring_values[4]
            tracer_blurred_image_manual_1 = 2.0 * central_values[1] + 3.0 * central_values[3] + central_values[0]
            tracer_blurred_image_manual_2 = 2.0 * central_values[2] + 3.0 * blurring_values[9] + blurring_values[6]
            tracer_blurred_image_manual_3 = 2.0 * central_values[3] + 3.0 * blurring_values[10] + central_values[2]

            assert tracer_blurred_image_manual_0 == pytest.approx(fitter._model_image[0], 1e-6)
            assert tracer_blurred_image_manual_1 == pytest.approx(fitter._model_image[1], 1e-6)
            assert tracer_blurred_image_manual_2 == pytest.approx(fitter._model_image[2], 1e-6)
            assert tracer_blurred_image_manual_3 == pytest.approx(fitter._model_image[3], 1e-6)


    class TestBlurredImagePlaneImageOfPlanes:

        def test__mock_tracer__2x2_image_all_1s__3x3_psf_all_1s__include_blurring_region_image_blur_to_9s(self, li_blur):

            tracer = MockTracer(image=li_blur.mask.map_2d_array_to_masked_1d_array(li_blur.image),
                                blurring_image=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

            fitter = fitting.ProfileFitter(lensing_image=li_blur, tracer=tracer)

            assert (fitter.model_images_of_planes[0] == np.array([[0.0, 0.0, 0.0, 0.0],
                                                                  [0.0, 9.0, 9.0, 0.0],
                                                                  [0.0, 9.0, 9.0, 0.0],
                                                                  [0.0, 0.0, 0.0, 0.0]])).all()

        def test__real_tracer__2x2_image__psf_is_non_symmetric_producing_l_shape(self):

            psf = image.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                             [0.0, 2.0, 1.0],
                                             [0.0, 0.0, 0.0]])))
            im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            g0 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_grids=li.grids)

            fitter = fitting.ProfileFitter(lensing_image=li, tracer=tracer)

            blurred_lens_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[0],
                                                                               tracer.image_plane_blurring_images_of_galaxies[0])

            blurred_source_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[1],
                                                                                 tracer.image_plane_blurring_images_of_galaxies[1])

            blurred_lens_image_plane_image = li.grids.image.map_to_2d(blurred_lens_image_plane_image)
            blurred_source_image_plane_image = li.grids.image.map_to_2d(blurred_source_image_plane_image)

            assert (fitter.model_images_of_planes[0] == blurred_lens_image_plane_image).all()
            assert (fitter.model_images_of_planes[1] == blurred_source_image_plane_image).all()

        def test__same_as_above_but_multi_tracer(self):

            psf = image.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                             [0.0, 2.0, 1.0],
                                             [0.0, 0.0, 0.0]])))
            im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            g0 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0), redshift=0.1)
            g1 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0), redshift=0.2)
            g2 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0), redshift=0.3)

            from astropy import cosmology as cosmo

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_grids=li.grids,
                                             cosmology=cosmo.Planck15)

            fitter = fitting.ProfileFitter(lensing_image=li, tracer=tracer)

            blurred_image_plane_image_0 = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[0],
                                                                            tracer.image_plane_blurring_images_of_galaxies[0])

            blurred_image_plane_image_1 = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[1],
                                                                            tracer.image_plane_blurring_images_of_galaxies[1])

            blurred_image_plane_image_2 = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[2],
                                                                            tracer.image_plane_blurring_images_of_galaxies[2])

            blurred_image_plane_image_0 = li.grids.image.map_to_2d(blurred_image_plane_image_0)
            blurred_image_plane_image_1 = li.grids.image.map_to_2d(blurred_image_plane_image_1)
            blurred_image_plane_image_2 = li.grids.image.map_to_2d(blurred_image_plane_image_2)

            assert (fitter.model_images_of_planes[0] == blurred_image_plane_image_0).all()
            assert (fitter.model_images_of_planes[1] == blurred_image_plane_image_1).all()
            assert (fitter.model_images_of_planes[2] == blurred_image_plane_image_2).all()


    class TestBlurredImagePlaneImageOfGalaxies:

        def test__mock_tracer__2x2_image_all_1s__3x3_psf_all_1s__include_blurring_region_image_blur_to_9s(self, li_blur):

            tracer = MockTracer(image=li_blur.mask.map_2d_array_to_masked_1d_array(li_blur.image),
                                blurring_image=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

            fitter = fitting.ProfileFitter(lensing_image=li_blur, tracer=tracer)

            assert (fitter.model_images_of_galaxies[0] == np.array([[0.0, 0.0, 0.0, 0.0],
                                                                    [0.0, 9.0, 9.0, 0.0],
                                                                    [0.0, 9.0, 9.0, 0.0],
                                                                    [0.0, 0.0, 0.0, 0.0]])).all()

        def test__real_tracer__1_galaxy_each_plane__2x2_image__psf_is_non_symmetric_producing_l_shape(self):

            psf = image.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                             [0.0, 2.0, 1.0],
                                             [0.0, 0.0, 0.0]])))
            im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            g0 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_grids=li.grids)

            fitter = fitting.ProfileFitter(lensing_image=li, tracer=tracer)

            blurred_lens_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[0],
                                                                               tracer.image_plane_blurring_images_of_galaxies[0])

            blurred_source_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[1],
                                                                                 tracer.image_plane_blurring_images_of_galaxies[1])

            blurred_lens_image_plane_image = li.grids.image.map_to_2d(blurred_lens_image_plane_image)
            blurred_source_image_plane_image = li.grids.image.map_to_2d(blurred_source_image_plane_image)

            assert (fitter.model_images_of_galaxies[0] == blurred_lens_image_plane_image).all()
            assert (fitter.model_images_of_galaxies[1] == blurred_source_image_plane_image).all()

        def test__same_as_above__real_tracer__multiple_galaxies_each_plane(self):

            psf = image.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                             [0.0, 2.0, 1.0],
                                             [0.0, 0.0, 0.0]])))
            im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            g0 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0))
            g3 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=4.0))
            g4 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=5.0))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1, g2], source_galaxies=[g3, g4],
                                                         image_grids=li.grids)

            fitter = fitting.ProfileFitter(lensing_image=li, tracer=tracer)

            blurred_g0_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[0],
                                                       tracer.image_plane_blurring_images_of_galaxies[0])

            blurred_g1_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[1],
                                                       tracer.image_plane_blurring_images_of_galaxies[1])

            blurred_g2_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[2],
                                                       tracer.image_plane_blurring_images_of_galaxies[2])

            blurred_g3_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[3],
                                                       tracer.image_plane_blurring_images_of_galaxies[3])

            blurred_g4_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[4],
                                                       tracer.image_plane_blurring_images_of_galaxies[4])

            blurred_g0_image_plane_image = li.grids.image.map_to_2d(blurred_g0_image_plane_image)
            blurred_g1_image_plane_image = li.grids.image.map_to_2d(blurred_g1_image_plane_image)
            blurred_g2_image_plane_image = li.grids.image.map_to_2d(blurred_g2_image_plane_image)
            blurred_g3_image_plane_image = li.grids.image.map_to_2d(blurred_g3_image_plane_image)
            blurred_g4_image_plane_image = li.grids.image.map_to_2d(blurred_g4_image_plane_image)

            assert (fitter.model_images_of_galaxies[0] == blurred_g0_image_plane_image).all()
            assert (fitter.model_images_of_galaxies[1] == blurred_g1_image_plane_image).all()
            assert (fitter.model_images_of_galaxies[2] == blurred_g2_image_plane_image).all()
            assert (fitter.model_images_of_galaxies[3] == blurred_g3_image_plane_image).all()
            assert (fitter.model_images_of_galaxies[4] == blurred_g4_image_plane_image).all()

        def test__same_as_above__real_tracer__multiple_galaxies_each_plane_but_now_swap_order_in_tracer(self):

            psf = image.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                             [0.0, 2.0, 1.0],
                                             [0.0, 0.0, 0.0]])))
            im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            g0 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0))
            g3 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=4.0))
            g4 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=5.0))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g1, g3, g4], source_galaxies=[g0, g2],
                                                         image_grids=li.grids)

            fitter = fitting.ProfileFitter(lensing_image=li, tracer=tracer)

            blurred_g1_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[0],
                                                       tracer.image_plane_blurring_images_of_galaxies[0])

            blurred_g3_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[1],
                                                       tracer.image_plane_blurring_images_of_galaxies[1])

            blurred_g4_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[2],
                                                       tracer.image_plane_blurring_images_of_galaxies[2])

            blurred_g0_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[3],
                                                       tracer.image_plane_blurring_images_of_galaxies[3])

            blurred_g2_image_plane_image = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[4],
                                                       tracer.image_plane_blurring_images_of_galaxies[4])

            blurred_g0_image_plane_image = li.grids.image.map_to_2d(blurred_g0_image_plane_image)
            blurred_g1_image_plane_image = li.grids.image.map_to_2d(blurred_g1_image_plane_image)
            blurred_g2_image_plane_image = li.grids.image.map_to_2d(blurred_g2_image_plane_image)
            blurred_g3_image_plane_image = li.grids.image.map_to_2d(blurred_g3_image_plane_image)
            blurred_g4_image_plane_image = li.grids.image.map_to_2d(blurred_g4_image_plane_image)

            assert (fitter.model_images_of_galaxies[0] == blurred_g1_image_plane_image).all()
            assert (fitter.model_images_of_galaxies[1] == blurred_g3_image_plane_image).all()
            assert (fitter.model_images_of_galaxies[2] == blurred_g4_image_plane_image).all()
            assert (fitter.model_images_of_galaxies[3] == blurred_g0_image_plane_image).all()
            assert (fitter.model_images_of_galaxies[4] == blurred_g2_image_plane_image).all()

        def test__same_as_above_but_multi_tracer(self):

            psf = image.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                             [0.0, 2.0, 1.0],
                                             [0.0, 0.0, 0.0]])))
            im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            g0 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0), redshift=0.1)
            g1 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0), redshift=0.2)
            g2 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0), redshift=0.3)

            from astropy import cosmology as cosmo

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_grids=li.grids,
                                             cosmology=cosmo.Planck15)

            fitter = fitting.ProfileFitter(lensing_image=li, tracer=tracer)

            blurred_image_plane_image_0 = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[0],
                                                                            tracer.image_plane_blurring_images_of_galaxies[0])

            blurred_image_plane_image_1 = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[1],
                                                                            tracer.image_plane_blurring_images_of_galaxies[1])

            blurred_image_plane_image_2 = li.convolver_image.convolve_image(tracer.image_plane_images_of_galaxies[2],
                                                                            tracer.image_plane_blurring_images_of_galaxies[2])

            blurred_image_plane_image_0 = li.grids.image.map_to_2d(blurred_image_plane_image_0)
            blurred_image_plane_image_1 = li.grids.image.map_to_2d(blurred_image_plane_image_1)
            blurred_image_plane_image_2 = li.grids.image.map_to_2d(blurred_image_plane_image_2)

            assert (fitter.model_images_of_galaxies[0] == blurred_image_plane_image_0).all()
            assert (fitter.model_images_of_galaxies[1] == blurred_image_plane_image_1).all()
            assert (fitter.model_images_of_galaxies[2] == blurred_image_plane_image_2).all()


    class TestPlaneImageOfPlane:

        def test__plane_image_of_plane_2d__ensure_index_goes_from_top_left(self, li_no_blur):

            # The grid coordinates -2.0 -> 2.0 mean a plane of shape (5,5) has arc second coordinates running over
            # -1.6, -0.8, 0.0, 0.8, 1.6. The centre -1.6, -1.6 of the galaxy means its brighest pixel should be
            # index 0 of the 1D grid and (0,0) of the 2d plane image.

            li_no_blur.grids.image[:,:] = np.array([[-2.0, -2.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]])

            g0 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(centre=(-1.6, -1.6), intensity=1.0))
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                         image_grids=li_no_blur.grids)
            fitter = fitting.ProfileFitter(lensing_image=li_no_blur, tracer=tracer)
            plane_image = fitter.images_of_planes(shape=(5, 5))[1]

            assert plane_image.shape == (5,5)
            assert np.unravel_index(plane_image.argmax(), plane_image.shape) == (0,0)


            g0 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(centre=(1.6, -1.6), intensity=1.0))
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                         image_grids=li_no_blur.grids)
            fitter = fitting.ProfileFitter(lensing_image=li_no_blur, tracer=tracer)
            plane_image = fitter.images_of_planes(shape=(5, 5))[1]

            assert np.unravel_index(plane_image.argmax(), plane_image.shape) == (0,4)


            g0 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(centre=(-1.6, 1.6), intensity=1.0))
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                         image_grids=li_no_blur.grids)
            fitter = fitting.ProfileFitter(lensing_image=li_no_blur, tracer=tracer)
            plane_image = fitter.images_of_planes(shape=(5, 5))[1]

            assert np.unravel_index(plane_image.argmax(), plane_image.shape) == (4,0)


            g0 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(centre=(1.6, 1.6), intensity=1.0))
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                         image_grids=li_no_blur.grids)
            fitter = fitting.ProfileFitter(lensing_image=li_no_blur, tracer=tracer)
            plane_image = fitter.images_of_planes(shape=(5, 5))[1]

            assert np.unravel_index(plane_image.argmax(), plane_image.shape) == (4,4)


    class TestLikelihood:

        def test__1x1_image__tracing_fits_data_perfectly__no_psf_blurring__lh_is_noise_term(self):

            psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])))

            im = image.Image(np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

            ma = mask.Mask(array=np.array([[True, True, True],
                                           [True, False, True],
                                           [True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            g0 = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0], image_grids=li.grids)

            fitter = fitting.ProfileFitter(lensing_image=li, tracer=tracer)

            assert fitter.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        def test___1x1_image__tracing_fits_data_with_chi_sq_5(self):

            psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])))

            im = image.Image(5.0 * np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 3)))

            ma = mask.Mask(array=np.array([[True, True, True],
                                           [True, False, True],
                                           [True, True, True]]), pixel_scale=1.0)

            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            # Setup as a ray trace instance, using a light profile for the lens

            g0 = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0], image_grids=li.grids)

            fitter = fitting.ProfileFitter(lensing_image=li, tracer=tracer)

            assert fitter.likelihood == -0.5 * (16.0 + np.log(2 * np.pi * 1.0))


    class TestPixelizationFitterFromProfileFitter:

        def test__profile_subtracted_image_is_passed_with_other_attributes(self, li_blur):

            tracer = MockTracer(image=li_blur.mask.map_2d_array_to_masked_1d_array(li_blur.image),
                                blurring_image=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

            profile_fitter = fitting.ProfileFitter(lensing_image=li_blur, tracer=tracer)

            # _model_image == np.array([4.0, 4.0, 4.0, 4.0])

            pix_fitter = profile_fitter.pixelization_fitter_with_profile_subtracted_lensing_image(li_blur)

            assert type(pix_fitter) == fitting.PixelizationFitter
            assert (pix_fitter._image == np.array([-3.0, -3.0, -3.0, -3.0])).all()
            assert pix_fitter.tracer == pix_fitter.tracer


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
            im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
            ma = mask.Mask(array=np.array([[True, True,  True,  True,  True],
                                           [True, False, False, False, True],
                                           [True, False, False, False, True],
                                           [True, False, False, False, True],
                                           [True, True,  True,  True,  True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            g0 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                         image_grids=li.grids)

            fitter = fitting.ProfileFitter(lensing_image=li, tracer=tracer)

            image_im = tracer.image_plane_image
            blurring_im = tracer.image_plane_blurring_image
            blurred_im = li.convolver_image.convolve_image(image_im, blurring_im)
            residuals = fitting.residuals_from_image_and_model(li, blurred_im)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, li.noise_map)
            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_noise(li.noise_map)
            likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

            assert li.grids.image.map_to_2d(blurred_im) == pytest.approx(fitter.model_image, 1e-4)
            assert li.grids.image.map_to_2d(residuals) == pytest.approx(fitter.residuals, 1e-4)
            assert li.grids.image.map_to_2d(chi_squareds) == pytest.approx(fitter.chi_squareds, 1e-4)
            assert chi_squared_term == pytest.approx(fitter.chi_squared_term, 1e-4)
            assert noise_term == pytest.approx(fitter._noise_term, 1e-4)
            assert likelihood == pytest.approx(fitter.likelihood, 1e-4)


class TestHyperProfileFitter:


    class TestScaledLikelihood:

        def test__hyper_galaxy_adds_to_noise_term_for_scaled_noise__chi_squared_is_0(self, li_no_blur_1x1):
    
            # Setup as a ray trace instance, using a light profile for the lens
    
            g0 = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))
            g1 = galaxy.Galaxy(light_profile=MockLightProfile(value=0.0))
    
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1],
                                                  image_grids=li_no_blur_1x1.grids)
    
            hyper_model_image = np.array([1.0])
            hyper_galaxy_images = [np.array([1.0]), np.array([1.0])]
    
            tracer.image_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=1.0,
                                                                             noise_power=1.0)
            tracer.image_plane.galaxies[1].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=2.0,
                                                                              noise_power=1.0)
    
            fitter = fitting.HyperProfileFitter(lensing_image=li_no_blur_1x1, tracer=tracer,
                                                hyper_model_image=hyper_model_image,
                                                hyper_galaxy_images=hyper_galaxy_images,
                                                hyper_minimum_values=[0.0, 0.0])
    
            chi_squared_term = 0.0
            scaled_noise_term = np.log(2 * np.pi * 4.0 ** 2.0)
    
            assert fitter.likelihood == -0.5 * (chi_squared_term + scaled_noise_term)
    
        def test__hyper_galaxy_adds_to_noise_term_for_scaled_noise__chi_squared_nonzero(self, no_galaxies, li_no_blur_1x1):
    
            li_no_blur_1x1[0] = 2.0
    
            g0 = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))
            g1 = galaxy.Galaxy(light_profile=MockLightProfile(value=0.0))
    
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=no_galaxies,
                                                         image_grids=li_no_blur_1x1.grids)
    
            hyper_model_image = np.array([1.0])
            hyper_galaxy_images = [np.array([1.0]), np.array([1.0])]
    
            tracer.image_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=1.0,
                                                                             noise_power=1.0)
            tracer.image_plane.galaxies[1].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=2.0,
                                                                              noise_power=1.0)
    
            fitter = fitting.HyperProfileFitter(lensing_image=li_no_blur_1x1, tracer=tracer,
                                                hyper_model_image=hyper_model_image,
                                                hyper_galaxy_images=hyper_galaxy_images,
                                                hyper_minimum_values=[0.0, 0.0])
    
            scaled_chi_squared_term = (1.0/(4.0))**2.0
            scaled_noise_term = np.log(2 * np.pi * 4.0 ** 2.0)
    
            assert fitter.likelihood == -0.5 * (scaled_chi_squared_term + scaled_noise_term)


    class TestPixelizationFitterFromProfileFitter:

        def test__profile_subtracted_image_is_passed_with_other_attributes(self, li_blur):

            tracer = MockTracer(image=li_blur.mask.map_2d_array_to_masked_1d_array(li_blur.image),
                                blurring_image=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

            hyper_model_image = np.array([1.0])
            hyper_galaxy_images = [np.array([1.0]), np.array([1.0])]

            profile_fitter = fitting.HyperProfileFitter(lensing_image=li_blur, tracer=tracer,
                                                hyper_model_image=hyper_model_image,
                                                hyper_galaxy_images=hyper_galaxy_images,
                                                hyper_minimum_values=[0.2, 0.7])

            # _model_image == np.array([4.0, 4.0, 4.0, 4.0])

            pix_fitter = profile_fitter.pixelization_fitter_with_profile_subtracted_lensing_image(li_blur)

            assert type(pix_fitter) == fitting.HyperPixelizationFitter
            assert (pix_fitter._image[:] == np.array([-3.0, -3.0, -3.0, -3.0])).all()
            assert pix_fitter.tracer == pix_fitter.tracer
            assert (pix_fitter.hyper_model_image == profile_fitter.hyper_model_image).all()
            assert (pix_fitter.hyper_galaxy_images[0] == profile_fitter.hyper_galaxy_images[0]).all()
            assert (pix_fitter.hyper_galaxy_images[1] == profile_fitter.hyper_galaxy_images[1]).all()
            assert pix_fitter.hyper_minimum_values[0] == profile_fitter.hyper_minimum_values[0]
            assert pix_fitter.hyper_minimum_values[1] == profile_fitter.hyper_minimum_values[1]


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
            im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
            ma = mask.Mask(array=np.array([[True, True,  True,  True,  True],
                                           [True, False, False, False, True],
                                           [True, False, False, False, True],
                                           [True, False, False, False, True],
                                           [True, True,  True,  True,  True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            hyper_model_image = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 8.0, 6.0, 4.0, 0.0])
            hyper_galaxy_images = [np.array([1.0, 3.0, 5.0, 7.0, 9.0, 8.0, 6.0, 4.0, 0.0]),
                                   np.array([1.0, 3.0, 5.0, 7.0, 9.0, 8.0, 6.0, 4.0, 0.0])]
    
            hyper_galaxy = galaxy.HyperGalaxy(contribution_factor=4.0, noise_factor=2.0, noise_power=3.0)
            g0 = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0),
                               hyper_galaxy=hyper_galaxy)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                         image_grids=li.grids)

            fitter = fitting.HyperProfileFitter(lensing_image=li, tracer=tracer, hyper_model_image=hyper_model_image, 
                                                hyper_galaxy_images=hyper_galaxy_images, hyper_minimum_values=[0.2, 0.8])

            image_im = tracer.image_plane_image
            blurring_im = tracer.image_plane_blurring_image
            blurred_im = li.convolver_image.convolve_image(image_im, blurring_im)
            residuals = fitting.residuals_from_image_and_model(li, blurred_im)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, li.noise_map)
            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_noise(li.noise_map)
            likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

            contributions = fitting.contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images,
                                                            [hyper_galaxy, hyper_galaxy], minimum_values=[0.2, 0.8])

            scaled_noise = fitting.scaled_noise_from_hyper_galaxies_and_contributions(contributions,
                                                                                      [hyper_galaxy, hyper_galaxy],
                                                                                      li.noise_map)
            scaled_chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, scaled_noise)
            scaled_chi_squared_term = fitting.chi_squared_term_from_chi_squareds(scaled_chi_squareds)
            scaled_noise_term = fitting.noise_term_from_noise(scaled_noise)
            scaled_likelihood = fitting.likelihood_from_chi_squared_and_noise_terms(scaled_chi_squared_term,
                                                                                    scaled_noise_term)

            assert li.grids.image.map_to_2d(blurred_im) == pytest.approx(fitter.model_image, 1e-4)
            assert li.grids.image.map_to_2d(residuals) == pytest.approx(fitter.residuals, 1e-4)
            assert li.grids.image.map_to_2d(contributions[0]) == pytest.approx(fitter.contributions[0], 1e-4)
            assert li.grids.image.map_to_2d(contributions[1]) == pytest.approx(fitter.contributions[1], 1e-4)
            assert li.grids.image.map_to_2d(scaled_noise) == pytest.approx(fitter.noise, 1e-4)
            assert li.grids.image.map_to_2d(scaled_chi_squareds) == pytest.approx(fitter.chi_squareds, 1e-4)
            assert scaled_chi_squared_term == pytest.approx(fitter.chi_squared_term, 1e-4)
            assert scaled_noise_term == pytest.approx(fitter.noise_term, 1e-4)
            assert scaled_likelihood == pytest.approx(fitter.likelihood, 1e-4)


class TestPixelizationFitter:


    class TestRectangularPixelization:

        def test__image_all_1s__direct_image_to_source_mapping__perfect_fit_even_with_regularization(self):

            im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]]).view(image.Image)

            ma = np.array([[True,  True,  True,  True, True],
                           [True, False, False, False, True],
                           [True, False, False, False, True],
                           [True, False, False, False, True],
                           [True,  True,  True,  True, True]])
            ma = mask.Mask(ma, pixel_scale=1.0)

            psf = image.PSF(array=np.array([[0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0]]))
            im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
            li = lensing_image.LensingImage(im, ma, sub_grid_size=2)

            pix = pixelization.RectangularRegConst(shape=(3, 3), regularization_coefficients=(1.0,))
            galaxy_pix = galaxy.Galaxy(pixelization=pix)
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy.Galaxy()], source_galaxies=[galaxy_pix],
                                                         image_grids=li.grids)
            fitter = fitting.PixelizationFitter(lensing_image=li, tracer=tracer)

            cov_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
            reg_matrix = np.array([[2.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [-1.0, 3.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, -1.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                   [-1.0, 0.0, 0.0, 3.0, -1.0, 0.0, -1.0, 0.0, 0.0],
                                   [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
                                   [0.0, 0.0, -1.0, 0.0, -1.0, 3.0, 0.0, 0.0, - 1.0],
                                   [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.0, -1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.0, -1.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.0]])
            reg_matrix = reg_matrix + 1e-8 * np.identity(9)
            cov_reg_matrix = cov_matrix + reg_matrix

            chi_sq_term = 0.0
            gl_term = 1e-8
            det_cov_reg_term = np.log(np.linalg.det(cov_reg_matrix))
            det_reg_term = np.log(np.linalg.det(reg_matrix))
            noise_term = 9.0 * np.log(2 * np.pi * 1.0 ** 2.0)
            evidence_expected = -0.5 * (chi_sq_term + gl_term + det_cov_reg_term - det_reg_term + noise_term)

            assert fitter.evidence == pytest.approx(evidence_expected, 1e-4)


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
            im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
            ma = mask.Mask(array=np.array([[True, True, True, True, True],
                                           [True, False, False, False, True],
                                           [True, False, False, False, True],
                                           [True, False, False, False, True],
                                           [True, True, True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            pix = pixelization.RectangularRegConst(shape=(3, 3), regularization_coefficients=(1.0,))
            reconstructor = pix.reconstructor_from_pixelization_and_grids(li.grids, li.borders)
            recon = reconstructor.reconstruction_from_reconstructor_and_data(li, li.noise_map, li.convolver_mapping_matrix)

            g0 = galaxy.Galaxy(pixelization=pix)
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy.Galaxy()], source_galaxies=[g0],
                                                         image_grids=li.grids)

            fitter = fitting.PixelizationFitter(lensing_image=li, tracer=tracer)

            residuals = fitting.residuals_from_image_and_model(li, fitter._reconstruction.reconstructed_image)
            chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, li.noise_map)
            chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
            noise_term = fitting.noise_term_from_noise(li.noise_map)
            regularization_term = recon.regularization_term
            covariance_regularization_term = recon.log_det_curvature_reg_matrix_term
            regularization_matrix_term = recon.log_det_regularization_matrix_term
            evidence = fitting.evidence_from_reconstruction_terms(chi_squared_term, regularization_term,
                                                                  covariance_regularization_term,
                                                                  regularization_matrix_term, noise_term)

            assert li.grids.image.map_to_2d(residuals) == pytest.approx(fitter.residuals, 1e-4)
            assert li.grids.image.map_to_2d(chi_squareds) == pytest.approx(fitter.chi_squareds, 1e-4)
            assert chi_squared_term == pytest.approx(fitter.chi_squared_term, 1e-4)
            assert noise_term == pytest.approx(fitter.noise_term, 1e-4)
            assert regularization_term == pytest.approx(fitter._reconstruction.regularization_term, 1e-4)
            assert covariance_regularization_term == pytest.approx(fitter._reconstruction.log_det_curvature_reg_matrix_term, 1e-4)
            assert regularization_matrix_term == pytest.approx(fitter._reconstruction.log_det_regularization_matrix_term, 1e-4)
            assert evidence == fitter.evidence


class TestHyperPixelizationFitter:

    class TestRectangularPixelization:

        def test__image_all_1s__direct_image_to_source_mapping__perfect_fit_even_with_regularization(self):

            im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]]).view(image.Image)
            ma = np.array([[True,  True,  True,  True, True],
                           [True, False, False, False, True],
                           [True, False, False, False, True],
                           [True, False, False, False, True],
                           [True,  True,  True,  True, True]])
            ma = mask.Mask(ma, pixel_scale=1.0)
            psf = image.PSF(array=np.array([[0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0]]))
            im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
            li = lensing_image.LensingImage(im, ma, sub_grid_size=2)

            pix = pixelization.RectangularRegConst(shape=(3, 3), regularization_coefficients=(1.0,))

            hyper_model_image = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            hyper_galaxy_images = [np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]

            hyper_galaxy = galaxy.HyperGalaxy(contribution_factor=0.0, noise_factor=1.0, noise_power=1.0)
            galaxy_pix = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0),
                                       hyper_galaxy=hyper_galaxy, pixelization=pix)
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy.Galaxy()], source_galaxies=[galaxy_pix],
                                                         image_grids=li.grids)
            fitter = fitting.HyperPixelizationFitter(lensing_image=li, tracer=tracer, hyper_model_image=hyper_model_image,
                                                     hyper_galaxy_images=hyper_galaxy_images, hyper_minimum_values=[0.0])

            cov_matrix = np.array([[0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0,  0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0,  0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0,  0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0,  0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
                                   [0.0,  0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
                                   [0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0],
                                   [0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
                                   [0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]])

            reg_matrix = np.array([[2.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [-1.0, 3.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, -1.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                   [-1.0, 0.0, 0.0, 3.0, -1.0, 0.0, -1.0, 0.0, 0.0],
                                   [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
                                   [0.0, 0.0, -1.0, 0.0, -1.0, 3.0, 0.0, 0.0, - 1.0],
                                   [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.0, -1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.0, -1.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.0]])
            reg_matrix = reg_matrix + 1e-8 * np.identity(9)

            cov_reg_matrix = cov_matrix + reg_matrix

            chi_sq_term = 0.0
            gl_term = 1e-8
            det_cov_reg_term = np.log(np.linalg.det(cov_reg_matrix))
            det_reg_term = np.log(np.linalg.det(reg_matrix))
            noise_term = 9.0 * np.log(2 * np.pi * 2.0 ** 2.0)

            evidence_expected = -0.5 * (chi_sq_term + gl_term + det_cov_reg_term - det_reg_term + noise_term)

            assert fitter.evidence == pytest.approx(evidence_expected, 1e-4)


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
            im = image.Image(im, pixel_scale=1.0, psf=psf, noise_map=np.ones((5, 5)))
            ma = mask.Mask(array=np.array([[True, True, True, True, True],
                                           [True, False, False, False, True],
                                           [True, False, False, False, True],
                                           [True, False, False, False, True],
                                           [True, True, True, True, True]]), pixel_scale=1.0)
            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            pix = pixelization.RectangularRegConst(shape=(3, 3), regularization_coefficients=(1.0,))
            reconstructor = pix.reconstructor_from_pixelization_and_grids(li.grids, li.borders)

            hyper_model_image = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 8.0, 6.0, 4.0, 0.0])
            hyper_galaxy_images = [np.array([1.0, 3.0, 5.0, 7.0, 9.0, 8.0, 6.0, 4.0, 0.0]),
                                   np.array([1.0, 3.0, 5.0, 7.0, 9.0, 8.0, 6.0, 4.0, 0.0])]
            hyper_model = galaxy.HyperGalaxy(contribution_factor=4.0, noise_factor=2.0, noise_power=3.0)
            hyper_galaxy = galaxy.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0),
                                         hyper_galaxy=hyper_model)
            hyper_pix_galaxy = galaxy.Galaxy(pixelization=pix, hyper_galaxy=hyper_model)
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[hyper_galaxy], source_galaxies=[hyper_pix_galaxy],
                                                         image_grids=li.grids)

            fitter = fitting.HyperPixelizationFitter(lensing_image=li, tracer=tracer, hyper_model_image=hyper_model_image,
                                                hyper_galaxy_images=hyper_galaxy_images, hyper_minimum_values=[0.2, 0.8])

            scaled_recon = reconstructor.reconstruction_from_reconstructor_and_data(li, fitter._noise,
                                                                                    li.convolver_mapping_matrix)

            residuals = fitting.residuals_from_image_and_model(li, fitter._reconstruction.reconstructed_image)
            regularization_term = scaled_recon.regularization_term
            scaled_covariance_regularization_term = scaled_recon.log_det_curvature_reg_matrix_term
            regularization_matrix_term = scaled_recon.log_det_regularization_matrix_term

            contributions = fitting.contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images,
                                                            [hyper_model, hyper_model], minimum_values=[0.2, 0.8])
            scaled_noise = fitting.scaled_noise_from_hyper_galaxies_and_contributions(contributions,
                                                                                      [hyper_model, hyper_model],
                                                                                      li.noise_map)

            scaled_chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, scaled_noise)
            scaled_chi_squared_term = fitting.chi_squared_term_from_chi_squareds(scaled_chi_squareds)
            scaled_noise_term = fitting.noise_term_from_noise(scaled_noise)
            scaled_evidence = fitting.evidence_from_reconstruction_terms(scaled_chi_squared_term, regularization_term,
                                                                  scaled_covariance_regularization_term,
                                                                  regularization_matrix_term, scaled_noise_term)

            assert li.grids.image.map_to_2d(contributions[0]) == pytest.approx(fitter.contributions[0], 1e-4)
            assert li.grids.image.map_to_2d(residuals) == pytest.approx(fitter.residuals, 1e-4)
            assert li.grids.image.map_to_2d(scaled_noise) == pytest.approx(fitter.noise, 1e-4)
            assert li.grids.image.map_to_2d(scaled_chi_squareds) == pytest.approx(fitter.chi_squareds, 1e-4)
            assert scaled_chi_squared_term == pytest.approx(fitter.chi_squared_term, 1e-4)
            assert scaled_noise_term == pytest.approx(fitter.noise_term, 1e-4)
            assert regularization_term == pytest.approx(fitter._reconstruction.regularization_term, 1e-4)
            assert scaled_covariance_regularization_term == pytest.approx(fitter._reconstruction.log_det_curvature_reg_matrix_term, 1e-4)
            assert regularization_matrix_term == pytest.approx(fitter._reconstruction.log_det_regularization_matrix_term, 1e-4)
            assert scaled_evidence == fitter.evidence


class MockTracerPositions:

    def __init__(self, positions, noise=None):

        self.positions = positions
        self.noise = noise


class TestPositionFitter:

    def test__x1_positions__mock_position_tracer__maximum_separation_is_correct(self):

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0]])])
        fitter = fitting.PositionFitter(positions=tracer.positions, noise=1.0)
        assert fitter.maximum_separations[0] == 1.0

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [1.0, 1.0]])])
        fitter = fitting.PositionFitter(positions=tracer.positions, noise=1.0)
        assert fitter.maximum_separations[0] == np.sqrt(2)

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [1.0, 3.0]])])
        fitter = fitting.PositionFitter(positions=tracer.positions, noise=1.0)
        assert fitter.maximum_separations[0] == np.sqrt(np.square(1.0) + np.square(3.0))

        tracer = MockTracerPositions(positions=[np.array([[-2.0, -4.0], [1.0, 3.0]])])
        fitter = fitting.PositionFitter(positions=tracer.positions, noise=1.0)
        assert fitter.maximum_separations[0] == np.sqrt(np.square(3.0) + np.square(7.0))

        tracer = MockTracerPositions(positions=[np.array([[8.0, 4.0], [-9.0, -4.0]])])
        fitter = fitting.PositionFitter(positions=tracer.positions, noise=1.0)
        assert fitter.maximum_separations[0] == np.sqrt(np.square(17.0) + np.square(8.0))

    def test_multiple_positions__mock_position_tracer__maximum_separation_is_correct(self):

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.5]])])
        fitter = fitting.PositionFitter(positions=tracer.positions, noise=1.0)
        assert fitter.maximum_separations[0] == 1.0

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]])])
        fitter = fitting.PositionFitter(positions=tracer.positions, noise=1.0)
        assert fitter.maximum_separations[0] == np.sqrt(18)

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])])
        fitter = fitting.PositionFitter(positions=tracer.positions, noise=1.0)
        assert fitter.maximum_separations[0] == np.sqrt(18)

        tracer = MockTracerPositions(positions=[np.array([[-2.0, -4.0], [1.0, 3.0], [0.1, 0.1], [-0.1, -0.1],
                                                          [0.3, 0.4], [-0.6, 0.5]])])
        fitter = fitting.PositionFitter(positions=tracer.positions, noise=1.0)
        assert fitter.maximum_separations[0] == np.sqrt(np.square(3.0) + np.square(7.0))

        tracer = MockTracerPositions(positions=[np.array([[8.0, 4.0], [8.0, 4.0], [-9.0, -4.0]])])
        fitter = fitting.PositionFitter(positions=tracer.positions, noise=1.0)
        assert fitter.maximum_separations[0] == np.sqrt(np.square(17.0) + np.square(8.0))

    def test_multiple_sets_of_positions__multiple_sets_of_max_distances(self):

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.5]]),
                                                np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]]),
                                                np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])])

        fitter = fitting.PositionFitter(positions=tracer.positions, noise=1.0)

        assert fitter.maximum_separations[0] == 1.0
        assert fitter.maximum_separations[1] == np.sqrt(18)
        assert fitter.maximum_separations[2] == np.sqrt(18)

    def test__likelihood__is_sum_of_separations_divided_by_noise(self):

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.5]]),
                                                np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]]),
                                                np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])])

        fitter = fitting.PositionFitter(positions=tracer.positions, noise=1.0)
        assert fitter.chi_squareds[0] == 1.0
        assert fitter.chi_squareds[1] == pytest.approx(18.0, 1e-4)
        assert fitter.chi_squareds[2] == pytest.approx(18.0, 1e-4)
        assert fitter.likelihood == pytest.approx(-0.5*(1.0 + 18 + 18), 1e-4)

        fitter = fitting.PositionFitter(positions=tracer.positions, noise=2.0)
        assert fitter.chi_squareds[0] == (1.0 / 2.0) ** 2.0
        assert fitter.chi_squareds[1] == pytest.approx((np.sqrt(18.0) / 2.0) ** 2.0, 1e-4)
        assert fitter.chi_squareds[2] == pytest.approx((np.sqrt(18.0) / 2.0) ** 2.0, 1e-4)
        assert fitter.likelihood == pytest.approx(-0.5*((1.0 / 2.0) ** 2.0 + (np.sqrt(18.0) / 2.0) ** 2.0 +
                                                        (np.sqrt(18.0) / 2.0) ** 2.0), 1e-4)

    def test__threshold__if_not_met_returns_ray_tracing_exception(self):

        tracer = MockTracerPositions(positions=[np.array([[0.0, 0.0], [0.0, 1.0]])])
        fitter = fitting.PositionFitter(positions=tracer.positions, noise=1.0)

        assert fitter.maximum_separation_within_threshold(threshold=100.0) == True
        assert fitter.maximum_separation_within_threshold(threshold=0.1) == False
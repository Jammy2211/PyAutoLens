import numpy as np
import pytest

from autolens.analysis import fitting, ray_tracing, galaxy
from autolens.imaging import mask as mask
from autolens.imaging import masked_image
from autolens.imaging import image
from autolens.imaging import convolution
from autolens.profiles import light_profiles
from autolens.pixelization import reconstruction


@pytest.fixture(name="no_galaxies", scope='function')
def make_no_galaxies():
    return [galaxy.Galaxy()]


@pytest.fixture(name="galaxy_light_sersic", scope='function')
def make_galaxy_light_sersic():
    sersic = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                             sersic_index=4.0)
    return galaxy.Galaxy(light_profile=sersic)


@pytest.fixture(name="fitter")
def make_fitter(galaxy_light_sersic, no_galaxies):
    im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=image.PSF(np.ones((3, 3)), 1), noise=np.ones((4, 4)))

    ma = mask.Mask(array=np.array([[True, True, True, True],
                                   [True, False, False, True],
                                   [True, False, False, True],
                                   [True, True, True, True]]), pixel_scale=1.0)

    image_2x2 = masked_image.MaskedImage(im, ma)

    ray_tracer = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic], source_galaxies=no_galaxies,
                                    image_plane_grids=image_2x2.grids)
    return fitting.Fitter(image_2x2, ray_tracer)


class MockMapping(object):

    def __init__(self, image_pixels, sub_grid_size, sub_to_image):
        self.image_pixels = image_pixels
        self.sub_pixels = sub_to_image.shape[0]
        self.sub_grid_size = sub_grid_size
        self.sub_grid_size_squared = sub_grid_size ** 2.0
        self.sub_to_image = sub_to_image

    def map_data_sub_to_image(self, data):
        data_image = np.zeros((self.image_pixels,))

        for sub_pixel in range(self.sub_pixels):
            data_image[self.sub_to_image[sub_pixel]] += data[sub_pixel]

        return data_image / self.sub_grid_size_squared


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

    def contributions_from_model_images(self, model_image, galaxy_image, minimum_value):
        contributions = galaxy_image / (model_image + self.contribution_factor)
        contributions = contributions / np.max(contributions)
        contributions[contributions < minimum_value] = 0.0
        return contributions

    def scaled_noise_for_contributions(self, noise, contributions):
        return self.noise_factor * (noise * contributions) ** self.noise_power


class TestFitData:

    def test__1x1_image__tracing_fits_data_perfectly__no_psf_blurring__lh_is_noise_term(self, no_galaxies):

        kernel = np.array([[0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0]])

        im = image.Image(np.ones((3, 3)), pixel_scale=1.0, psf=image.PSF(kernel, 1), noise=np.ones((3, 3)))
        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)

        image_1x1 = masked_image.MaskedImage(im, ma, sub_grid_size=1)

        # Setup as a ray trace instance, using a light profile for the lens

        mock_galaxy = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))
        ray_trace = ray_tracing.Tracer(lens_galaxies=[mock_galaxy], source_galaxies=no_galaxies,
                                       image_plane_grids=image_1x1.grids)

        fitter = fitting.Fitter(masked_image=image_1x1, tracer=ray_trace)

        likelihood = fitter.fit_data_with_profiles()

        assert likelihood == -0.5 * np.log(2 * np.pi * 1.0)

    def test___1x1_image__tracing_fits_data_perfectly__psf_blurs_model_to_5__lh_is_chi_sq_plus_noise(self, no_galaxies):

        kernel = np.array([[0.0, 1.0, 0.0],
                           [1.0, 1.0, 1.0],
                           [0.0, 1.0, 0.0]])

        im = image.Image(np.ones((3, 3)), pixel_scale=1.0, psf=image.PSF(kernel, 1), noise=np.ones((3, 3)))

        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)

        image_1x1 = masked_image.MaskedImage(im, ma, sub_grid_size=1)

        # Setup as a ray trace instance, using a light profile for the lens

        mock_galaxy = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))
        ray_trace = ray_tracing.Tracer(lens_galaxies=[mock_galaxy], source_galaxies=no_galaxies,
                                       image_plane_grids=image_1x1.grids)

        fitter = fitting.Fitter(masked_image=image_1x1, tracer=ray_trace)

        likelihood = fitter.fit_data_with_profiles()

        assert likelihood == -0.5 * (16.0 + np.log(2 * np.pi * 1.0))


class TestGenerateBlurredLightProfileImage:

    def test__1x1_image__no_psf_blurring_into_mask_from_region(self, galaxy_light_sersic, no_galaxies):
        kernel = np.array([[0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0]])

        im = image.Image(np.ones((3, 3)), pixel_scale=1.0, psf=image.PSF(kernel, 1), noise=np.ones((3, 3)))

        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)

        image_1x1 = masked_image.MaskedImage(im, ma, sub_grid_size=1)

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic], source_galaxies=no_galaxies,
                                       image_plane_grids=image_1x1.grids)

        fitter = fitting.Fitter(masked_image=image_1x1, tracer=ray_trace)

        non_blurred_value = ray_trace.generate_image_of_galaxy_light_profiles()
        blurred_value = fitter.blurred_light_profile_image()

        assert non_blurred_value == blurred_value

    def test__1x1_image__psf_all_1s_so_blurs_into_image(self, galaxy_light_sersic, no_galaxies):

        kernel = np.array([[1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0]])

        im = image.Image(np.ones((3, 3)), pixel_scale=1.0, psf=image.PSF(kernel, 1), noise=np.ones((3, 3)))

        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)

        image_1x1 = masked_image.MaskedImage(im, ma, sub_grid_size=1)

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic], source_galaxies=no_galaxies,
                                       image_plane_grids=image_1x1.grids)

        fitter = fitting.Fitter(masked_image=image_1x1, tracer=ray_trace)

        blurred_value = fitter.blurred_light_profile_image()

        # Manually compute result of convolution, which for our PSF of all 1's is just the central value +
        # the (central value x each blurring region value).

        central_value = ray_trace.generate_image_of_galaxy_light_profiles()
        blurring_values = ray_trace.generate_blurring_image_of_galaxy_light_profiles()
        blurred_value_manual = sum(blurring_values[:]) + central_value

        assert blurred_value[0] == pytest.approx(blurred_value_manual[0], 1e-6)

    def test__2x2_image__psf_is_non_symmetric_producing_l_shape(self, galaxy_light_sersic, no_galaxies):
        kernel = np.array([[0.0, 3.0, 0.0],
                           [0.0, 2.0, 1.0],
                           [0.0, 0.0, 0.0]])

        im = image.Image(array=np.ones((4, 4)), pixel_scale=1.0, psf=image.PSF(kernel, 1), noise=np.ones((4, 4)))

        ma = mask.Mask(array=np.array([[True, True, True, True],
                                       [True, False, False, True],
                                       [True, False, False, True],
                                       [True, True, True, True]]), pixel_scale=1.0)

        image_2x2 = masked_image.MaskedImage(im, ma, sub_grid_size=1)

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic], source_galaxies=no_galaxies,
                                       image_plane_grids=image_2x2.grids)

        fitter = fitting.Fitter(masked_image=image_2x2, tracer=ray_trace)

        blurred_value = fitter.blurred_light_profile_image()

        # Manually compute result of convolution, which is each central value *2.0 plus its 2 appropriate neighbors

        central_values = ray_trace.generate_image_of_galaxy_light_profiles()
        blurring_values = ray_trace.generate_blurring_image_of_galaxy_light_profiles()

        blurred_value_manual_0 = 2.0 * central_values[0] + 3.0 * central_values[2] + blurring_values[4]
        blurred_value_manual_1 = 2.0 * central_values[1] + 3.0 * central_values[3] + central_values[0]
        blurred_value_manual_2 = 2.0 * central_values[2] + 3.0 * blurring_values[9] + blurring_values[6]
        blurred_value_manual_3 = 2.0 * central_values[3] + 3.0 * blurring_values[10] + central_values[2]

        assert blurred_value_manual_0 == pytest.approx(blurred_value[0], 1e-6)
        assert blurred_value_manual_1 == pytest.approx(blurred_value[1], 1e-6)
        assert blurred_value_manual_2 == pytest.approx(blurred_value[2], 1e-6)
        assert blurred_value_manual_3 == pytest.approx(blurred_value[3], 1e-6)


class TestFitDataWithProfilesHyperGalaxy:

    def test__chi_sq_is_0__hyper_galaxy_adds_to_noise_term(self, no_galaxies):

        kernel = np.array([[0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0]])

        im = image.Image(np.ones((3, 3)), pixel_scale=1.0, psf=image.PSF(kernel, 1), noise=np.ones((3, 3)))

        ma = mask.Mask(array=np.array([[True, True, True],
                                       [True, False, True],
                                       [True, True, True]]), pixel_scale=1.0)

        image_1x1 = masked_image.MaskedImage(im, ma, sub_grid_size=1)

        # Setup as a ray trace instance, using a light profile for the lens

        mock_galaxy = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))

        ray_trace = ray_tracing.Tracer(lens_galaxies=[mock_galaxy], source_galaxies=no_galaxies,
                                       image_plane_grids=image_1x1.grids)

        model_image = np.array([1.0])
        galaxy_images = [np.array([1.0]), np.array([1.0])]

        ray_trace.image_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=1.0,
                                                                         noise_power=1.0)
        ray_trace.source_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=0.0, noise_factor=2.0,
                                                                          noise_power=1.0)

        fitter = fitting.Fitter(masked_image=image_1x1,
                                tracer=ray_trace)

        likelihood = fitter.fit_data_with_profiles_and_model_images(model_image, galaxy_images)

        assert likelihood == -0.5 * np.log(2 * np.pi * 4.0 ** 2.0)  # should be 1


class TestComputeBlurredImages:

    def test__psf_just_central_1_so_no_blurring__no_blurring_region__image_in_is_image_out(self):

        image_2d = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])

        kernel = np.array([[0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0]])

        ma = np.array([[True, True, True, True],
                       [True, False, False, True],
                       [True, False, False, True],
                       [True, True, True, True]])

        ma = mask.Mask(array=ma, pixel_scale=1.0)
        blurring_mask = ma.blurring_mask_for_kernel_shape(kernel_shape=kernel.shape)

        convolver = convolution.ConvolverImage(mask=ma, blurring_mask=blurring_mask, psf=kernel)

        im = ma.map_to_1d(image_2d)

        blurring_image = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        blurred_image = fitting.blur_image_including_blurring_region(im, blurring_image, convolver)

        assert (blurred_image == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__psf_all_1s_so_blurring_gives_4s__no_blurring_region__image_in_is_image_out(self):

        image_2d = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])

        kernel = np.array([[1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0]])

        ma = np.array([[True, True, True, True],
                       [True, False, False, True],
                       [True, False, False, True],
                       [True, True, True, True]])

        ma = mask.Mask(array=ma, pixel_scale=1.0)
        blurring_mask = ma.blurring_mask_for_kernel_shape(kernel_shape=kernel.shape)

        convolver = convolution.ConvolverImage(mask=ma, blurring_mask=blurring_mask, psf=kernel)

        im = ma.map_to_1d(image_2d)
        blurring_image = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        blurred_image = fitting.blur_image_including_blurring_region(im, blurring_image, convolver)

        assert (blurred_image == np.array([4.0, 4.0, 4.0, 4.0])).all()

    def test__psf_just_central_1__include_blurring_region_blurring_region_not_blurred_in_so_return_image(self):

        image_2d = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])

        kernel = np.array([[0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0]])

        ma = np.array([[True, True, True, True],
                       [True, False, False, True],
                       [True, False, False, True],
                       [True, True, True, True]])

        ma = mask.Mask(array=ma, pixel_scale=1.0)
        blurring_mask = ma.blurring_mask_for_kernel_shape(kernel_shape=kernel.shape)

        convolver = convolution.ConvolverImage(mask=ma, blurring_mask=blurring_mask, psf=kernel)

        im = ma.map_to_1d(image_2d)
        blurring_image = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        blurred_image = fitting.blur_image_including_blurring_region(im, blurring_image, convolver)

        assert (blurred_image == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__psf_all_1s__include_blurring_region_image_turns_to_9s(self):

        image_2d = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])

        kernel = np.array([[1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0]])

        ma = np.array([[True, True, True, True],
                       [True, False, False, True],
                       [True, False, False, True],
                       [True, True, True, True]])

        ma = mask.Mask(array=ma, pixel_scale=1.0)
        blurring_mask = ma.blurring_mask_for_kernel_shape(kernel_shape=kernel.shape)

        convolver = convolution.ConvolverImage(mask=ma, blurring_mask=blurring_mask, psf=kernel)

        im = ma.map_to_1d(image_2d)

        blurring_image = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        blurred_image = fitting.blur_image_including_blurring_region(im, blurring_image, convolver)

        assert (blurred_image == np.array([9.0, 9.0, 9.0, 9.0])).all()


class TestGenerateContributions:

    def test__x1_hyper_galaxy__model_image_is_galaxy_image__contributions_all_1(self):

        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        model_image = np.array([[1.0, 1.0, 1.0]])

        galaxy_images = [np.array([[1.0, 1.0, 1.0]])]

        minimum_values = [0.0]

        contributions = fitting.generate_contributions(model_image, galaxy_images, hyper_galaxies, minimum_values)

        assert (contributions[0] == np.array([[1.0, 1.0, 1.0]])).all()

    def test__x1_hyper_galaxy__model_image_and_galaxy_image_different_contributions_change(self):

        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        model_image = np.array([[0.5, 1.0, 1.5]])

        galaxy_images = [np.array([[0.5, 1.0, 1.5]])]

        minimum_values = [0.6]

        contributions = fitting.generate_contributions(model_image, galaxy_images, hyper_galaxies, minimum_values)

        assert (contributions[0] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()

    def test__x2_hyper_galaxy__model_image_and_galaxy_image_different_contributions_change(self):

        hyper_galaxies = [MockHyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
                          MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        model_image = np.array([[0.5, 1.0, 1.5]])

        galaxy_images = [np.array([[0.5, 1.0, 1.5]]), np.array([[0.5, 1.0, 1.5]])]

        minimum_values = [0.5, 0.6]

        contributions = fitting.generate_contributions(model_image, galaxy_images, hyper_galaxies, minimum_values)

        assert (contributions[0] == np.array([[1.0, 1.0, 1.0]])).all()
        assert (contributions[1] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()

    def test__x2_hyper_galaxy__same_as_above_use_real_hyper_galaxy(self):

        hyper_galaxies = [galaxy.HyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
                          galaxy.HyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        model_image = np.array([[0.5, 1.0, 1.5]])

        galaxy_images = [np.array([[0.5, 1.0, 1.5]]), np.array([[0.5, 1.0, 1.5]])]

        minimum_values = [0.5, 0.6]

        contributions = fitting.generate_contributions(model_image, galaxy_images, hyper_galaxies, minimum_values)

        assert (contributions[0] == np.array([[1.0, 1.0, 1.0]])).all()
        assert (contributions[1] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()


class TestGenerateScaledNoise:

    def test__x1_hyper_galaxy__noise_factor_is_0__scaled_noise_is_input_noise(self, fitter):

        fitter.tracer.image_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0,
                                                                             noise_power=1.0)

        fitter.image.noise = np.array([1.0, 1.0, 1.0])

        contributions = [np.array([1.0, 1.0, 2.0])]

        scaled_noise = fitter.scaled_noise_for_contributions(contributions)

        assert (scaled_noise == fitter.image.noise).all()

    def test__x1_hyper_galaxy__noise_factor_and_power_are_1__scaled_noise_added_to_input_noise(self, fitter):
        fitter.tracer.image_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0,
                                                                             noise_power=1.0)

        fitter.image.noise = np.array([1.0, 1.0, 1.0])

        contributions = [np.array([1.0, 1.0, 0.5])]

        scaled_noise = fitter.scaled_noise_for_contributions(contributions)

        assert (scaled_noise == np.array([2.0, 2.0, 1.5])).all()

    def test__x1_hyper_galaxy__noise_factor_1_and_power_is_2__scaled_noise_added_to_input_noise(self, fitter):
        fitter.tracer.image_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0,
                                                                             noise_power=2.0)

        fitter.image.noise = np.array([1.0, 1.0, 1.0])

        contributions = [np.array([1.0, 1.0, 0.5])]

        scaled_noise = fitter.scaled_noise_for_contributions(contributions)

        assert (scaled_noise == np.array([2.0, 2.0, 1.25])).all()

    def test__x2_hyper_galaxy__noise_factor_1_and_power_is_2__scaled_noise_added_to_input_noise(self, fitter):
        fitter.image.noise = np.array([1.0, 1.0, 1.0])

        fitter.tracer.image_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0,
                                                                             noise_power=2.0)
        fitter.tracer.source_plane.galaxies[0].hyper_galaxy = MockHyperGalaxy(contribution_factor=1.0, noise_factor=2.0,
                                                                              noise_power=1.0)

        contributions = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]

        scaled_noise = fitter.scaled_noise_for_contributions(contributions)

        assert (scaled_noise == np.array([2.5, 2.5, 1.75])).all()

    def test__x2_hyper_galaxy__same_as_above_but_use_real_hyper_galaxy(self, fitter):
        fitter.tracer.image_plane.galaxies[0].hyper_galaxy = galaxy.HyperGalaxy(contribution_factor=1.0,
                                                                                noise_factor=1.0, noise_power=2.0)
        fitter.tracer.source_plane.galaxies[0].hyper_galaxy = galaxy.HyperGalaxy(contribution_factor=1.0,
                                                                                 noise_factor=2.0, noise_power=1.0)

        fitter.image.noise = np.array([1.0, 1.0, 1.0])

        contributions = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]

        scaled_noise = fitter.scaled_noise_for_contributions(contributions)

        assert (scaled_noise == np.array([2.5, 2.5, 1.75])).all()


class TestLikelihood:

    def test__model_matches_data__noise_all_2s__lh_is_noise_term(self):
        im = np.array([10.0, 10.0, 10.0, 10.0])
        noise = np.array([2.0, 2.0, 2.0, 2.0])
        model_image = np.array([10.0, 10.0, 10.0, 10.0])

        likelihood = fitting.compute_likelihood(im, noise, model_image)

        chi_sq_term = 0
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(
            2 * np.pi * 4.0)

        assert likelihood == -0.5 * (chi_sq_term + noise_term)

    def test__model_data_mismatch__chi_sq_term_contributes_to_lh(self):
        im = np.array([10.0, 10.0, 10.0, 10.0])
        noise = np.array([2.0, 2.0, 2.0, 2.0])
        model_image = np.array([11.0, 10.0, 9.0, 8.0])

        likelihood = fitting.compute_likelihood(im, noise, model_image)

        # chi squared = 0.25, 0, 0.25, 1.0
        # likelihood = -0.5*(0.25+0+0.25+1.0)

        chi_sq_term = 1.5
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(
            2 * np.pi * 4.0)

        assert likelihood == -0.5 * (chi_sq_term + noise_term)

    def test__same_as_above_but_different_noise_in_each_pixel(self):
        im = np.array([10.0, 10.0, 10.0, 10.0])
        noise = np.array([1.0, 2.0, 3.0, 4.0])
        model_image = np.array([11.0, 10.0, 9.0, 8.0])

        likelihood = fitting.compute_likelihood(im, noise, model_image)

        # chi squared = (1.0/1.0)**2, (0.0), (-1.0/3.0)**2.0, (2.0/4.0)**2.0

        chi_sq_term = 1.0 + (1.0 / 9.0) + 0.25
        noise_term = np.log(2 * np.pi * 1.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 9.0) + np.log(
            2 * np.pi * 16.0)

        assert likelihood == pytest.approx(-0.5 * (chi_sq_term + noise_term), 1e-4)


class TestPixelizationEvidence:

    def test__simple_values(self):
        im = np.array([10.0, 10.0, 10.0, 10.0])
        noise = np.array([2.0, 2.0, 2.0, 2.0])
        model_image = np.array([10.0, 10.0, 10.0, 10.0])

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

        evidence = fitting.compute_pixelization_evidence(im, noise, model_image, pix_fit)

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
        model_image = np.array([11.0, 10.0, 9.0, 8.0])

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

        evidence = fitting.compute_pixelization_evidence(im, noise, model_image, pix_fit)

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
        model_image = np.array([11.0, 13.0, 9.0, 8.0])

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

        evidence = fitting.compute_pixelization_evidence(im, noise, model_image, pix_fit)

        chi_sq_term = fitting.compute_chi_sq_term(im, noise, model_image)
        reg_term = pix_fit.regularization_term_from_reconstruction()
        log_det_cov_reg = pix_fit.log_determinant_of_matrix_cholesky(pix_fit.covariance_regularization)
        log_det_reg = pix_fit.log_determinant_of_matrix_cholesky(pix_fit.regularization)
        noise_term = fitting.compute_noise_term(noise)

        assert evidence == pytest.approx(-0.5 * (chi_sq_term + reg_term + log_det_cov_reg - log_det_reg + noise_term),
                                         1e-4)

import numpy as np
import pytest

from src.analysis import fitting, ray_tracing, galaxy
from src.imaging import grids
from src.imaging import mask as mask
from src.profiles import light_profiles
from src.pixelization import frame_convolution


@pytest.fixture(name="no_galaxies", scope='function')
def make_no_galaxies():
    return [galaxy.Galaxy()]


@pytest.fixture(name="galaxy_light_sersic", scope='function')
def make_galaxy_light_sersic():
    sersic = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                             sersic_index=4.0)
    return galaxy.Galaxy(light_profile=sersic)


@pytest.fixture(name="image_1x1", scope='function')
def make_1x1_image():
    class Im(object):
        pass

    im = Im()

    im.ma = np.array([[True, True, True],
                      [True, False, True],
                      [True, True, True]])
    im.ma = mask.Mask(array=im.ma, pixel_scale=1.0)

    im.image = grids.GridData(grid_data=np.array([1.0]))
    im.noise = grids.GridData(grid_data=np.array([1.0]))
    im.exposure_time = grids.GridData(grid_data=np.array([1.0]))

    im.frame = frame_convolution.FrameMaker(mask=im.ma)
    im.convolver = im.frame.convolver_for_kernel_shape(kernel_shape=(3, 3),
                                                       blurring_region_mask=im.ma.compute_blurring_mask(
                                                           kernel_shape=(3, 3)))

    im.grid_datas = grids.DataCollection(image=im.image, noise=im.noise, exposure_time=im.exposure_time)
    im.grid_collection = grids.CoordsCollection.from_mask(mask=im.ma, grid_size_sub=1, blurring_shape=(3, 3))

    return im


@pytest.fixture(name="image_2x2", scope='function')
def make_2x2_image():
    class Im(object):
        pass

    im = Im()

    im.ma = np.array([[True, True, True, True],
                      [True, False, False, True],
                      [True, False, False, True],
                      [True, True, True, True]])
    im.ma = mask.Mask(array=im.ma, pixel_scale=1.0)

    im.image = grids.GridData(grid_data=np.array([1.0]))
    im.noise = grids.GridData(grid_data=np.array([1.0]))
    im.exposure_time = grids.GridData(grid_data=np.array([1.0]))

    im.frame = frame_convolution.FrameMaker(mask=im.ma)
    im.convolver = im.frame.convolver_for_kernel_shape(kernel_shape=(3, 3),
                                                       blurring_region_mask=im.ma.compute_blurring_mask(
                                                           kernel_shape=(3, 3)))

    im.grid_datas = grids.DataCollection(image=im.image, noise=im.noise, exposure_time=im.exposure_time)
    im.grid_collection = grids.CoordsCollection.from_mask(mask=im.ma, grid_size_sub=1, blurring_shape=(3, 3))

    return im


# noinspection PyUnusedLocal
class MockLightProfile(object):

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

    def compute_contributions(self, model_image, galaxy_image, minimum_value):
        contributions = galaxy_image / (model_image + self.contribution_factor)
        contributions = contributions / np.max(contributions)
        contributions[contributions < minimum_value] = 0.0
        return contributions

    def compute_scaled_noise(self, noise, contributions):
        return self.noise_factor * (noise * contributions) ** self.noise_power


class TestFitData:

    def test__1x1_image__tracing_fits_data_perfectly__no_psf_blurring__lh_is_noise_term(self, image_1x1, no_galaxies):
        # Setup the mask, grid data and PSF

        kernel_convolver = image_1x1.convolver.convolver_for_kernel(kernel=np.array([[0.0, 0.0, 0.0],
                                                                                     [0.0, 1.0, 0.0],
                                                                                     [0.0, 0.0, 0.0]]))

        # Setup as a ray trace instance, using a light profile for the lens

        mock_galaxy = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))
        ray_trace = ray_tracing.Tracer(lens_galaxies=[mock_galaxy], source_galaxies=no_galaxies,
                                       image_plane_grids=image_1x1.grid_collection)

        likelihood = fitting.fit_data_with_profiles(image_1x1.grid_datas, kernel_convolver, ray_trace)

        assert likelihood == -0.5 * np.log(2 * np.pi * 1.0)

    def test___1x1_image__tracing_fits_data_perfectly__psf_blurs_model_to_5__lh_is_chi_sq_plus_noise(self, image_1x1,
                                                                                                     no_galaxies):
        kernel_convolver = image_1x1.convolver.convolver_for_kernel(kernel=np.array([[0.0, 1.0, 0.0],
                                                                                     [1.0, 1.0, 1.0],
                                                                                     [0.0, 1.0, 0.0]]))

        # Setup as a ray trace instance, using a light profile for the lens

        mock_galaxy = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))
        ray_trace = ray_tracing.Tracer(lens_galaxies=[mock_galaxy], source_galaxies=no_galaxies,
                                       image_plane_grids=image_1x1.grid_collection)

        likelihood = fitting.fit_data_with_profiles(image_1x1.grid_datas, kernel_convolver, ray_trace)

        assert likelihood == -0.5 * (16.0 + np.log(2 * np.pi * 1.0))


class TestGenerateBlurredLightProfileImage:

    def test__1x1_image__no_psf_blurring_into_mask_from_region(self, image_1x1,
                                                               galaxy_light_sersic, no_galaxies):
        kernel_convolver = image_1x1.convolver.convolver_for_kernel(kernel=np.array([[0.0, 0.0, 0.0],
                                                                                     [0.0, 1.0, 0.0],
                                                                                     [0.0, 0.0, 0.0]]))

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic], source_galaxies=no_galaxies,
                                       image_plane_grids=image_1x1.grid_collection)

        non_blurred_value = ray_trace.generate_image_of_galaxy_light_profiles()
        blurred_value = fitting.generate_blurred_light_profile_image(tracer=ray_trace,
                                                                     kernel_convolver=kernel_convolver)

        assert non_blurred_value == blurred_value

    def test__1x1_image__psf_all_1s_so_blurs_into_image(self, image_1x1, galaxy_light_sersic, no_galaxies):
        kernel_convolver = image_1x1.convolver.convolver_for_kernel(kernel=np.array([[1.0, 1.0, 1.0],
                                                                                     [1.0, 1.0, 1.0],
                                                                                     [1.0, 1.0, 1.0]]))

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic], source_galaxies=no_galaxies,
                                       image_plane_grids=image_1x1.grid_collection)

        blurred_value = fitting.generate_blurred_light_profile_image(tracer=ray_trace,
                                                                     kernel_convolver=kernel_convolver)

        # Manually compute result of convolution, which for our PSF of all 1's is just the central value +
        # the (central value x each blurring region value).

        central_value = ray_trace.generate_image_of_galaxy_light_profiles()
        blurring_values = ray_trace.generate_blurring_image_of_galaxy_light_profiles()
        blurred_value_manual = sum(blurring_values[:]) + central_value

        assert blurred_value[0] == pytest.approx(blurred_value_manual[0], 1e-6)

    def test__2x2_image__psf_is_non_symmetric_producing_l_shape(self, image_2x2, galaxy_light_sersic, no_galaxies):
        kernel_convolver = image_2x2.convolver.convolver_for_kernel(kernel=np.array([[0.0, 3.0, 0.0],
                                                                                     [0.0, 2.0, 1.0],
                                                                                     [0.0, 0.0, 0.0]]))

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic], source_galaxies=no_galaxies,
                                       image_plane_grids=image_2x2.grid_collection)

        blurred_value = fitting.generate_blurred_light_profile_image(tracer=ray_trace,
                                                                     kernel_convolver=kernel_convolver)

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

    def test__chi_sq_is_0__hyper_galaxy_adds_to_noise_term(self, image_1x1, no_galaxies):
        kernel_convolver = image_1x1.convolver.convolver_for_kernel(kernel=np.array([[0.0, 0.0, 0.0],
                                                                                     [0.0, 1.0, 0.0],
                                                                                     [0.0, 0.0, 0.0]]))

        # Setup as a ray trace instance, using a light profile for the lens

        mock_galaxy = galaxy.Galaxy(light_profile=MockLightProfile(value=1.0))

        ray_trace = ray_tracing.Tracer(lens_galaxies=[mock_galaxy], source_galaxies=no_galaxies,
                                       image_plane_grids=image_1x1.grid_collection)

        model_image = np.array([1.0])
        galaxy_images = [np.array([1.0]), np.array([1.0])]
        minimum_values = [0.0, 0.0]

        hyper_galaxies = [MockHyperGalaxy(contribution_factor=0.0, noise_factor=1.0, noise_power=1.0),
                          MockHyperGalaxy(contribution_factor=0.0, noise_factor=2.0, noise_power=1.0)]

        likelihood = fitting.fit_data_with_profiles_hyper_galaxies(image_1x1.grid_datas, kernel_convolver, ray_trace,
                                                                   model_image, galaxy_images, minimum_values,
                                                                   hyper_galaxies)

        assert likelihood == -0.5 * np.log(2 * np.pi * 4.0 ** 2.0)  # should be 1


class TestComputeBlurredImages:

    def test__psf_just_central_1_so_no_blurring__no_blurring_region__image_in_is_image_out(self):
        image_2d = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])

        ma = np.array([[True, True, True, True],
                       [True, False, False, True],
                       [True, False, False, True],
                       [True, True, True, True]])

        ma = mask.Mask(array=ma, pixel_scale=1.0)

        frame = frame_convolution.FrameMaker(mask=ma)
        convolver = frame.convolver_for_kernel_shape(kernel_shape=(3, 3),
                                                     blurring_region_mask=ma.compute_blurring_mask(kernel_shape=(3, 3)))
        kernel_convolver = convolver.convolver_for_kernel(kernel=np.array([[0., 0., 0.],
                                                                           [0., 1., 0.],
                                                                           [0., 0., 0.]]))

        image = grids.GridData.from_mask(data=image_2d, mask=ma)
        blurring_image = grids.GridData(grid_data=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        blurred_image = fitting.blur_image_including_blurring_region(image, blurring_image, kernel_convolver)

        assert (blurred_image == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__psf_all_1s_so_blurring_gives_4s__no_blurring_region__image_in_is_image_out(self):
        image_2d = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])

        ma = np.array([[True, True, True, True],
                       [True, False, False, True],
                       [True, False, False, True],
                       [True, True, True, True]])
        ma = mask.Mask(array=ma, pixel_scale=1.0)

        frame = frame_convolution.FrameMaker(mask=ma)
        convolver = frame.convolver_for_kernel_shape(kernel_shape=(3, 3),
                                                     blurring_region_mask=ma.compute_blurring_mask(kernel_shape=(3, 3)))
        kernel_convolver = convolver.convolver_for_kernel(kernel=np.array([[1.0, 1.0, 1.0],
                                                                           [1.0, 1.0, 1.0],
                                                                           [1.0, 1.0, 1.0]]))

        image = grids.GridData.from_mask(data=image_2d, mask=ma)
        blurring_image = grids.GridData(grid_data=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        blurred_image = fitting.blur_image_including_blurring_region(image, blurring_image, kernel_convolver)

        assert (blurred_image == np.array([4.0, 4.0, 4.0, 4.0])).all()

    def test__psf_just_central_1__include_blurring_region_blurring_region_not_blurred_in_so_return_image(self):
        image_2d = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])

        ma = np.array([[True, True, True, True],
                       [True, False, False, True],
                       [True, False, False, True],
                       [True, True, True, True]])
        ma = mask.Mask(array=ma, pixel_scale=1.0)

        frame = frame_convolution.FrameMaker(mask=ma)
        convolver = frame.convolver_for_kernel_shape(kernel_shape=(3, 3),
                                                     blurring_region_mask=ma.compute_blurring_mask(kernel_shape=(3, 3)))
        kernel_convolver = convolver.convolver_for_kernel(kernel=np.array([[0.0, 0.0, 0.0],
                                                                           [0.0, 1.0, 0.0],
                                                                           [0.0, 0.0, 0.0]]))

        image = grids.GridData.from_mask(data=image_2d, mask=ma)
        blurring_image = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        blurred_image = fitting.blur_image_including_blurring_region(image, blurring_image, kernel_convolver)

        assert (blurred_image == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__psf_all_1s__include_blurring_region_image_turns_to_9s(self):
        image_2d = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])
        ma = np.array([[True, True, True, True],
                       [True, False, False, True],
                       [True, False, False, True],
                       [True, True, True, True]])
        ma = mask.Mask(array=ma, pixel_scale=1.0)

        frame = frame_convolution.FrameMaker(mask=ma)
        convolver = frame.convolver_for_kernel_shape(kernel_shape=(3, 3),
                                                     blurring_region_mask=ma.compute_blurring_mask(kernel_shape=(3, 3)))
        kernel_convolver = convolver.convolver_for_kernel(kernel=np.array([[1.0, 1.0, 1.0],
                                                                           [1.0, 1.0, 1.0],
                                                                           [1.0, 1.0, 1.0]]))

        image = grids.GridData.from_mask(data=image_2d, mask=ma)
        blurring_image = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        blurred_image = fitting.blur_image_including_blurring_region(image, blurring_image, kernel_convolver)

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

    def test__x1_hyper_galaxy__noise_factor_is_0__scaled_noise_is_input_noise(self):
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        noise = np.array([1.0, 1.0, 1.0])

        contributions = np.array([1.0, 1.0, 2.0])

        scaled_noise = fitting.generate_scaled_noise(noise, contributions, hyper_galaxies)

        assert (scaled_noise == noise).all()

    def test__x1_hyper_galaxy__noise_factor_and_power_are_1__scaled_noise_added_to_input_noise(self):
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=1.0)]

        noise = np.array([1.0, 1.0, 1.0])

        contributions = [np.array([1.0, 1.0, 0.5])]

        scaled_noise = fitting.generate_scaled_noise(noise, contributions, hyper_galaxies)

        assert (scaled_noise == np.array([2.0, 2.0, 1.5])).all()

    def test__x1_hyper_galaxy__noise_factor_1_and_power_is_2__scaled_noise_added_to_input_noise(self):
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0)]

        noise = np.array([1.0, 1.0, 1.0])

        contributions = [np.array([1.0, 1.0, 0.5])]

        scaled_noise = fitting.generate_scaled_noise(noise, contributions, hyper_galaxies)

        assert (scaled_noise == np.array([2.0, 2.0, 1.25])).all()

    def test__x2_hyper_galaxy__noise_factor_1_and_power_is_2__scaled_noise_added_to_input_noise(self):
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
                          MockHyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]

        noise = np.array([1.0, 1.0, 1.0])

        contributions = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]

        scaled_noise = fitting.generate_scaled_noise(noise, contributions, hyper_galaxies)

        assert (scaled_noise == np.array([2.5, 2.5, 1.75])).all()

    def test__x2_hyper_galaxy__same_as_above_but_use_real_hyper_galaxy(self):
        hyper_galaxies = [galaxy.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
                          galaxy.HyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]

        noise = np.array([1.0, 1.0, 1.0])

        contributions = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]

        scaled_noise = fitting.generate_scaled_noise(noise, contributions, hyper_galaxies)

        assert (scaled_noise == np.array([2.5, 2.5, 1.75])).all()


class TestLikelihood:

    def test__model_matches_data__noise_all_2s__lh_is_noise_term(self):
        image = grids.GridData(grid_data=np.array([10.0, 10.0, 10.0, 10.0]))
        noise = grids.GridData(grid_data=np.array([2.0, 2.0, 2.0, 2.0]))
        model_image = grids.GridData(grid_data=np.array([10.0, 10.0, 10.0, 10.0]))

        likelihood = fitting.compute_likelihood(image, noise, model_image)

        chi_sq_term = 0
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(
            2 * np.pi * 4.0)

        assert likelihood == -0.5 * (chi_sq_term + noise_term)

    def test__model_data_mismatch__chi_sq_term_contributes_to_lh(self):
        image = grids.GridData(grid_data=np.array([10.0, 10.0, 10.0, 10.0]))
        noise = grids.GridData(grid_data=np.array([2.0, 2.0, 2.0, 2.0]))
        model_image = grids.GridData(grid_data=np.array([11.0, 10.0, 9.0, 8.0]))

        likelihood = fitting.compute_likelihood(image, noise, model_image)

        # chi squared = 0.25, 0, 0.25, 1.0
        # likelihood = -0.5*(0.25+0+0.25+1.0)

        chi_sq_term = 1.5
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(
            2 * np.pi * 4.0)

        assert likelihood == -0.5 * (chi_sq_term + noise_term)

    def test__same_as_above_but_different_noise_in_each_pixel(self):
        image = grids.GridData(grid_data=np.array([10.0, 10.0, 10.0, 10.0]))
        noise = grids.GridData(grid_data=np.array([1.0, 2.0, 3.0, 4.0]))
        model_image = grids.GridData(grid_data=np.array([11.0, 10.0, 9.0, 8.0]))

        likelihood = fitting.compute_likelihood(image, noise, model_image)

        # chi squared = (1.0/1.0)**2, (0.0), (-1.0/3.0)**2.0, (2.0/4.0)**2.0

        chi_sq_term = 1.0 + (1.0 / 9.0) + 0.25
        noise_term = np.log(2 * np.pi * 1.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 9.0) + np.log(
            2 * np.pi * 16.0)

        assert likelihood == pytest.approx(-0.5 * (chi_sq_term + noise_term), 1e-4)


class TestComputeRegularizationTerm:

    def test__s_vector_all_1s__regularization_matrix_simple(self):
        s_vector = np.array([1.0, 1.0, 1.0])

        regularization_matrix = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 1.0]])

        # G_l term, Warren & Dye 2003 / Nightingale /2015 2018

        # G_l = s_T * H * s

        # Matrix multiplication:

        # s_T * H = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [(1.0*1.0) + (1.0*0.0) + (1.0*0.0)] = [1.0, 1.0, 1.0]
        #                             [1.0, 1.0, 1.0]   [(1.0*0.0) + (1.0*1.0) + (1.0*0.0)]
        #                             [1.0, 1.0, 1.0]   [(1.0*0.0) + (1.0*0.0) + (1.0*1.0)]

        # (s_T * H) * s = [1.0, 1.0, 1.0] * [1.0] = 3.0
        #                                   [1.0]
        #                                   [1.0]

        assert fitting.compute_regularization_term(s_vector, regularization_matrix) == 3.0

    def test__s_vector_and_regularization_matrix_range_of_values(self):
        s_vector = np.array([2.0, 3.0, 5.0])

        regularization_matrix = np.array([[2.0, -1.0, 0.0],
                                          [-1.0, 2.0, -1.0],
                                          [0.0, -1.0, 2.0]])

        # G_l term, Warren & Dye 2003 / Nightingale /2015 2018

        # G_l = s_T * H * s

        # Matrix multiplication:

        # s_T * H = [2.0, 3.0, 5.0] * [2.0,  -1.0,  0.0] = [(2.0* 2.0) + (3.0*-1.0) + (5.0 *0.0)] = [1.0, -1.0, 7.0]
        #                             [-1.0,  2.0, -1.0]   [(2.0*-1.0) + (3.0* 2.0) + (5.0*-1.0)]
        #                             [ 0.0, -1.0,  2.0]   [(2.0* 0.0) + (3.0*-1.0) + (5.0 *2.0)]

        # (s_T * H) * s = [1.0, -1.0, 7.0] * [2.0] = 34.0
        #                                    [3.0]
        #                                    [5.0]

        assert fitting.compute_regularization_term(s_vector, regularization_matrix) == 34.0


class TestLogDetMatrix:

    def test__determinant_of_ordinary_matrix(self):
        matrix = np.array([[2.0, -3.0, 1.0],
                           [2.0, 0.0, -1.0],
                           [1.0, 4.0, 5.0]])

        assert fitting.compute_log_determinant_of_matrix(matrix) == pytest.approx(np.log(49), 1e-4)

    def test__determinant_of_positive_definite_matrix_via_cholesky(self):
        matrix = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])

        log_determinant = np.log(np.linalg.det(matrix))

        assert log_determinant == pytest.approx(fitting.compute_log_determinant_of_matrix_cholesky(matrix), 1e-4)

    def test__determinant_of_positive_definite_matrix_2_via_cholesky(self):
        matrix = np.array([[2.0, -1.0, 0.0],
                           [-1.0, 2.0, -1.0],
                           [0.0, -1.0, 2.0]])

        log_determinant = np.log(np.linalg.det(matrix))

        assert log_determinant == pytest.approx(fitting.compute_log_determinant_of_matrix_cholesky(matrix), 1e-4)

    def test__determinant_of_positive_definite_matrix_compare_normal_and_cholesky_routines(self):
        matrix = np.array([[2.0, -1.0, 0.0],
                           [-1.0, 2.0, -1.0],
                           [0.0, -1.0, 2.0]])

        log_determinant = fitting.compute_log_determinant_of_matrix(matrix)
        log_determinant_chol = fitting.compute_log_determinant_of_matrix_cholesky(matrix)

        assert log_determinant == pytest.approx(log_determinant_chol, 1e-4)


class TestPixModelImageFromSVector:

    def test__s_vector_all_1s__simple_blurred_mapping_matrix__correct_model_image(self):
        s_vector = np.array([1.0, 1.0, 1.0, 1.0])

        blurred_mapping_matrix = np.array([[1.0, 1.0, 1.0, 1.0],
                                           [1.0, 0.0, 1.0, 1.0],
                                           [1.0, 0.0, 0.0, 0.0]])

        model_image = fitting.pixelization_model_image_from_s_vector(s_vector, blurred_mapping_matrix)

        # Image pixel 0 maps to 4 sources pixxels -> value is 4.0
        # Image pixel 1 maps to 3 sources pixxels -> value is 3.0
        # Image pixel 2 maps to 1 sources pixxels -> value is 1.0

        assert (model_image == np.array([4.0, 3.0, 1.0])).all()

    def test__s_vector_different_values__simple_blurred_mapping_matrix__correct_model_image(self):
        s_vector = np.array([1.0, 2.0, 3.0, 4.0])

        blurred_mapping_matrix = np.array([[1.0, 1.0, 1.0, 1.0],
                                           [1.0, 0.0, 1.0, 1.0],
                                           [1.0, 0.0, 0.0, 0.0]])

        model_image = fitting.pixelization_model_image_from_s_vector(s_vector, blurred_mapping_matrix)

        # Image pixel 0 maps to 4 sources pixxels -> value is 1.0 + 2.0 + 3.0 + 4.0 = 10.0
        # Image pixel 1 maps to 3 sources pixxels -> value is 1.0 + 3.0 + 4.0
        # Image pixel 2 maps to 1 sources pixxels -> value is 1.0

        assert (model_image == np.array([10.0, 8.0, 1.0])).all()


class TestBayesianEvidence:

    def test__simple_values(self):
        image = grids.GridData(grid_data=np.array([10.0, 10.0, 10.0, 10.0]))
        noise = grids.GridData(grid_data=np.array([2.0, 2.0, 2.0, 2.0]))
        model_image = grids.GridData(grid_data=np.array([10.0, 10.0, 10.0, 10.0]))

        s_vector = np.array([1.0, 1.0, 1.0])

        cov_reg_matrix = np.array([[2.0, -1.0, 0.0],
                                   [-1.0, 2.0, -1.0],
                                   [0.0, -1.0, 2.0]])

        reg_matrix = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])

        evidence = fitting.compute_bayesian_evidence(image, noise, model_image, s_vector, cov_reg_matrix, reg_matrix)

        chi_sq_term = 0
        reg_term = 3.0
        log_det_cov_reg = np.log(np.linalg.det(cov_reg_matrix))
        log_det_reg = np.log(np.linalg.det(reg_matrix))
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + np.log(
            2 * np.pi * 4.0)

        assert evidence == pytest.approx(-0.5 * (chi_sq_term + reg_term + log_det_cov_reg - log_det_reg + noise_term),
                                         1e-4)

    def test__complicated_values(self):
        image = grids.GridData(grid_data=np.array([10.0, 10.0, 10.0, 10.0]))
        noise = grids.GridData(grid_data=np.array([1.0, 2.0, 3.0, 4.0]))
        model_image = grids.GridData(grid_data=np.array([11.0, 10.0, 9.0, 8.0]))

        s_vector = np.array([2.0, 3.0, 5.0])

        cov_reg_matrix = np.array([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]])

        reg_matrix = np.array([[2.0, -1.0, 0.0],
                               [-1.0, 2.0, -1.0],
                               [0.0, -1.0, 2.0]])

        evidence = fitting.compute_bayesian_evidence(image, noise, model_image, s_vector, cov_reg_matrix, reg_matrix)

        chi_sq_term = 1.0 + (1.0 / 9.0) + 0.25
        reg_term = 34.0
        log_det_cov_reg = np.log(np.linalg.det(cov_reg_matrix))
        log_det_reg = np.log(np.linalg.det(reg_matrix))
        noise_term = np.log(2 * np.pi * 1.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 9.0) + np.log(
            2 * np.pi * 16.0)

        assert evidence == pytest.approx(-0.5 * (chi_sq_term + reg_term + log_det_cov_reg - log_det_reg + noise_term),
                                         1e-4)

    def test__use_fitting_functions_to_compute_terms(self):
        image = grids.GridData(grid_data=np.array([10.0, 100.0, 0.0, 10.0]))
        noise = grids.GridData(grid_data=np.array([1.0, 2.0, 77.0, 4.0]))
        model_image = grids.GridData(grid_data=np.array([11.0, 13.0, 9.0, 8.0]))

        s_vector = np.array([8.0, 7.0, 3.0])

        cov_reg_matrix = np.array([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]])

        reg_matrix = np.array([[2.0, -1.0, 0.0],
                               [-1.0, 2.0, -1.0],
                               [0.0, -1.0, 2.0]])

        evidence = fitting.compute_bayesian_evidence(image, noise, model_image, s_vector, cov_reg_matrix, reg_matrix)

        chi_sq_term = fitting.compute_chi_sq_term(image, noise, model_image)
        reg_term = fitting.compute_regularization_term(s_vector, reg_matrix)
        log_det_cov_reg = fitting.compute_log_determinant_of_matrix_cholesky(cov_reg_matrix)
        log_det_reg = fitting.compute_log_determinant_of_matrix_cholesky(reg_matrix)
        noise_term = fitting.compute_noise_term(noise)

        assert evidence == pytest.approx(-0.5 * (chi_sq_term + reg_term + log_det_cov_reg - log_det_reg + noise_term),
                                         1e-4)

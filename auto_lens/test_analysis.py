import numpy as np
import pytest

from auto_lens import analysis
from auto_lens import ray_tracing
from auto_lens.imaging import grids
from auto_lens.imaging import imaging
from auto_lens import galaxy
from auto_lens.profiles import mass_profiles, light_profiles

@pytest.fixture(scope='function')
def no_galaxies():
    return [galaxy.Galaxy()]

@pytest.fixture(scope='function')
def galaxy_light_sersic():
    sersic = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                             sersic_index=4.0)
    return galaxy.Galaxy(light_profiles=[sersic])


class TestLikelihood:

    def test__model_mathces_data__noise_all_2s__lh_is_noise_term(self):

        image = grids.GridData(grid_data=np.array([10.0, 10.0, 10.0, 10.0]))
        noise = grids.GridData(grid_data=np.array([2.0, 2.0, 2.0, 2.0]))
        model_image = grids.GridData(grid_data=np.array([10.0, 10.0, 10.0, 10.0]))

        likelihood = analysis.compute_likelihood(image, noise, model_image)

        chi_sq_term = 0
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0) + \
                     np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0)

        assert likelihood == -0.5 * (chi_sq_term + noise_term)

    def test__model_data_mismatch__chi_sq_term_contributes_to_lh(self):

        image = grids.GridData(grid_data=np.array([10.0, 10.0, 10.0, 10.0]))
        noise = grids.GridData(grid_data=np.array([2.0, 2.0, 2.0, 2.0]))
        model_image = grids.GridData(grid_data=np.array([11.0, 10.0, 9.0, 8.0]))

        likelihood = analysis.compute_likelihood(image, noise, model_image)

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

        likelihood = analysis.compute_likelihood(image, noise, model_image)

        # chi squared = (1.0/1.0)**2, (0.0), (-1.0/3.0)**2.0, (2.0/4.0)**2.0

        chi_sq_term = 1.0 + (1.0 / 9.0) + 0.25
        noise_term = np.log(2 * np.pi * 1.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 9.0) + np.log(
            2 * np.pi * 16.0)

        assert likelihood == pytest.approx(-0.5 * (chi_sq_term + noise_term), 1e-4)


class TestGenerateBlurredLightProfileImage:
    
    def test__simple_1_pixel_image__no_psf_blurring_into_mask_from_region(self, galaxy_light_sersic, no_galaxies):

        # The PSF the light profile image is convolved with

        psf = imaging.PSF(data=np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]]), pixel_scale=1.0)

        # Setup the Image and blurring masks

        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])
        mask = imaging.Mask(mask=mask, pixel_scale=1.0)

        # Setup the image and blurring coordinate grids

        grid_collection = grids.GridCoordsCollection.from_mask(mask=mask, blurring_size=psf.pixel_dimensions)
        grid_mappers = grids.GridMapperCollection.from_mask(mask=mask, blurring_size=psf.pixel_dimensions)

        ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[galaxy_light_sersic], source_galaxies=no_galaxies,
                                                    image_plane_grids=grid_collection)

        # For this PSF, the blurring region does not blur any flux into the central pixel.

        non_blurred_value = ray_trace.generate_image_of_galaxy_light_profiles()
        blurred_value = analysis.generate_blurred_light_profie_image(ray_tracing=ray_trace, psf=psf,
                                                                     grid_mappers=grid_mappers)

        assert non_blurred_value == blurred_value

    def test__simple_image_1_pixel__psf_all_1s_so_blurs_into_image(self, galaxy_light_sersic, no_galaxies):

        # The PSF the light profile image is convolved with

        psf = imaging.PSF(data=np.array([[1.0, 1.0, 1.0],
                                         [1.0, 1.0, 1.0],
                                         [1.0, 1.0, 1.0]]), pixel_scale=1.0)

        # Setup the Image and blurring masks

        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])
        mask = imaging.Mask(mask=mask, pixel_scale=1.0)

        # Setup the image and blurring coordinate grids

        grid_collection = grids.GridCoordsCollection.from_mask(mask=mask, blurring_size=psf.pixel_dimensions)
        grid_mappers = grids.GridMapperCollection.from_mask(mask=mask, blurring_size=psf.pixel_dimensions)

        ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[galaxy_light_sersic], source_galaxies=no_galaxies,
                                                    image_plane_grids=grid_collection)

        blurred_value = analysis.generate_blurred_light_profie_image(ray_tracing=ray_trace, psf=psf,
                                                                     grid_mappers=grid_mappers)

        # Manually compute result of convolution, which for our PSF of all 1's is just the central value +
        # the (central value x each blurring region value).

        central_value = ray_trace.generate_image_of_galaxy_light_profiles()
        blurring_values = ray_trace.generate_blurring_image_of_galaxy_light_profiles()
        blurred_value_manual = sum(blurring_values[:]) + central_value

        assert blurred_value[0] == pytest.approx(blurred_value_manual[0], 1e-6)

    def test__image_is_2x2__psf_is_non_symmetric_l_shape(self, galaxy_light_sersic, no_galaxies):

        # The PSF the light profile image is convolved with

        psf = imaging.PSF(data=np.array([[0.0, 3.0, 0.0],
                                         [0.0, 2.0, 1.0],
                                         [0.0, 0.0, 0.0]]), pixel_scale=1.0)

        # Setup the Image and blurring masks

        mask = np.array([[True, True,  True,  True],
                         [True, False, False, True],
                         [True, False, False, True],
                         [True, True,  True,  True]])
        mask = imaging.Mask(mask=mask, pixel_scale=1.0)

        # Setup the image and blurring coordinate grids

        grid_collection = grids.GridCoordsCollection.from_mask(mask=mask, blurring_size=psf.pixel_dimensions)
        grid_mappers = grids.GridMapperCollection.from_mask(mask=mask, blurring_size=psf.pixel_dimensions)

        ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[galaxy_light_sersic], source_galaxies=no_galaxies,
                                                    image_plane_grids=grid_collection)

        blurred_value = analysis.generate_blurred_light_profie_image(ray_tracing=ray_trace, psf=psf,
                                                                     grid_mappers=grid_mappers)

        # Manually compute result of convolution, which is each central value *2.0 plus its 2 appropriate neighbors

        central_values = ray_trace.generate_image_of_galaxy_light_profiles()
        blurring_values = ray_trace.generate_blurring_image_of_galaxy_light_profiles()

        blurred_value_manual_0 = 2.0*central_values[0] + 3.0*central_values[2] + blurring_values[4]
        blurred_value_manual_1 = 2.0*central_values[1] + 3.0*central_values[3] + central_values[0]
        blurred_value_manual_2 = 2.0*central_values[2] + 3.0*blurring_values[9] + blurring_values[6]
        blurred_value_manual_3 = 2.0*central_values[3] + 3.0*blurring_values[10] + central_values[2]

        assert blurred_value_manual_0 == pytest.approx(blurred_value[0], 1e-6)
        assert blurred_value_manual_1 == pytest.approx(blurred_value[1], 1e-6)
        assert blurred_value_manual_2 == pytest.approx(blurred_value[2], 1e-6)
        assert blurred_value_manual_3 == pytest.approx(blurred_value[3], 1e-6)

    def test__same_as_above_blurring_region_off__blurring_only_uses_mask_pixels(self, galaxy_light_sersic, no_galaxies):

        # The PSF the light profile image is convolved with

        psf = imaging.PSF(data=np.array([[0.0, 3.0, 0.0],
                                         [0.0, 2.0, 1.0],
                                         [0.0, 0.0, 0.0]]), pixel_scale=1.0)

        # Setup the Image and blurring masks

        mask = np.array([[True, True,  True,  True],
                         [True, False, False, True],
                         [True, False, False, True],
                         [True, True,  True,  True]])
        mask = imaging.Mask(mask=mask, pixel_scale=1.0)

        # Setup the image and blurring coordinate grids

        grid_collection = grids.GridCoordsCollection.from_mask(mask=mask)
        grid_mappers = grids.GridMapperCollection.from_mask(mask=mask)

        ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[galaxy_light_sersic], source_galaxies=no_galaxies,
                                                    image_plane_grids=grid_collection)

        blurred_value = analysis.generate_blurred_light_profie_image(ray_tracing=ray_trace, psf=psf,
                                                                     grid_mappers=grid_mappers)

        # Manually compute result of convolution, which is each central value *2.0 plus its 2 appropriate neighbors

        central_values = ray_trace.generate_image_of_galaxy_light_profiles()

        blurred_value_manual_0 = 2.0*central_values[0] + 3.0*central_values[2]
        blurred_value_manual_1 = 2.0*central_values[1] + 3.0*central_values[3] + central_values[0]
        blurred_value_manual_2 = 2.0*central_values[2]
        blurred_value_manual_3 = 2.0*central_values[3] + central_values[2]

        assert blurred_value_manual_0 == pytest.approx(blurred_value[0], 1e-6)
        assert blurred_value_manual_1 == pytest.approx(blurred_value[1], 1e-6)
        assert blurred_value_manual_2 == pytest.approx(blurred_value[2], 1e-6)
        assert blurred_value_manual_3 == pytest.approx(blurred_value[3], 1e-6)


class TestComputeBlurredImages:

    def test__psf_just_central_1_so_no_blurring__no_blurring_region__image_in_is_image_out(self):
        
        image_2d = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])

        image_mask = np.array([[True, True, True, True],
                               [True, False, False, True],
                               [True, False, False, True],
                               [True, True, True, True]])

        image_mask = imaging.Mask(mask=image_mask, pixel_scale=1.0)
        
        image = grids.GridData.from_mask(data=image_2d, mask=image_mask)

        image_to_pixel = grids.GridMapperDataToPixel.from_mask(image_mask)
        
        psf = np.array([[0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0]])

        psf = imaging.PSF(data=psf, pixel_scale=1.0)

        blurred_image = analysis.blur_image_including_blurring_region(image, image_to_pixel, psf)

        assert (blurred_image == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__psf_all_1s_so_blurring_gives_4s__no_blurring_region__image_in_is_image_out(self):

        image_2d = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])

        image_mask = np.array([[True, True, True, True],
                               [True, False, False, True],
                               [True, False, False, True],
                               [True, True, True, True]])

        image_mask = imaging.Mask(mask=image_mask, pixel_scale=1.0)

        image = grids.GridData.from_mask(data=image_2d, mask=image_mask)

        image_to_pixel = grids.GridMapperDataToPixel.from_mask(image_mask)

        psf = np.array([[1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0]])

        psf = imaging.PSF(data=psf, pixel_scale=1.0)

        blurred_image = analysis.blur_image_including_blurring_region(image, image_to_pixel, psf)

        assert (blurred_image == np.array([4.0, 4.0, 4.0, 4.0])).all()

    def test__psf_just_central_1__include_blurring_regionb_blurring_region_not_blurred_in_so_return_image(self):
        image_2d = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])
        image_mask = np.array([[True, True, True, True],
                               [True, False, False, True],
                               [True, False, False, True],
                               [True, True, True, True]])
        image_mask = imaging.Mask(mask=image_mask, pixel_scale=1.0)
        image = grids.GridData.from_mask(data=image_2d, mask=image_mask)
        image_to_pixel = grids.GridMapperDataToPixel.from_mask(image_mask)

        psf = np.array([[0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0]])
        psf = imaging.PSF(data=psf, pixel_scale=1.0)

        blurring_mask = np.array([[False, False, False, False],
                                  [False, True, True, False],
                                  [False, True, True, False],
                                  [False, False, False, False]])
        blurring_mask = imaging.Mask(mask=blurring_mask, pixel_scale=1.0)
        blurring_image = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        blurring_to_pixel = grids.GridMapperDataToPixel.from_mask(blurring_mask)

        blurred_image = analysis.blur_image_including_blurring_region(image, image_to_pixel, psf, blurring_image,
                                                                      blurring_to_pixel)

        assert (blurred_image == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__psf_all_1s__include_blurring_region_image_turns_to_9s(self):
        image_2d = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])
        image_mask = np.array([[True, True, True, True],
                               [True, False, False, True],
                               [True, False, False, True],
                               [True, True, True, True]])
        image_mask = imaging.Mask(mask=image_mask, pixel_scale=1.0)
        image = grids.GridData.from_mask(data=image_2d, mask=image_mask)
        image_to_pixel = grids.GridMapperDataToPixel.from_mask(image_mask)

        psf = np.array([[1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0]])
        psf = imaging.PSF(data=psf, pixel_scale=1.0)

        blurring_mask = np.array([[False, False, False, False],
                                  [False, True, True, False],
                                  [False, True, True, False],
                                  [False, False, False, False]])
        blurring_mask = imaging.Mask(mask=blurring_mask, pixel_scale=1.0)
        blurring_image = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        blurring_to_pixel = grids.GridMapperDataToPixel.from_mask(blurring_mask)

        blurred_image = analysis.blur_image_including_blurring_region(image, image_to_pixel, psf, blurring_image,
                                                                      blurring_to_pixel)

        assert (blurred_image == np.array([9.0, 9.0, 9.0, 9.0])).all()
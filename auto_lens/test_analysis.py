import numpy as np
import pytest

from auto_lens import analysis
from auto_lens import ray_tracing
from auto_lens.imaging import grids
from auto_lens.imaging import imaging
from auto_lens import galaxy
from auto_lens.profiles import mass_profiles, light_profiles


class TestGenerateBlurredLightProfileImage:
    
    def test__image_is_1_central_pixel__psf_is_1_central_pixel_value_1__blurred_image_is_image(self):

        # The PSF the light profile image is convolved with

        psf = imaging.PSF(data=np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]]), pixel_scale=1.0)

        # Setup the Image and blurring masks

        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])
        mask = imaging.Mask(mask=mask, pixel_scale=1.0)
        blurring_mask = mask.compute_blurring_mask(psf_size=psf.pixel_dimensions)

        # Setup the image and blurring coordinate grids

        grid_collection = grids.GridCoordsCollection.from_mask(mask=mask, blurring_size=psf.pixel_dimensions)

        # Setup the GridMappers

        image_to_pixel = grids.GridMapperDataToPixel.from_mask(mask)
        blurring_to_pixel = grids.GridMapperDataToPixel.from_mask(blurring_mask)
        grid_mappers = grids.GridMapperCollection(image_to_pixel=image_to_pixel, blurring_to_pixel=blurring_to_pixel)

        #Setup the Ray Tracing as a single Sersic profile galaxy

        sersic = light_profiles.EllipticalSersic()
        lens_galaxy = galaxy.Galaxy(light_profiles=[sersic])
        
        ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_galaxy], source_galaxies=[galaxy.Galaxy()], 
                                                    image_plane_grids=grid_collection)

        non_blurred_value = ray_trace.generate_image_of_galaxy_light_profiles()
        blurred_value = analysis.generate_blurred_light_profie_image(ray_tracing=ray_trace, psf=psf,
                                                                     grid_mappers=grid_mappers)

        assert non_blurred_value == blurred_value

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
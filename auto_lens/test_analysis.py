import numpy as np
import pytest

from auto_lens import analysis
from auto_lens.imaging import grids
from auto_lens.imaging import imaging
from auto_lens import galaxy
from auto_lens.profiles import mass_profiles, light_profiles


@pytest.fixture(scope='function')
def grid_datas_and_mappers():
    mask = np.array([[False, False, False],
                     [False, False, False],
                     [False, False, False]])

    mask = imaging.Mask(mask=mask, pixel_scale=1.0)

    image = grids.GridData.from_mask(data=np.array([[0.0, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.0]]), mask=mask)

    noise = grids.GridData.from_mask(data=np.array([[0.0, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.0]]), mask=mask)

    exposure_time = grids.GridData.from_mask(data=np.array([[1.0, 1.0, 1.0],
                                                            [1.0, 1.0, 1.0],
                                                            [1.0, 1.0, 1.0]]), mask=mask)

    psf = imaging.PSF(data=np.array([[0.0, 1.0, 0.0],
                                     [1.0, 2.0, 1.0],
                                     [0.0, 1.0, 0.0]]), pixel_scale=1.0)

    grid_datas = grids.GridDataCollection(image=image, noise=noise, exposure_time=exposure_time, psf=psf)

    mapper_to_2d = grids.GridMapperDataToPixel.from_mask(mask)
    grid_mappers = grids.GridMapperCollection(data_to_pixels=mapper_to_2d)

    return grid_datas, grid_mappers


@pytest.fixture(scope='function')
def ray_tracing():
    sersic = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                             sersic_index=4.0)
    sis = mass_profiles.SphericalIsothermal(einstein_radius=1.0)

    lens_galaxy = galaxy.Galaxy(light_profiles=[sersic], mass_profiles=[sis])
    source_galaxy = galaxy.Galaxy(light_profiles=[sersic])

    ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                image_plane_grids=grid_image)



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

        blurred_image = analysis.compute_blurred_light_profile_image(image, image_to_pixel, psf)

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

        blurred_image = analysis.compute_blurred_light_profile_image(image, image_to_pixel, psf)

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

        blurred_image = analysis.compute_blurred_light_profile_image(image, image_to_pixel, psf, blurring_image,
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

        blurred_image = analysis.compute_blurred_light_profile_image(image, image_to_pixel, psf, blurring_image,
                                                                     blurring_to_pixel)

        assert (blurred_image == np.array([9.0, 9.0, 9.0, 9.0])).all()
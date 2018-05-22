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

    def test__simple_image_and_psf_in__correct_blurred_image_generated(self):

        mask = np.array([[False, False, False],
                         [False, False, False],
                         [False, False, False]])

        mask = imaging.Mask(mask=mask, pixel_scale=1.0)

        image = grids.GridData.from_mask(data=np.array([[0.0, 0.0, 0.0],
                                                        [0.0, 1.0, 0.0],
                                                        [0.0, 0.0, 0.0]]), mask=mask)

        psf = imaging.PSF(data=np.array([[0.0, 1.0, 0.0],
                                         [1.0, 2.0, 1.0],
                                         [0.0, 1.0, 0.0]]), pixel_scale=1.0)

        pixel_mapper = grids.GridMapperDataToPixel.from_mask(mask)

        blurred_image = analysis.compute_blurred_light_profile_image(image, psf, pixel_mapper)

        # In 2D the blurred image should be
        # [[0.0, 1.0, 0.0],
        #  [1.0, 2.0, 1.0],
        #  [0.0, 1.0, 0.0]])

        assert (blurred_image == np.array([0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0])).all()



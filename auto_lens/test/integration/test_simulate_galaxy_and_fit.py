from auto_lens.imaging import scaled_array
from auto_lens.imaging import image
from auto_lens.imaging import mask
from auto_lens.imaging import grids
from auto_lens.profiles import light_profiles as lp
from auto_lens.profiles import mass_profiles as mp
from auto_lens.pixelization import frame_convolution
from auto_lens.analysis import fitting
from auto_lens.analysis import ray_tracing
from auto_lens.analysis import galaxy

import numpy as np
import pytest

@pytest.fixture(name='sim_grid', scope='function')
def sim_grid():

    sim_grid.ma = np.ma.array([[True, True, True, True, True, True, True],
                      [True, False, False, False, False, False, True],
                      [True, False, False, False, False, False, True],
                      [True, False, False, False, False, False, True],
                      [True, False, False, False, False, False, True],
                      [True, False, False, False, False, False, True],
                      [True, True, True, True, True, True, True]])

    sim_grid.ma = mask.Mask(array=sim_grid.ma, pixel_scale=1.0)
    sim_grid.image_grid = grids.GridCoordsCollection.from_mask(mask=sim_grid.ma, grid_size_sub=1, blurring_size=(3, 3))
    sim_grid.mappers = grids.GridMapperCollection.from_mask(mask=sim_grid.ma)
    return sim_grid

@pytest.fixture(name='fit_grid', scope='function')
def fit_grid():

    fit_grid.ma = np.ma.array([[True, True, True, True, True, True, True],
                               [True, True, True, True, True, True, True],
                               [True, True, False, False, False, True, True],
                               [True, True, False, False, False, True, True],
                               [True, True, False, False, False, True, True],
                               [True, True, True, True, True, True, True],
                               [True, True, True, True, True, True, True]])

    fit_grid.ma = mask.Mask(array=fit_grid.ma, pixel_scale=1.0)
    fit_grid.image_grid = grids.GridCoordsCollection.from_mask(mask=fit_grid.ma, grid_size_sub=1, blurring_size=(3, 3))
    fit_grid.mappers = grids.GridMapperCollection.from_mask(mask=fit_grid.ma)
    return fit_grid

@pytest.fixture(scope='function')
def galaxy_mass_sis():
    sis = mp.SphericalIsothermal(einstein_radius=1.0)
    return galaxy.Galaxy(mass_profiles=[sis])


@pytest.fixture(scope='function')
def galaxy_light_sersic():
    sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                             sersic_index=4.0)
    return galaxy.Galaxy(light_profiles=[sersic])


class TestCase:

    def test_simulate_1_galaxy_no_instrumental_effects__fit_with_self__likelihood_is_noise_term(self, sim_grid,
                                                                                                fit_grid,
                                                                                                galaxy_light_sersic):
        ### Simulate Image ###

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic], source_galaxies=[],
                                       image_plane_grids=sim_grid.image_grid)

        galaxy_image_1d = ray_trace.generate_image_of_galaxy_light_profiles()
        galaxy_image_2d = sim_grid.mappers.data_to_pixel.map_to_2d(galaxy_image_1d)

        sim_image = image.Image.simulate(array=galaxy_image_2d)

        ### Fit Image With Same Model ###

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic], source_galaxies=[],
                                       image_plane_grids=fit_grid.image_grid)

        fit_grid_datas = grids.GridDataCollection.from_mask(mask=fit_grid.ma, image=sim_image, noise=np.ones((7,7)),
                                                            exposure_time=np.ones((7,7)))

        frame = frame_convolution.FrameMaker(mask=fit_grid.ma)
        convolver = frame.convolver_for_kernel_shape(kernel_shape=(3,3),
                                             blurring_region_mask=fit_grid.ma.compute_blurring_mask(kernal_shape=(3, 3)))
        # This PSF leads to no blurring, so equivalent to being off.
        kernel_convolver = convolver.convolver_for_kernel(kernel=np.array([[0., 0., 0.],
                                                                           [0., 1., 0.],
                                                                           [0., 0., 0.]]))

        likelihood = fitting.fit_data_with_model(grid_datas=fit_grid_datas, grid_mappers=fit_grid.mappers,
                                                 kernel_convolver=kernel_convolver, tracer=ray_trace)

        assert likelihood == -0.5*9.0*np.log(2 * np.pi * 1.0 ** 2.0)

    def test_simulate_1_galaxy_with_psf__fit_to_self__likelihood_is_noise_term(self, sim_grid, fit_grid,
                                                                               galaxy_light_sersic):
        
        ### Simulate image ###

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic], source_galaxies=[], 
                                       image_plane_grids=sim_grid.image_grid)

        galaxy_image_1d = ray_trace.generate_image_of_galaxy_light_profiles()
        galaxy_image_2d = sim_grid.mappers.data_to_pixel.map_to_2d(galaxy_image_1d)

        sim_psf = image.PSF(np.array([[1., 2., 3.],
                                      [4., 5., 6.],
                                      [7., 8., 9.]]), pixel_scale=1.0)
        sim_image = image.Image.simulate(array=galaxy_image_2d, psf=sim_psf)

        ### Fit Image With Same Model ###

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic], source_galaxies=[],
                                       image_plane_grids=fit_grid.image_grid)

        fit_grid_datas = grids.GridDataCollection.from_mask(mask=fit_grid.ma, image=sim_image, noise=np.ones((7,7)),
                                                            exposure_time=np.ones((7,7)))

        frame = frame_convolution.FrameMaker(mask=fit_grid.ma)
        convolver = frame.convolver_for_kernel_shape(kernel_shape=(3,3),
                                             blurring_region_mask=fit_grid.ma.compute_blurring_mask(kernal_shape=(3, 3)))
        kernel_convolver = convolver.convolver_for_kernel(kernel=sim_psf)

        likelihood = fitting.fit_data_with_model(grid_datas=fit_grid_datas, grid_mappers=fit_grid.mappers,
                                                 kernel_convolver=kernel_convolver, tracer=ray_trace)

        assert likelihood == -0.5*9.0*np.log(2 * np.pi * 1.0 ** 2.0)

    def test_simulate_lots_of_galaxies_no_instrumental_effects__fit_with_self__likelihood_is_noise_term(self, sim_grid,
                                                                                                fit_grid,
                                                                                                galaxy_mass_sis,
                                                                                                galaxy_light_sersic):
        ### Simulate Image ###

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic, galaxy_light_sersic, galaxy_mass_sis],
                                       source_galaxies=[galaxy_light_sersic, galaxy_light_sersic],
                                       image_plane_grids=sim_grid.image_grid)

        galaxy_image_1d = ray_trace.generate_image_of_galaxy_light_profiles()
        galaxy_image_2d = sim_grid.mappers.data_to_pixel.map_to_2d(galaxy_image_1d)

        sim_image = image.Image.simulate(array=galaxy_image_2d)

        ### Fit Image With Same Model ###

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_sersic, galaxy_light_sersic, galaxy_mass_sis],
                                       source_galaxies=[galaxy_light_sersic, galaxy_light_sersic],
                                       image_plane_grids=fit_grid.image_grid)

        fit_grid_datas = grids.GridDataCollection.from_mask(mask=fit_grid.ma, image=sim_image, noise=np.ones((7,7)),
                                                            exposure_time=np.ones((7,7)))

        frame = frame_convolution.FrameMaker(mask=fit_grid.ma)
        convolver = frame.convolver_for_kernel_shape(kernel_shape=(3,3),
                                             blurring_region_mask=fit_grid.ma.compute_blurring_mask(kernal_shape=(3, 3)))
        # This PSF leads to no blurring, so equivalent to being off.
        kernel_convolver = convolver.convolver_for_kernel(kernel=np.array([[0., 0., 0.],
                                                                           [0., 1., 0.],
                                                                           [0., 0., 0.]]))

        likelihood = fitting.fit_data_with_model(grid_datas=fit_grid_datas, grid_mappers=fit_grid.mappers,
                                                 kernel_convolver=kernel_convolver, tracer=ray_trace)

        assert likelihood == -0.5*9.0*np.log(2 * np.pi * 1.0 ** 2.0)
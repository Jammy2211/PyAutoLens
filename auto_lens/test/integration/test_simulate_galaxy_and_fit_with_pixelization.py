from auto_lens.imaging import scaled_array
from auto_lens.imaging import image
from auto_lens.imaging import mask
from auto_lens.imaging import grids
from auto_lens.profiles import light_profiles as lp
from auto_lens.profiles import mass_profiles as mp
from auto_lens.pixelization import pixelization
from auto_lens.pixelization import frame_convolution
from auto_lens.analysis import fitting
from auto_lens.analysis import ray_tracing
from auto_lens.analysis import galaxy

import numpy as np
import pytest

# TODO : Still suffer border issues described in profile integration test

@pytest.fixture(name='sim_grid_9x9', scope='function')
def sim_grid_9x9():
    sim_grid_9x9.ma = mask.Mask.for_simulate(shape_arc_seconds=(5.5, 5.5), pixel_scale=0.5, psf_size=(3, 3))
    sim_grid_9x9.image_grid = grids.GridCoordsCollection.from_mask(mask=sim_grid_9x9.ma, grid_size_sub=1,
                                                                     blurring_size=(3, 3))
    sim_grid_9x9.mappers = grids.GridMapperCollection.from_mask(mask=sim_grid_9x9.ma)
    return sim_grid_9x9

@pytest.fixture(name='fit_grid_9x9', scope='function')
def fit_grid_9x9():
    fit_grid_9x9.ma = mask.Mask.for_simulate(shape_arc_seconds=(4.5, 4.5), pixel_scale=0.5, psf_size=(5, 5))
    fit_grid_9x9.ma = mask.Mask(array=fit_grid_9x9.ma, pixel_scale=1.0)
    fit_grid_9x9.image_grid = grids.GridCoordsCollection.from_mask(mask=fit_grid_9x9.ma, grid_size_sub=2,
                                                                     blurring_size=(3, 3))
    fit_grid_9x9.mappers = grids.GridMapperCollection.from_mask(mask=fit_grid_9x9.ma)
    return fit_grid_9x9

@pytest.fixture(scope='function')
def galaxy_mass_sis():
    sis = mp.SphericalIsothermal(einstein_radius=1.0)
    return galaxy.Galaxy(mass_profiles=[sis])


@pytest.fixture(scope='function')
def galaxy_light_sersic():
    sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=2.0,
                                             sersic_index=1.0)
    return galaxy.Galaxy(light_profiles=[sersic])


class TestCase:

    def test_sim_1_galaxy_no_instrumental_effects__fit_cluster_pixelization__likelihood_near_noise_term(self, sim_grid_9x9,
                                                                                                        fit_grid_9x9,
                                                                                                        galaxy_mass_sis,
                                                                                                        galaxy_light_sersic):
        ### Simulate Image ###

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_mass_sis], source_galaxies=[galaxy_light_sersic],
                                       image_plane_grids=sim_grid_9x9.image_grid)

        galaxy_image_1d = ray_trace.generate_image_of_galaxy_light_profiles()
        galaxy_image_2d = sim_grid_9x9.mappers.data_to_pixel.map_to_2d(galaxy_image_1d)

        sim_image = image.Image.simulate(array=galaxy_image_2d)

        ### Fit Image Cluster Pixelization ###

        ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_mass_sis], source_galaxies=[],
                                       image_plane_grids=fit_grid_9x9.image_grid)

        fit_grid_datas = grids.GridDataCollection.from_mask(mask=fit_grid_9x9.ma, image=sim_image,
                                                            noise=np.ones((19,19)),
                                                            exposure_time=np.ones((19,19)))

        mapper_cluster = grids.GridMapperCluster.from_mask(mask=fit_grid_9x9.ma, cluster_grid_size=3)

        pix = pixelization.ClusterPixelization(pixels=len(mapper_cluster.cluster_to_image),
                                               regularization_coefficients=(0.0,))

        frame = frame_convolution.FrameMaker(mask=fit_grid_9x9.ma)
        convolver = frame.convolver_for_kernel_shape(kernel_shape=(3,3))
        # This PSF leads to no blurring, so equivalent to being off.
        kernel_convolver = convolver.convolver_for_kernel(kernel=np.array([[0.,    0.125, 0.],
                                                                           [0.125, 0.5,   0.125],
                                                                           [0.,    0.125, 0.]]))

        model_image = fitting.fit_data_with_pixelization(grid_data=fit_grid_datas, pix=pix,
                                                 kernel_convolver=kernel_convolver, tracer=ray_trace,
                                                 mapper_cluster=mapper_cluster)

        for i in range(model_image.shape[0]):
            print(i, fit_grid_datas.image[i], model_image[i], fit_grid_datas.image[i] - model_image[i])

        print(np.sum(fit_grid_datas.image - model_image))

     #   assert likelihood == -0.5*9.0*np.log(2 * np.pi * 1.0 ** 2.0)
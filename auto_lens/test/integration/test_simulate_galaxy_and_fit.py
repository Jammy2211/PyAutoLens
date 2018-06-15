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

class TestCase:

    def test_simulate_1_galaxy_no_instrumental_effects__fit_with_self__likelihood_is_noise_term(self):

        ### Simulate Image ###

        ### Setup mask + grid of this image ###

        ma = np.ma.array([[True,  True,  True,  True, True],
                          [True, False, False, False, True],
                          [True, False, False, False, True],
                          [True, False, False, False, True],
                          [True,  True,  True,  True, True]])

        ma = mask.Mask(array=ma, pixel_scale=1.0)
        image_grids = grids.GridCoordsCollection.from_mask(mask=ma, grid_size_sub=1, blurring_size=(3,3))
        mappers = grids.GridMapperCollection.from_mask(mask=ma, blurring_size=(3,3))

        ### Setup the ray tracing model, and use to generate the 2D galaxy image ###

        gal = galaxy.Galaxy(light_profiles=[lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.7,
                            phi=45.0, intensity=0.1, effective_radius=0.5, sersic_index=1.0)])

        ray_trace = ray_tracing.Tracer(lens_galaxies=[gal], source_galaxies=[], image_plane_grids=image_grids)

        grid_galaxy_image = ray_trace.generate_image_of_galaxy_light_profiles()
        galaxy_image = mappers.data_to_pixel.map_to_2d(grid_galaxy_image)

        ### Setup the image as an image.

        sim_image = image.Image.simulate(array=galaxy_image)

        ### Fit Image With Same Model ###

        sim_grid_datas = grids.GridDataCollection.from_mask(mask=ma, image=sim_image, noise=np.ones((5,5)),
                                                            exposure_time=np.ones((5,5)))

        frame = frame_convolution.FrameMaker(mask=ma)
        convolver = frame.convolver_for_kernel_shape(kernel_shape=(3,3),
                                             blurring_region_mask=ma.compute_blurring_mask(kernal_shape=(3, 3)))
        # This PSF leads to no blurring, so equivalent to being off.
        kernel_convolver = convolver.convolver_for_kernel(kernel=np.array([[0., 0., 0.],
                                                                           [0., 1., 0.],
                                                                           [0., 0., 0.]]))

        likelihood = fitting.fit_data_with_model(grid_datas=sim_grid_datas, grid_mappers=mappers,
                                                 kernel_convolver=kernel_convolver, tracer=ray_trace)

        assert likelihood == -0.5*9.0*np.log(2 * np.pi * 1.0 ** 2.0)

    # def test_simulate_1_galaxy_psf_in__fit_with_self__likelihood_is_noise_term(self):
    #
    #     ### Simulate Image ###
    #
    #     ### Setup mask + grid of this image ###
    #
    #     ma = np.ma.array([[True,  True,  True,  True, True],
    #                       [True, False, False, False, True],
    #                       [True, False, False, False, True],
    #                       [True, False, False, False, True],
    #                       [True,  True,  True,  True, True]])
    #
    #     ma = mask.Mask(array=ma, pixel_scale=1.0)
    #     image_grids = grids.GridCoordsCollection.from_mask(mask=ma, grid_size_sub=1, blurring_size=(3,3))
    #     mappers = grids.GridMapperCollection.from_mask(mask=ma, blurring_size=(3,3))
    #
    #     ### Setup the ray tracing model, and use to generate the 2D galaxy image ###
    #
    #     gal = galaxy.Galaxy(light_profiles=[lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.7,
    #                         phi=45.0, intensity=0.1, effective_radius=0.5, sersic_index=1.0)])
    #
    #     ray_trace = ray_tracing.Tracer(lens_galaxies=[gal], source_galaxies=[], image_plane_grids=image_grids)
    #
    #     grid_galaxy_image = ray_trace.generate_image_of_galaxy_light_profiles()
    #     galaxy_image = mappers.data_to_pixel.map_to_2d(grid_galaxy_image)
    #
    #     ### Setup the image as an image.
    #
    #     sim_psf = image.PSF(np.array([[0., 0., 0.],
    #                                   [1., 1., 1.],
    #                                   [0., 0., 0.]]), pixel_scale=1.0)
    #     sim_image = image.Image.simulate(array=galaxy_image, psf=sim_psf)
    #
    #     ### Fit Image With Same Model ###
    #
    #     sim_grid_datas = grids.GridDataCollection.from_mask(mask=ma, image=sim_image, noise=np.ones((5,5)),
    #                                                         exposure_time=np.ones((5,5)))
    #
    #     frame = frame_convolution.FrameMaker(mask=ma)
    #     convolver = frame.convolver_for_kernel_shape(kernel_shape=(3,3),
    #                                          blurring_region_mask=ma.compute_blurring_mask(kernal_shape=(3, 3)))
    #     kernel_convolver = convolver.convolver_for_kernel(kernel=sim_psf)
    #
    #     likelihood = fitting.fit_data_with_model(grid_datas=sim_grid_datas, grid_mappers=mappers,
    #                                              kernel_convolver=kernel_convolver, tracer=ray_trace)
    #
    #     print(likelihood)
    #
    #     assert likelihood == -0.5*9.0*np.log(2 * np.pi * 1.0 ** 2.0)
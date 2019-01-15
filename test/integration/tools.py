import os
import shutil

import numpy as np

from autolens.data import ccd
from autolens.data.array.util import array_util
from autolens.data.array import grids, scaled_array
from autolens.lens import ray_tracing

dirpath = os.path.dirname(os.path.realpath(__file__))

def reset_paths(test_name, output_path):

    try:
        shutil.rmtree(dirpath + '/hyper/' + test_name)
    except FileNotFoundError:
        pass

    try:
        shutil.rmtree(output_path + '/' + test_name)
    except FileNotFoundError:
        pass

def simulate_integration_image(test_name, pixel_scale, lens_galaxies, source_galaxies, target_signal_to_noise):
    
    output_path = "{}/hyper/".format(os.path.dirname(os.path.realpath(__file__))) + test_name + '/'
    psf_shape = (11, 11)
    image_shape = (150, 150)

    psf = ccd.PSF.simulate_as_gaussian(shape=psf_shape, pixel_scale=pixel_scale, sigma=pixel_scale)

    grid_stack = grids.GridStack.grid_stack_for_simulation(shape=image_shape, pixel_scale=pixel_scale,
                                                            sub_grid_size=1, psf_shape=psf_shape)

    image_shape = grid_stack.regular.padded_shape

    if not source_galaxies:

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=lens_galaxies, image_plane_grid_stack=grid_stack)

    elif source_galaxies:

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=lens_galaxies, source_galaxies=source_galaxies,
                                                     image_plane_grid_stack=grid_stack)

    ### Setup as a simulated image_coords and output as a fits for an lensing ###

    ccd_simulated = ccd.CCDData.simulate_to_target_signal_to_noise(array=tracer.image_plane_image_for_simulation,
                                                                   pixel_scale=pixel_scale,
                                                                   target_signal_to_noise=target_signal_to_noise,
                                                                   exposure_time_map=np.ones(image_shape),
                                                                   background_sky_map=10.0 * np.ones(image_shape),
                                                                   psf=psf, seed=1)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    array_util.numpy_array_to_fits(array=ccd_simulated.image, path=output_path + '/image.fits')
    array_util.numpy_array_to_fits(array=ccd_simulated.noise_map, path=output_path + '/noise_map.fits')
    array_util.numpy_array_to_fits(array=psf, path=output_path + '/psf.fits')
    array_util.numpy_array_to_fits(array=ccd_simulated.exposure_time_map, path=output_path + '/exposure_map.fits')


def load_image(test_name, pixel_scale):

    data_dir = "{}/hyper/{}".format(dirpath, test_name)
    data = scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=data_dir + '/image.fits', hdu=0,
                                                                          pixel_scale=pixel_scale)
    noise = scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=data_dir + '/noise_map.fits', hdu=0,
                                                                           pixel_scale=pixel_scale)
    psf = ccd.PSF.from_fits_with_scale(file_path=data_dir + '/psf.fits', hdu=0, pixel_scale=pixel_scale)

    return ccd.CCDData(image=data, pixel_scale=pixel_scale, psf=psf, noise_map=noise)

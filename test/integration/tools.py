import os
import shutil

import numpy as np

from autofit import conf
from autolens.data.imaging import image as im
from autolens.data.array.util import array_util
from autolens.data.array import grids, scaled_array
from autolens.lensing import ray_tracing

dirpath = os.path.dirname(os.path.realpath(__file__))

def reset_paths(data_name, pipeline_name, output_path):

    conf.instance.output_path = output_path

    try:
        shutil.rmtree(dirpath + '/data' + data_name)
    except FileNotFoundError:
        pass

    try:
        shutil.rmtree(output_path + '/' + pipeline_name)
    except FileNotFoundError:
        pass

def simulate_integration_image(data_name, pixel_scale, lens_galaxies, source_galaxies, target_signal_to_noise):
    output_path = "{}/data/".format(os.path.dirname(os.path.realpath(__file__))) + data_name + '/'
    psf_shape = (11, 11)
    image_shape = (150, 150)

    psf = im.PSF.simulate_as_gaussian(shape=psf_shape, pixel_scale=pixel_scale, sigma=pixel_scale)

    image_grids = grids.DataGrids.grids_for_simulation(shape=image_shape, pixel_scale=pixel_scale,
                                                       sub_grid_size=1, psf_shape=psf_shape)

    image_shape = image_grids.regular.padded_shape

    if not source_galaxies:

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=lens_galaxies, image_plane_grids=[image_grids])

    elif source_galaxies:

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=lens_galaxies, source_galaxies=source_galaxies,
                                                     image_plane_grids=[image_grids])

    ### Setup as a simulated image_coords and output as a fits for an lensing ###

    sim_image = im.Image.simulate_to_target_signal_to_noise(array=tracer.image_plane_images_for_simulation[0],
                                                            pixel_scale=pixel_scale,
                                                            target_signal_to_noise=target_signal_to_noise,
                                                            exposure_time_map=np.ones(image_shape),
                                                            background_sky_map=10.0 * np.ones(image_shape),
                                                            psf=psf, seed=1)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    array_util.numpy_array_to_fits(sim_image, path=output_path + 'regular.fits')
    array_util.numpy_array_to_fits(sim_image.noise_map, path=output_path + 'noise_map_.fits')
    array_util.numpy_array_to_fits(psf, path=output_path + '/psf.fits')
    array_util.numpy_array_to_fits(sim_image.exposure_time_map, path=output_path + 'exposure_map.fits')


def load_image(data_name, pixel_scale):
    data_dir = "{}/data/{}".format(dirpath, data_name)

    data = scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=data_dir + '/regular.fits', hdu=0,
                                                                          pixel_scale=pixel_scale)
    noise = scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=data_dir + '/noise_map_.fits', hdu=0,
                                                                           pixel_scale=pixel_scale)
    psf = im.PSF.from_fits_with_scale(file_path=data_dir + '/psf.fits', hdu=0, pixel_scale=pixel_scale)

    return im.Image(array=data, pixel_scale=pixel_scale, psf=psf, noise_map=noise)

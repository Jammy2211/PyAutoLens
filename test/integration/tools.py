from autolens.imaging import image as im
from autolens.imaging import scaled_array
from autolens.imaging import mask
from autolens.imaging import imaging_util
from autolens.lensing import ray_tracing
from autolens import conf

import matplotlib.pyplot as plt
import numpy as np
import shutil
import os

dirpath = os.path.dirname(os.path.realpath(__file__))
output_path = '/gpfs/data/pdtw24/Lens/integration/'

def reset_paths(data_name, pipeline_name, output_path):

    conf.instance.output_path = output_path

    try:
        shutil.rmtree(dirpath + '/data' + data_name)
    except FileNotFoundError:
        pass

    # try:
    #     shutil.rmtree(output_path + pipeline_name)
    # except FileNotFoundError:
    #     pass


def simulate_integration_image(data_name, pixel_scale, lens_galaxies, source_galaxies, target_signal_to_noise):

    output_path = "{}/data/".format(os.path.dirname(os.path.realpath(__file__))) + data_name + '/'
    psf_shape = (11, 11)
    image_shape = (150, 150)

    psf = im.PSF.simulate_as_gaussian(shape=psf_shape, sigma=0.6)

    image_grids = mask.ImagingGrids.unmasked_grids_for_simulation(shape=image_shape, pixel_scale=pixel_scale,
                                                                  sub_grid_size=1, psf_shape=psf_shape)

    image_shape = image_grids.image.padded_shape

    if not source_galaxies:

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=lens_galaxies, image_plane_grids=image_grids)

    elif source_galaxies:

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=lens_galaxies, source_galaxies=source_galaxies,
                                                     image_plane_grids=image_grids)

    ### Setup as a simulated image_coords and output as a fits for an lensing ###

    sim_image = im.PreparatoryImage.simulate_to_target_signal_to_noise(array=tracer.image_plane_image_for_simulation,
                                                                       pixel_scale=pixel_scale,
                                                                       target_signal_to_noise=target_signal_to_noise,
                                                                       effective_exposure_map=np.ones(image_shape),
                                                                       background_sky_map=10.0*np.ones(image_shape),
                                                                       psf=psf, seed=1)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    imaging_util.numpy_array_to_fits(sim_image, path=output_path + 'image.fits')
    imaging_util.numpy_array_to_fits(sim_image.noise_map, path=output_path + 'noise_map.fits')
    imaging_util.numpy_array_to_fits(psf, path=output_path + '/psf.fits')
    imaging_util.numpy_array_to_fits(sim_image.effective_exposure_map, path=output_path + 'exposure_map.fits')

def load_image(data_name, pixel_scale):

    data_dir = "{}/data/{}".format(dirpath, data_name)

    data = scaled_array.ScaledArray.from_fits_with_scale(file_path=data_dir + '/image.fits', hdu=0,
                                                         pixel_scale=pixel_scale)
    noise = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/noise_map.fits', hdu=0)
    psf = im.PSF.from_fits(file_path=data_dir + '/psf.fits', hdu=0)

    return im.Image(array=data, pixel_scale=pixel_scale, psf=psf, noise_map=noise)
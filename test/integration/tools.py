from autolens.pipeline import pipeline as pl
from autolens.imaging import image as im
from autolens.imaging import scaled_array
from autolens.imaging import mask
from autolens.imaging import imaging_util
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.lensing import ray_tracing
from autolens.lensing import galaxy
from autolens.autofit import non_linear as nl
from autolens import conf

import numpy as np
import shutil
import os

dirpath = os.path.dirname(os.path.realpath(__file__))
output_path = '/gpfs/data/pdtw24/Lens/integration/'

def simulate_integration_image(data_name, pixel_scale, lens_galaxies, source_galaxies, target_signal_to_noise):

    path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))
    output_path = "{}/data/".format(os.path.dirname(os.path.realpath(__file__))) + data_name + '/'
    psf_size = (11, 11)

    psf = im.PSF.from_fits(file_path=path + '/data/psf', hdu=0).trim(psf_size)
    psf = psf.renormalize()
    ma = mask.Mask.padded_mask_unmasked_psf_edges(shape_arc_seconds=(15.0, 15.0), pixel_scale=pixel_scale, pad_size=psf_size)

    image_plane_grids = mask.GridCollection.grids_from_mask_sub_grid_size_and_blurring_shape(mask=ma, sub_grid_size=1,
                                                                                             psf_shape=psf_size)

    if not source_galaxies:

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=lens_galaxies, image_grids=image_plane_grids)

    elif source_galaxies:

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=lens_galaxies, source_galaxies=source_galaxies,
                                                     image_grids=image_plane_grids)

    image_plane_image = tracer.image_plane_image
    image_plane_image_2d = ma.map_padded_1d_array_to_original_2d_array(image_plane_image)

    ### Setup as a simulated image_coords and output as a fits for an lensing ###

    shape = image_plane_image_2d.shape
    sim_image = im.PreparatoryImage.simulate_to_target_signal_to_noise(array=image_plane_image_2d, pixel_scale=pixel_scale,
                                                                       target_signal_to_noise=target_signal_to_noise, effective_exposure_map=10.0 * np.ones(shape),
                                                                       background_sky_map=20.0*np.ones(shape), psf=psf, include_poisson_noise=True, seed=1)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    imaging_util.numpy_array_to_fits(sim_image, path=output_path + 'image')
    imaging_util.numpy_array_to_fits(sim_image.estimated_noise, path=output_path + 'noise_map')
    imaging_util.numpy_array_to_fits(psf, path=output_path + '/psf')
    imaging_util.numpy_array_to_fits(sim_image.effective_exposure_time, path=output_path + 'exposure_time')

def load_image(data_name, pixel_scale):

    data_dir = "{}/data/{}".format(dirpath, data_name)

    data = scaled_array.ScaledArray.from_fits_with_scale(file_path=data_dir + '/image', hdu=0, pixel_scale=pixel_scale)
    noise = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/noise_map', hdu=0)
    psf = im.PSF.from_fits(file_path=data_dir + '/psf', hdu=0)

    return im.Image(array=data, pixel_scale=pixel_scale, psf=psf, noise_map=noise)
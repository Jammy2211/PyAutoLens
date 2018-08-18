from autolens.pipeline import pipeline as pl
from autolens.imaging import image as im
from autolens.imaging import scaled_array
from autolens.imaging import mask
from autolens.imaging import array_util
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.analysis import ray_tracing
from autolens.analysis import galaxy
from autolens.autopipe import non_linear as nl
from autolens import conf

import numpy as np
import shutil
import os

dirpath = os.path.dirname(os.path.realpath(__file__))
output_path = '/gpfs/data/pdtw24/Lens/integration/'

def simulate_integration_image(data_name, pixel_scale, lens_galaxies, source_galaxies):

    path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))
    output_path = "{}/data/".format(os.path.dirname(os.path.realpath(__file__))) + data_name + '/'
    psf_size = (11, 11)

    psf = im.PSF.from_fits(file_path=path + '/data/psf', hdu=0).trim(psf_size)
    psf = psf.renormalize()
    ma = mask.Mask.for_simulate(shape_arc_seconds=(15.0, 15.0), pixel_scale=pixel_scale, psf_size=psf_size)

    image_plane_grids = mask.GridCollection.from_mask_sub_grid_size_and_blurring_shape(mask=ma, sub_grid_size=1,
                                                                                       blurring_shape=psf_size)

    tracer = ray_tracing.Tracer(lens_galaxies=lens_galaxies, source_galaxies=source_galaxies,
                                image_plane_grids=image_plane_grids)

    galaxy_image_1d = tracer.galaxy_light_profiles_image_from_planes()
    galaxy_image_2d = ma.map_to_2d(galaxy_image_1d)

    ### Setup as a simulated image_coords and output as a fits for an analysis ###

    shape = galaxy_image_2d.shape
    sim_image = im.PrepatoryImage.simulate(array=galaxy_image_2d, effective_exposure_time=1000.0 * np.ones(shape),
                                              pixel_scale=pixel_scale, background_sky_map=10.0 * np.ones(shape),
                                              psf=psf,
                                              include_poisson_noise=True, seed=1)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    array_util.numpy_array_to_fits(sim_image, file_path=output_path + 'image')
    array_util.numpy_array_to_fits(sim_image.estimated_noise, file_path=output_path + 'noise')
    array_util.numpy_array_to_fits(psf, file_path=output_path + '/psf')
    array_util.numpy_array_to_fits(sim_image.effective_exposure_time, file_path=output_path + 'exposure_time')

def load_image(data_name, pixel_scale):

    data_dir = "{}/data/{}".format(dirpath, data_name)

    data = scaled_array.ScaledArray.from_fits_with_scale(file_path=data_dir + '/image', hdu=0, pixel_scale=pixel_scale)
    noise = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/noise', hdu=0)
    psf = im.PSF.from_fits(file_path=data_dir + '/psf', hdu=0)

    return im.Image(array=data, pixel_scale=pixel_scale, psf=psf, noise=noise)
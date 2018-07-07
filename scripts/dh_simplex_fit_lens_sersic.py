import sys

import src.config.config

sys.path.append("../")
import os

from src.imaging import scaled_array
from src.imaging import image
from src.imaging import mask
from src.imaging import grids
from src.profiles import light_profiles as lp
from src.profiles import mass_profiles as mp
from src.pixelization import frame_convolution
from src.analysis import model_mapper
from src.analysis import non_linear
from src.analysis import ray_tracing
from src.analysis import galaxy
from src.analysis import fitting

import scipy.optimize

lens_name = 'lens_sersic'
data_dir = "../data/"+lens_name.format(os.path.dirname(os.path.realpath(__file__)))

im = scaled_array.ScaledArray.from_fits(file_path=data_dir+'/image', hdu=0, pixel_scale=0.1)
noise = scaled_array.ScaledArray.from_fits(file_path=data_dir+'/noise', hdu=0, pixel_scale=0.1)
exposure_time = scaled_array.ScaledArray.from_fits(file_path=data_dir+'/exposure_time', hdu=0,
                                                   pixel_scale=0.1)
psf = image.PSF.from_fits(file_path=data_dir+'/psf', hdu=0, pixel_scale=0.1)

data = image.Image(array=im, effective_exposure_time=exposure_time, pixel_scale=0.1, psf=psf,
                   background_noise=noise, poisson_noise=noise)

msk = mask.Mask.circular(shape_arc_seconds=data.shape_arc_seconds, pixel_scale=data.pixel_scale, radius_mask=2.0)

grid_coords = grids.CoordsCollection.from_mask(mask=msk, grid_size_sub=1, blurring_shape=psf.shape)
grid_data = grids.DataCollection.from_mask(mask=msk, image=data, noise=data.background_noise,
                                           exposure_time=data.effective_exposure_time)
mappers = grids.MapperCollection.from_mask(mask=msk)

frame = frame_convolution.FrameMaker(mask=msk)
convolver = frame.convolver_for_kernel_shape(kernel_shape=data.psf.shape,
                                             blurring_region_mask=msk.blurring_mask_for_kernel_shape(
                                                 kernel_shape=data.psf.shape))
kernel_convolver = convolver.convolver_for_kernel(kernel=data.psf)

# NON LINEAR ANALYSIS #

config = src.config.config.DefaultPriorConfig(config_folder_path='../src/config')
model_map = model_mapper.ModelMapper(config=config, light_profile=lp.EllipticalSersic)

def likelihood(params, model_mapper, grid_coords, grid_data, grid_mappers, kernel_convolver):

    physical_model = model_mapper.from_physical_vector(params)

    gal = galaxy.Galaxy(light_profiles=[physical_model.light_profile])
    ray_trace = ray_tracing.Tracer(lens_galaxies=[gal], source_galaxies=[], image_plane_grids=grid_coords)

    return -2.0 * fitting.fit_data_with_profiles(grid_data=grid_data, kernel_convolver=kernel_convolver,
                                                 tracer=ray_trace)

result = scipy.optimize.fmin(likelihood, x0=[0.0, 0.0, 0.5, 50.0, 0.5, 1.0, 5.0],
                                 args=(model_map, grid_coords, grid_data, mappers, kernel_convolver))

print(result)
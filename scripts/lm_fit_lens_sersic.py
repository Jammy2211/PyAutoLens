import sys
sys.path.append("../")
import os

from auto_lens.imaging import scaled_array
from auto_lens.imaging import image
from auto_lens.imaging import mask
from auto_lens.imaging import grids
from auto_lens.profiles import light_profiles as lp
from auto_lens.profiles import mass_profiles as mp
from auto_lens.pixelization import frame_convolution
from auto_lens.analysis import model_mapper
from auto_lens.analysis import non_linear
from auto_lens.analysis import ray_tracing
from auto_lens.analysis import galaxy
from auto_lens.analysis import fitting

import scipy.optimize
import lmfit

import numpy as np
from functools import partial

import matplotlib.pyplot as plt

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

grid_coords = grids.GridCoordsCollection.from_mask(mask=msk, grid_size_sub=1, blurring_size=psf.shape)
grid_data = grids.GridDataCollection.from_mask(mask=msk, image=data, noise=data.background_noise,
                                               exposure_time=data.effective_exposure_time)
mappers = grids.GridMapperCollection.from_mask(mask=msk)

frame = frame_convolution.FrameMaker(mask=msk)
convolver = frame.convolver_for_kernel_shape(kernel_shape=data.psf.shape,
                                             blurring_region_mask=msk.compute_blurring_mask(kernal_shape=data.psf.shape))
kernel_convolver = convolver.convolver_for_kernel(kernel=data.psf)

# NON LINEAR ANALYSIS #

config = model_mapper.Config(config_folder_path='../auto_lens/config')
model_map = model_mapper.ModelMapper(config=config, light_profile=lp.EllipticalSersic)

def likelihood(params, model_mapper, grid_coords, grid_data, grid_mappers, kernel_convolver):

    print(params)

    physical_model = model_mapper.from_physical_vector(params)

    gal = galaxy.Galaxy(light_profiles=[physical_model.light_profile])
    ray_trace = ray_tracing.Tracer(lens_galaxies=[gal], source_galaxies=[], image_plane_grids=grid_coords)

    return -2.0*fitting.fit_data_with_model(grid_data=grid_data, grid_mappers=grid_mappers,
                                            kernel_convolver=kernel_convolver, tracer=ray_trace)

# result = scipy.optimize.minimize(likelihood, x0=[0.1, 0.1, 0.5, 80.0, 0.5, 1.0, 5.0],
#                                  options={'gtol': 1e-6, 'disp': True},
#                                  args=(model_map, grid_coords, grid_data, mappers, kernel_convolver))

result = scipy.optimize.fmin(likelihood, x0=[0.0, 0.0, 0.5, 50.0, 0.5, 1.0, 5.0],
                                 args=(model_map, grid_coords, grid_data, mappers, kernel_convolver))

# params = lmfit.Parameters()
# params.add('x_center', value=0.0, min=-1.0, max=1.0)
# params.add('y_center', value=0.0, min=-1.0, max=1.0)
# params.add('axis_ratio', value=0.8, min=0.1, max=1.0)
# params.add('phi', value=90.0, min=0.0, max=180.0)
# params.add('intensity', value=0.5, min=0.0, max=3.0)
# params.add('effective_radius', value=1.0, min=0.0, max=4.0)
# params.add('sersic_index', value=4.0, min=0.6, max=8.0)
#
# result = lmfit.fmin(likelihood, params, method='Nelder-Mead', tol=0.1,
#                         args=(model_map, grid_coords, grid_data, mappers, kernel_convolver))

print(result)
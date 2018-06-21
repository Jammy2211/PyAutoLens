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

import numpy as np
from functools import partial
import pymultinest

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

grid_coords = grids.GridCoordsCollection.from_mask(mask=msk, grid_size_sub=1, blurring_shape=psf.shape)
grid_data = grids.GridDataCollection.from_mask(mask=msk, image=data, noise=data.background_noise,
                                               exposure_time=data.effective_exposure_time)
mappers = grids.GridMapperCollection.from_mask(mask=msk)

frame = frame_convolution.FrameMaker(mask=msk)
convolver = frame.convolver_for_kernel_shape(kernel_shape=data.psf.shape,
                                             blurring_region_mask=msk.compute_blurring_mask(
                                                 kernel_shape=data.psf.shape))
kernel_convolver = convolver.convolver_for_kernel(kernel=data.psf)

# NON LINEAR ANALYSIS #

config = model_mapper.Config(config_folder_path='../auto_lens/config')
model_map = model_mapper.ModelMapper(config=config, light_profile=lp.EllipticalSersic)
multinest = non_linear.MultiNest(path='../results/', obj_name='test_fit', model_mapper=model_map)

def prior(cube, ndim, n_params, model_mapper):

    phys_cube = model_mapper.physical_vector_from_hypercube_vector(hypercube_vector=cube)

    for i in range(n_params):
        cube[i] = phys_cube[i]

    return cube

def likelihood(physical_cube, ndim, n_params, model_mapper, grid_coords, grid_data, grid_mappers, kernel_convolver):

    physical_model = model_mapper.from_physical_vector(physical_cube)

    gal = galaxy.Galaxy(light_profiles=[physical_model.light_profile])
    ray_trace = ray_tracing.Tracer(lens_galaxies=[gal], source_galaxies=[], image_plane_grids=grid_coords)

    return fitting.fit_data_with_profiles(grid_data=grid_data, kernel_convolver=kernel_convolver, tracer=ray_trace)

prior_pass = partial(prior, model_mapper=model_map)
likelihood_pass = partial(likelihood, model_mapper=model_map, grid_coords=grid_coords, grid_data=grid_data,
                          grid_mappers=mappers, kernel_convolver=kernel_convolver)

pymultinest.run(likelihood_pass, prior_pass, n_dims=multinest.total_parameters, n_params=multinest.total_parameters,
                n_clustering_params=None, wrapped_params=None, importance_nested_sampling=True, multimodal=True,
                const_efficiency_mode=False, n_live_points=50, evidence_tolerance=0.5, sampling_efficiency=0.2,
                n_iter_before_update=100, null_log_evidence=-1e+90, max_modes=100, mode_tolerance=-1e+90,
                outputfiles_basename=multinest.results_path + multinest.obj_name + '_', seed=-1, verbose=False,
                resume=True, context=0, write_output=True, log_zero=-1e+100, max_iter=0, init_MPI=True)



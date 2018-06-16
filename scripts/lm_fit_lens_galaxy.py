import sys
sys.path.append("../")

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

### Simulate Image ###

### Setup mask + grid of this image ###

ma = mask.Mask.circular(shape_arc_seconds=(4.0, 4.0), pixel_scale=0.1, radius_mask=1.6)
ma = mask.Mask(array=ma, pixel_scale=0.1)

image_grids = grids.GridCoordsCollection.from_mask(mask=ma, grid_size_sub=1, blurring_size=(3, 3))
mappers = grids.GridMapperCollection.from_mask(mask=ma)

### Setup the ray tracing model, and use to generate the 2D galaxy image ###

gal = galaxy.Galaxy(light_profiles=[lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.7,
                                                        phi=90.0, intensity=0.5, effective_radius=1.0,
                                                        sersic_index=4.0)])

ray_trace = ray_tracing.Tracer(lens_galaxies=[gal], source_galaxies=[], image_plane_grids=image_grids)

grid_galaxy_image = ray_trace.generate_image_of_galaxy_light_profiles()
galaxy_image = mappers.data_to_pixel.map_to_2d(grid_galaxy_image)

# plt.imshow(galaxy_image)
# plt.show()

### Setup the image as an image.

sim_image = image.Image.simulate(array=galaxy_image)


### Fit Image With Same Model ###

sim_grid_datas = grids.GridDataCollection.from_mask(mask=ma, image=sim_image, noise=0.1*np.ones(ma.shape),
                                                    exposure_time=np.ones(ma.shape))
mappers = grids.GridMapperCollection.from_mask(mask=ma)

frame = frame_convolution.FrameMaker(mask=ma)
convolver = frame.convolver_for_kernel_shape(kernel_shape=(3, 3),
                                             blurring_region_mask=ma.compute_blurring_mask(kernal_shape=(3, 3)))
# This PSF leads to no blurring, so equivalent to being off.
kernel_convolver = convolver.convolver_for_kernel(kernel=np.array([[0., 0., 0.],
                                                                   [0., 1., 0.],
                                                                   [0., 0., 0.]]))

# NON LINEAR ANALYSIS #

config = model_mapper.Config(config_folder_path='../auto_lens/config')
model_map = model_mapper.ModelMapper(config=config, light_profile=lp.EllipticalSersic)
multinest = non_linear.MultiNest(path='../results/', obj_name='test_fit', model_mapper=model_map)

def prior(cube, ndim, n_params, model_mapper):

    phys_cube = model_mapper.physical_vector_from_hypercube_vector(hypercube_vector=cube)

    for i in range(n_params):
        cube[i] = phys_cube[i]

    return cube

def likelihood(physical_cube, ndim, n_params, model_mapper, grid_datas, grid_mappers, kernel_convolver):

    physical_model = model_mapper.from_physical_vector(physical_cube)

    gal = galaxy.Galaxy(light_profiles=[physical_model.light_profile])
    ray_trace = ray_tracing.Tracer(lens_galaxies=[gal], source_galaxies=[], image_plane_grids=image_grids)

    return fitting.fit_data_with_model(grid_datas=grid_datas, grid_mappers=grid_mappers,
                                       kernel_convolver=kernel_convolver, tracer=ray_trace)

prior_pass = partial(prior, model_mapper=model_map)
likelihood_pass = partial(likelihood, model_mapper=model_map, grid_datas=sim_grid_datas, grid_mappers=mappers,
                          kernel_convolver=kernel_convolver)

pymultinest.run(likelihood_pass, prior_pass, n_dims=multinest.total_parameters, n_params=multinest.total_parameters,
                n_clustering_params=None, wrapped_params=None, importance_nested_sampling=True, multimodal=True,
                const_efficiency_mode=False, n_live_points=50, evidence_tolerance=0.5, sampling_efficiency=0.2,
                n_iter_before_update=100, null_log_evidence=-1e+90, max_modes=100, mode_tolerance=-1e+90,
                outputfiles_basename=multinest.results_path + multinest.obj_name + '_', seed=-1, verbose=False,
                resume=True, context=0, write_output=True, log_zero=-1e+100, max_iter=0, init_MPI=True)



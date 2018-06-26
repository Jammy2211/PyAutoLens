import sys
sys.path.append("../")
import os
import numpy as np

from src.imaging import image
from src.imaging import mask
from src.imaging import grids
from src.profiles import light_profiles as lp
from src.profiles import mass_profiles as mp
from src.analysis import ray_tracing
from src.analysis import galaxy
from src.tools import arrays

output_dir = "{}/../data/".format(os.path.dirname(os.path.realpath(__file__)))

### Setup mask + grid of this image ###

ma = mask.Mask.for_simulate(shape_arc_seconds=(5.0, 5.0), pixel_scale=0.1, psf_size=(3,3))
image_grids = grids.CoordsCollection.from_mask(mask=ma, grid_size_sub=1, blurring_shape=(3, 3))
mappers = grids.MapperCollection.from_mask(mask=ma)

### Setup the ray tracing model, and use to generate the 2D galaxy image ###

# lens_name = 'lens_sersic'
#gal = galaxy.Galaxy(light_profiles=[lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.7,
#                    phi=90.0, intensity=0.5, effective_radius=1.0, sersic_index=4.0)])
#ray_trace = ray_tracing.Tracer(lens_galaxies=[gal], source_galaxies=[], image_plane_grids=image_grids)

lens_name = 'source_sersic'
lens_galaxy = galaxy.Galaxy(mass_profiles=[mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.7, phi=90.0,
                                                                   einstein_radius=1.0)])
source_galaxy = galaxy.Galaxy(light_profiles=[lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.7,
                                                        phi=90.0, intensity=0.5, effective_radius=1.0,
                                                        sersic_index=2.0)])
ray_trace = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                               image_plane_grids=image_grids)

galaxy_image_1d = ray_trace.generate_image_of_galaxy_light_profiles()
galaxy_image_2d = mappers.data_to_pixel.map_to_2d(galaxy_image_1d)

### Setup as a simulated image and output as a fits for an analysis ###

sim_image = image.Image.simulate(array=galaxy_image_2d)

if os.path.exists(output_dir+lens_name) == False:
    os.makedirs(output_dir+lens_name)

arrays.numpy_array_to_fits(sim_image, file_path=output_dir+lens_name+'/image')
arrays.numpy_array_to_fits(np.ones(sim_image.shape), file_path=output_dir+lens_name+'/noise')
arrays.numpy_array_to_fits(np.ones(sim_image.shape), file_path=output_dir+lens_name+'/exposure_time')
arrays.numpy_array_to_fits(np.array([[0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]]), file_path=output_dir+lens_name+'/psf')
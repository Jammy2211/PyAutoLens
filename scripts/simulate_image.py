import sys
sys.path.append("../")
import os
import numpy as np
import matplotlib.pyplot as plt

from src.imaging import image
from src.imaging import mask
from src.profiles import light_profiles as lp
from src.profiles import mass_profiles as mp
from src.analysis import ray_tracing
from src.analysis import galaxy
from src.tools import arrays

path =  "{}/".format(os.path.dirname(os.path.realpath(__file__)))
output_path = "{}/../data/profiling/".format(os.path.dirname(os.path.realpath(__file__)))

lens_name = 'LSST/'
pixel_scale = 0.2
psf_size = (7, 7)

psf = image.PSF.from_fits(file_path=path+'../data/SLACS_05_WHT_All/slacs_1_post', hdu=3, pixel_scale=pixel_scale)
psf = psf.trim(psf_size)
print(psf.shape)
ma = mask.Mask.for_simulate(shape_arc_seconds=(20.0, 20.0), pixel_scale=pixel_scale, psf_size=psf_size)

print(ma)
print(ma.shape)

image_plane_grids = mask.CoordinateCollection.from_mask_subgrid_size_and_blurring_shape(mask=ma, subgrid_size=4,
                                                                                        blurring_shape=psf_size)


### Setup the ray tracing model, and use to generate the 2D galaxy image_coords ###

sersic = lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=0.5, effective_radius=1.3,
                             sersic_index=3.0)
isothermal = mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, einstein_radius=1.0)

lens_galaxy = galaxy.Galaxy(light_profiles=[sersic], mass_profiles=[isothermal])
source_galaxy = galaxy.Galaxy(light_profiles=[sersic])

ray_trace = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                               image_plane_grids=image_plane_grids)

galaxy_image_1d = ray_trace.generate_image_of_galaxy_light_profiles()
galaxy_image_2d = ma.map_to_2d(galaxy_image_1d)

### Setup as a simulated image_coords and output as a fits for an analysis ###

shape = galaxy_image_2d.shape
sim_image = image.Image.simulate(array=galaxy_image_2d, effective_exposure_time=np.ones(shape), pixel_scale=pixel_scale,
                                 background_sky_map=np.ones(shape), psf=psf, include_poisson_noise=True, seed=1)

if os.path.exists(output_path + lens_name) == False:
    os.makedirs(output_path + lens_name)

arrays.numpy_array_to_fits(sim_image, file_path=output_path + lens_name + 'image')
arrays.numpy_array_to_fits(np.ones(sim_image.shape), file_path=output_path + lens_name + 'noise')
arrays.numpy_array_to_fits(np.ones(sim_image.shape), file_path=output_path + lens_name + 'exposure_time')
arrays.numpy_array_to_fits(psf, file_path=output_path + lens_name + '/psf')
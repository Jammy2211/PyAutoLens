import sys
sys.path.append("../")
import os
import numpy as np
import matplotlib.pyplot as plt

from autolens.imaging import image
from autolens.imaging import mask
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.analysis import ray_tracing
from autolens.analysis import galaxy
from autolens.imaging import array_util

path =  "{}/".format(os.path.dirname(os.path.realpath(__file__)))
output_path = "{}/../data/integration_old/".format(os.path.dirname(os.path.realpath(__file__)))

lens_name = 'lowres_lens_and_source/'
psf_size = (11, 11)
pixel_scale = 0.2

psf = image.PSF.from_fits(file_path=path+'../data/integration_old/psf', hdu=0).trim(psf_size)
psf = psf.renormalize()
ma = mask.Mask.for_simulate(shape_arc_seconds=(15.0, 15.0), pixel_scale=pixel_scale, psf_size=psf_size)

image_plane_grids = mask.GridCollection.from_mask_sub_grid_size_and_blurring_shape(mask=ma, sub_grid_size=1,
                                                                                   blurring_shape=psf_size)

sersic_lens = lp.EllipticalSersic(centre=(0.01, 0.01), axis_ratio=0.8, phi=0.0, intensity=1.0,
                                  effective_radius=1.3, sersic_index=3.0)
isothermal = mp.EllipticalIsothermal(centre=(0.01, 0.01), axis_ratio=0.8, phi=0.0, einstein_radius=2.0)
sersic_source = lp.EllipticalSersic(centre=(0., 0.), axis_ratio=0.9, phi=90.0, intensity=1.0,
                                    effective_radius=1.0, sersic_index=2.0)

lens_galaxy = galaxy.Galaxy(light_profile=sersic_lens, mass_profile=isothermal)
source_galaxy = galaxy.Galaxy(light_profile=sersic_source)

tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                            image_plane_grids=image_plane_grids)

galaxy_image_1d = tracer.galaxy_light_profiles_image_from_planes()
galaxy_image_2d = ma.map_to_2d(galaxy_image_1d)

### Setup as a simulated image_coords and output as a fits for an analysis ###

shape = galaxy_image_2d.shape
sim_image = image.PrepatoryImage.simulate(array=galaxy_image_2d, effective_exposure_time=1000.0*np.ones(shape),
                                 pixel_scale=pixel_scale, background_sky_map=10.0*np.ones(shape), psf=psf,
                                          include_poisson_noise=True, seed=1)

if os.path.exists(output_path + lens_name) == False:
    os.makedirs(output_path + lens_name)

array_util.numpy_array_to_fits(sim_image, file_path=output_path + lens_name + 'image')
array_util.numpy_array_to_fits(sim_image.estimated_noise, file_path=output_path + lens_name + 'noise')
array_util.numpy_array_to_fits(sim_image.effective_exposure_time, file_path=output_path + lens_name + 'exposure_time')
array_util.numpy_array_to_fits(psf, file_path=output_path + lens_name + '/psf')
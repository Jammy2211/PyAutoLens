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
from imaging import array_util

path =  "{}/".format(os.path.dirname(os.path.realpath(__file__)))
output_path = "{}/../data/integration/".format(os.path.dirname(os.path.realpath(__file__)))

lens_name = 'hst_1/'
psf_size = (21, 21)
pixel_scale = 0.05

psf = image.PSF.from_fits(file_path=path+'../profiling/data/psf', hdu=3, pixel_scale=pixel_scale).trim(psf_size)
psf = psf.renormalize()
ma = mask.Mask.for_simulate(shape_arc_seconds=(15.0, 15.0), pixel_scale=pixel_scale, psf_size=psf_size)

image_plane_grids = mask.GridCollection.from_mask_sub_grid_size_and_blurring_shape(mask=ma, sub_grid_size=1,
                                                                                   blurring_shape=psf_size)

### Setup the ray tracing model, and use to generate the 2D galaxy image_coords ###

# sersic_lens = lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=0.5, effective_radius=1.6,
#                              sersic_index=4.0)
# isothermal = mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, einstein_radius=1.3)
# sersic_source = lp.EllipticalSersic(centre=(0.3, 0.2), axis_ratio=0.6, phi=45.0, intensity=2.0, effective_radius=1.0,
#                              sersic_index=1.0)

sersic_lens = lp.EllipticalSersic(centre=(0.1, -0.1), axis_ratio=0.7, phi=90.0, intensity=1.0, effective_radius=1.3,
                             sersic_index=3.0)
isothermal = mp.EllipticalIsothermal(centre=(0.1, -0.1), axis_ratio=0.6, phi=30.0, einstein_radius=0.9)
sersic_source = lp.EllipticalSersic(centre=(-0.2, 0.2), axis_ratio=0.9, phi=45.0, intensity=1.0, effective_radius=1.0,
                             sersic_index=2.0)

lens_galaxy = galaxy.Galaxy(light_profile=sersic_lens, mass_profile=isothermal)
source_galaxy = galaxy.Galaxy(light_profile=sersic_source)

ray_trace = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                               image_plane_grids=image_plane_grids)

galaxy_image_1d = ray_trace.generate_image_of_galaxy_light_profiles()
galaxy_image_2d = ma.map_to_2d(galaxy_image_1d)

### Setup as a simulated image_coords and output as a fits for an analysis ###

shape = galaxy_image_2d.shape
sim_image = image.Image.simulate(array=galaxy_image_2d, effective_exposure_time=200.0*np.ones(shape),
                                 pixel_scale=pixel_scale,
                                 background_sky_map=10.0*np.ones(shape), psf=psf, include_poisson_noise=True, seed=1)

print(np.max(sim_image))
print(np.max(sim_image) / np.max(sim_image.estimated_noise))

plt.imshow(sim_image)
plt.show()

if os.path.exists(output_path + lens_name) == False:
    os.makedirs(output_path + lens_name)

array_util.numpy_array_to_fits(sim_image, file_path=output_path + lens_name + 'image')
array_util.numpy_array_to_fits(sim_image.estimated_noise, file_path=output_path + lens_name + 'noise')
array_util.numpy_array_to_fits(sim_image.effective_exposure_time, file_path=output_path + lens_name + 'exposure_time')
array_util.numpy_array_to_fits(psf, file_path=output_path + lens_name + '/psf')
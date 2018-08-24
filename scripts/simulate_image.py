import sys
sys.path.append("../")
import os
import numpy as np
import matplotlib.pyplot as plt

from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.analysis import ray_tracing
from autolens.analysis import galaxy
from autolens.imaging import array_util

def simulate_image(data_name, pixel_scale, psf_shape, lens_galaxies, source_galaxies, target_signal_to_noise):

    path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))
    output_path = "{}/../data/".format(os.path.dirname(os.path.realpath(__file__))) + data_name + '/'

    psf = im.PSF.from_fits(file_path=path + '../data/integration/psf', hdu=0).trim(psf_shape)
    psf = psf.renormalize()
    ma = mask.Mask.for_simulate(shape_arc_seconds=(7.0, 7.0), pixel_scale=pixel_scale, psf_size=psf_shape)

    image_plane_grids = mask.GridCollection.from_mask_sub_grid_size_and_blurring_shape(mask=ma, sub_grid_size=1,
                                                                                       blurring_shape=psf_shape)

    if not source_galaxies:

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=lens_galaxies, image_grids=image_plane_grids)

    elif source_galaxies:

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=lens_galaxies, source_galaxies=source_galaxies,
                                                     image_grids=image_plane_grids)

    image_plane_image = tracer.image_plane_image
    image_plane_image_2d = ma.map_to_2d(image_plane_image)

    ### Setup as a simulated image_coords and output as a fits for an analysis ###

    shape = image_plane_image_2d.shape
    sim_image = im.PrepatoryImage.simulate_to_target_signal_to_noise(array=image_plane_image_2d, pixel_scale=pixel_scale,
                target_signal_to_noise=target_signal_to_noise, effective_exposure_time=10.0 * np.ones(shape),
                background_sky_map=20.0*np.ones(shape), psf=psf, include_poisson_noise=True, seed=1)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    array_util.numpy_array_to_fits(sim_image, file_path=output_path + 'image')
    array_util.numpy_array_to_fits(sim_image.estimated_noise, file_path=output_path + 'noise')
    array_util.numpy_array_to_fits(psf, file_path=output_path + '/psf')
    array_util.numpy_array_to_fits(sim_image.effective_exposure_time, file_path=output_path + 'exposure_time')

# sersic_lens = lp.EllipticalSersicLP(centre=(0.01, 0.01), axis_ratio=0.8, phi=0.0, intensity=1.0,
#                                     effective_radius=1.3, sersic_index=3.0)
isothermal = mp.EllipticalIsothermalMP(centre=(0.01, 0.01), axis_ratio=0.9, phi=45.0, einstein_radius=1.4)
sis = mp.SphericalIsothermalMP(centre=(0.6, 0.6), einstein_radius=0.2)
sersic_source = lp.EllipticalSersicLP(centre=(0.01, 0.01), axis_ratio=0.9, phi=90.0, intensity=1.0,
                                      effective_radius=1.0, sersic_index=2.0)

lens_galaxy = galaxy.Galaxy(lens=isothermal, subhalo=sis)
source_galaxy = galaxy.Galaxy(light_profile=sersic_source)

simulate_image(data_name='source_sub3/', pixel_scale=0.05, psf_shape=(11, 11), lens_galaxies=[lens_galaxy],
               source_galaxies=[source_galaxy], target_signal_to_noise=30.0)
from autolens.imaging import imaging_util
from autolens.imaging import image
from autolens.imaging import mask
from autolens.lensing import ray_tracing
from autolens.lensing import galaxy as g
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
import matplotlib.pyplot as plt
import os

# In this example, we'll simulate another lens + source galaxy system, but include the lens's light and make the
# lens and source profiles more complex

# We called these functions in example/simulate/1_simulate.py - check back there if you need a reminder!
psf = image.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.75)
imaging_grids = mask.ImagingGrids.unmasked_grids_for_simulation(shape=(100, 100), pixel_scale=0.1, psf_shape=psf.shape)

# We again use the galaxy, light_profiles and mass_profiles modules to set up our lens and source. However, we've made
# things a bit more complex than before:
# 1) The lens galaxy includes a light component (an elliptical dev Vaucoleurs).
# 2) The lens galaxy mass profile also includes an external shear.
# 3) The source galaxy has two light profiles, representing a bulge and disk

# The key-point to note below is that we can add any number of light and / or mass profiles to a galaxy, and they'll
# be included in the simulation.

lens_galaxy = g.Galaxy(light=lp.EllipticalDevVaucouleurs(centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0, intensity=0.1,
                                                 effective_radius=0.8),
                       sie=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=40.0, einstein_radius=1.8),
                       shear=mp.ExternalShear(magnitude=0.05, phi=45.0))
source_galaxy = g.Galaxy(light=lp.EllipticalExponential(centre=(0.0, 0.0), axis_ratio=0.9, phi=90.0, intensity=0.5,
                                                        effective_radius=0.3))

# Pass into the ray-tracing module and simulate the image.
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_grids=imaging_grids)
image_simulated = image.PreparatoryImage.simulate(array=tracer.image_plane_image, pixel_scale=0.07,
                                                  exposure_time=300.0, psf=psf, background_sky_level=5.0,
                                                  add_noise=False)

plt.imshow(image_simulated)
plt.show()

# Finally, lets output these files to.fits so that we can fit them in the phase and pipeline examples
path = "{}".format(os.path.dirname(os.path.realpath(__file__))) # Setup path so we can output the simulated data.
imaging_util.numpy_array_to_fits(array=image_simulated, path=path+'/../data/2_intermediate/image.fits')
imaging_util.numpy_array_to_fits(array=image_simulated.estimated_noise_map, path=path + '/../data/2_intermediate/noise_map.fits')
imaging_util.numpy_array_to_fits(array=psf, path=path+'/../data/2_intermediate/psf.fits')
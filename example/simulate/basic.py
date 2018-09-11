from autolens.imaging import imaging_util
from autolens.imaging import image
from autolens.lensing import grids
from autolens.lensing import ray_tracing
from autolens.lensing import galaxy as g
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
import matplotlib.pyplot as plt
import os

# In this example, we'll simulate a basic lens + source galaxy system and output the images (as .fits) for modeling
# with AutoLens (see the phase/basic.py example).

# First, lets setup the PSF we are going to blur our simulated image with, using a Gaussian profile on an 11x11 grid.
psf = image.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.75)
# plt.imshow(psf)
# plt.show()

# We need to set up the grids of Cartesian coordinates we will use to perform ray-tracing. The function below
# sets these grids up using the shape and pixel-scale of the image we will ultimately simulate. The PSF shape is
# required to ensure that edge-effects do not impact PSF blurring later in the simulation.
lensing_grids = grids.LensingGrids.padded_grids_for_simulation(shape=(100, 100), pixel_scale=0.07, psf_shape=psf.shape)

# Use the 'galaxy' module (imported as 'g'), 'light_profiles' module (imported as 'lp') and 'mass profiles' module
# (imported as 'mp') to setup the lens galaxy. The lens below has an elliptical Sersic light profile and singular
# isothermal ellipsoid (SIE) mass profile.
lens_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0, intensity=0.1,
                                                 effective_radius=0.8, sersic_index=3.0),
                       mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=40.0, einstein_radius=1.8))

# Use the above modules to setup the source galaxy, which in this example has an elliptical Exponential profile.
source_galaxy = g.Galaxy(light=lp.EllipticalExponential(centre=(0.0, 0.0), axis_ratio=0.9, phi=90.0, intensity=0.5,
                                                        effective_radius=0.3))

# Pass these galaxies into the 'ray_tracing' module, in this particular case a tracer which has both an image and source
# plane. Using the lens galaxy's mass profile(s), deflection-angle and ray-tracing calculations are performed to
# setup the source-plane.
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_grids=lensing_grids)

# The tracer class has lots of in-built properties for extracting different images in our lensing system. In this case,
# we want to extract the image-plane image of our lens and source galaxies - lets have a look at what it looks like.

### NOTE - currently we have to explictly map this image from a 1D array, which is used to perform autolens analysis.
### In the near future this 2D mapping will be perform implicitly within the tracer class and the confusing syntax below
### will be hidden from the user - e.g. (tracer.image_plane_image) will be all you need!
image_plane_image_2d = imaging_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(tracer.image_plane_image,
                                                                                shape=lensing_grids.image.padded_shape)
#plt.imshow(image_plane_image_2d)
#plt.show()


# Next, we simulate the image, using the simulate function in the imaging module. We add various effects, including
# PSF blurring, the background sky and Poisson noise_map. The simulation requires an effective exposure time map and
# background sky image - we'll just use arrays of a single value to keep it simple for now.
image_simulated = image.PreparatoryImage.simulate(array=image_plane_image_2d, pixel_scale=0.07, exposure_time=300.0,
                                                  psf=psf, background_sky_level=5.0, include_poisson_noise=True)
plt.imshow(image_simulated)
plt.show()

# Finally, lets output these files to.fits so that we can fit them in the phase and pipeline examples
path = "{}".format(os.path.dirname(os.path.realpath(__file__))) # Setup path so we can output the simulated data.
imaging_util.numpy_array_to_fits(array=image_simulated, path=path+'/../data/basic/image')
imaging_util.numpy_array_to_fits(array=image_simulated.estimated_noise, path=path+'/../data/basic/noise_map')
imaging_util.numpy_array_to_fits(array=psf, path=path+'/../data/basic/psf')
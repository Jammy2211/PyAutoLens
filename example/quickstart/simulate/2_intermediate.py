from autolens.imaging import imaging_util
from autolens.imaging import image
from autolens.imaging import mask
from autolens.lensing import ray_tracing
from autolens.lensing import galaxy as g
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
import matplotlib.pyplot as plt
import os

# In this example, we'll simulate a basic lens + source galaxy system and output the images (as .fits) for modeling
# with AutoLens (see the phase/1_basic.py example).

# First, lets setup the PSF we are going to blur our simulated image with, using a Gaussian profile on an 11x11 grid.
psf = image.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.75)
# plt.imshow(psf)
# plt.show()

# We need to set up the grids of Cartesian coordinates we will use to perform ray-tracing. The function below
# sets these grids up using the shape and pixel-scale of the image we will ultimately simulate. The PSF shape is
# required to ensure that edge-effects do not impact PSF blurring later in the simulation.
imaging_grids = mask.ImagingGrids.padded_grids_for_simulation(shape=(100, 100), pixel_scale=0.07, psf_shape=psf.shape)

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
                                             image_grids=imaging_grids)

# The tracer class has lots of in-built properties for extracting different images in our lensing system. In this case,
# we want to extract the 2d image-plane image of our lens and source galaxies - lets have a look at what it looks like.
#plt.imshow(tracer.image_plane_image_2d)
#plt.show()

# Next, we simulate the image, using the simulate function in the imaging module. We add various effects, including
# PSF blurring, the background sky and Poisson noise_map.
image_simulated = image.PreparatoryImage.simulate(array=tracer.image_plane_image_2d, pixel_scale=0.07,
                                                  exposure_time=300.0, psf=psf, background_sky_level=5.0,
                                                  add_noise=False)
plt.imshow(image_simulated)
plt.show()

# Finally, lets output these files to.fits so that we can fit them in the phase and pipeline examples
path = "{}".format(os.path.dirname(os.path.realpath(__file__))) # Setup path so we can output the simulated data.
imaging_util.numpy_array_to_fits(array=image_simulated, path=path+'/../data/basic/image.fits')
imaging_util.numpy_array_to_fits(array=image_simulated.estimated_noise_map, path=path + '/../data/basic/noise_map.fits')
imaging_util.numpy_array_to_fits(array=psf, path=path+'/../data/basic/psf.fits')
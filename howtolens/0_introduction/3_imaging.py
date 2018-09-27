from autolens.imaging import imaging_util
from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.lensing import ray_tracing
from autolens.lensing import galaxy as g
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.plotting import ray_tracing_plotters
from autolens.plotting import imaging_plotters
import os

# In the previous examples, we used the lensing module to create images of strong lens configurations. In this example,
# we'll use the 'imaging' to simulate these images, as if they were observed on a real telescope.

# To simulate an image, we use a special type of grid, we'll come back to this below.
image_plane_grids = mask.ImagingGrids.grids_for_simulation(shape=(100, 100), pixel_scale=0.1, psf_shape=(11, 11))

# Now, lets setup, galaxies and tracer - you've seen all this code before!
lens_galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6))
source_galaxy = g.Galaxy(light=lp.EllipticalExponential(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                                 intensity=1.0, effective_radius=1.0))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)

# Lets look at the tracer's image-plane image - the image we'll be simulating as if it were observed on a telescope.

# We can use a plotters 'individual' method to plot individual images, like the image-plane image.

ray_tracing_plotters.plot_ray_tracing_individual(tracer=tracer, plot_image_plane_image=True)

# To simulate the image a telescope takes, we need a Point-Spread Function to represent the telescope's optics.
# We can use the imaging module to simulate a PSF as a Gaussian:
psf = im.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.75)

# To simulate the image, we pass the tracer's image-plane image to the imaging module's simulate function. We add the
# following effects:
# 1) Telescope optics: Using the Point Spread Functions above.
# 2) The Background Sky: However the image that is returned is background sky subtracted.
# 3) Poisson noise: Due to the background sky, lens galaxy and source galaxy counts.

# We don't use the tracer's image-plane image that we plotted above. Instead, we use an image-plane image that which
# has been generated specifically for simulating an image (remember we changed our grid above, when making the image
# grids).

# This image has a padded boarder, which ensures that edge-effects do not degrade the simulation of the telescopes
# optics PSF during convolution.

image_simulated = im.PreparatoryImage.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                                               exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

# Lets plot the image - we can see the image has been blurred and noise has been added.
imaging_plotters.plot_image(image=image_simulated)

# Finally, lets output these files to.fits files, we'll begin to analysis them in the next tutorial!

path = "{}".format(os.path.dirname(os.path.realpath(__file__))) # Setup path so we can output the simulated data.
im.output_imaging_to_fits(image=image_simulated, image_path=path+'/data/image.fits',
                                                 noise_map_path=path+'/data/noise_map.fits',
                                                 psf_path=path+'/data/psf.fits',
                          overwrite=True)
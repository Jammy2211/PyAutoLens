from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.lensing import ray_tracing
from autolens.lensing import galaxy as g
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.plotting import ray_tracing_plotters
from autolens.plotting import imaging_plotters
import os

# In this example, we'll use the 'imaging' module (imported as 'im') to 'simulate' the images of strong lenses we made
# in the previous example. That is, we'll make them appear as if we had observed using a real telescope, with the
# settings of this example chosen to make an image indicative of the Hubble Space Telescope.

# To simulate an image, we use a special type of grid. This pads shape that we input relative to the PSF-shape,
# to ensure that the edge's of our simulated image are not degraded (we'll come back to this below).
image_plane_grids = mask.ImagingGrids.grids_for_simulation(shape=(130, 130), pixel_scale=0.05, psf_shape=(11, 11))
print(image_plane_grids.image.mask_shape)
print(image_plane_grids.image.padded_shape)

# Now, lets setup our galaxies and tracer.
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=45.0,
                                                        intensity=1.0, effective_radius=1.0, sersic_index=2.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)

# Lets look at the tracer's image-plane image - this is the image we'll be simulating.
ray_tracing_plotters.plot_ray_tracing_individual(tracer=tracer, plot_image_plane_image=True)

# To simulate an image, we need to model the telescope's optics. We'll do this by convolving the image with a
# Point-Spread Function (PSF), which we can simulate as a Gaussian using the imaging module.
psf = im.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.75)

# To simulate the image, we don't use the tracer's image-plane image that we plotted above. Instead, we use an
# image-plane image which has been generated specifically for simulating an image, using the padded grid above that
# ensures edge-effects do not degrade our simulation.
image_to_simulate = tracer.image_plane_image_for_simulation

# This image has a padded boarder, which ensures that edge-effects do not degrade the simulation of the telescopes
# optics PSF during convolution.
print(tracer.image_plane_image.shape)
print(image_to_simulate.shape)

# Now, to simulate the image, we pass the tracer's image-plane image to the imaging module's simulate function. This
# adds the following effects:
# 1) Telescope optics: Using the Point Spread Function above.
# 2) The Background Sky: Although the image that is returned is automatically background sky subtracted.
# 3) Poisson noise: Due to the background sky, lens galaxy and source galaxy Poisson photon counts.

image_simulated = im.PreparatoryImage.simulate(array=image_to_simulate, pixel_scale=0.05, exposure_time=300.0, psf=psf,
                                               background_sky_level=0.1, add_noise=True)

# Lets plot the image - we can see the image has been blurred due to the telescope optics and noise has been added.
imaging_plotters.plot_image(image=image_simulated)

# Finally, lets output these files to.fits files, we'll begin to analyze them in the next tutorial!
path = "{}".format(os.path.dirname(os.path.realpath(__file__))) # Setup path so we can output the simulated data.
im.output_imaging_to_fits(image=image_simulated, image_path=path+'/data/image.fits',
                                                 noise_map_path=path+'/data/noise_map.fits',
                                                 psf_path=path+'/data/psf.fits',
                          overwrite=True)

# You've just completed your third tutorial, try the following:

# 1) Change the size (sigma) of the PSF, what happens to the lensed source galaxy's appearance?
# 2) Increase the background sky level of the simulation - what happens to the image?
# 3) Increase the expousre time of the simulation - what happens as you change this and the background sky?
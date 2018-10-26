from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.lensing import ray_tracing
from autolens.galaxy import galaxy as g
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.plotting import ray_tracing_plotters
from autolens.plotting import imaging_plotters

# In this example, we'll use the 'imaging' module (imported as 'im') to 'simulate' an images of a strong lens made
# using a tracer. By simulate, we mean that it will appear as if we had observed it using a real telescope,
# with this example making an images representative of Hubble Space Telescope imaging.

# To simulate an images, we need to model the telescope's optics. We'll do this by convolving the images with a
# Point-Spread Function, which we can simulate as a Gaussian using the imaging module.
psf = im.PSF.simulate_as_gaussian(shape=(11, 11), pixel_scale=0.1, sigma=0.1)

# To simulate an images, we use a special type of grid. This pads the grids 2D dimensions relative to the PSF-shape,
# to ensure that the edge's of our simulated images are not degraded.
image_plane_grids = mask.ImagingGrids.grids_for_simulation(shape=(100, 100), pixel_scale=0.1, psf_shape=psf.shape)
print(image_plane_grids.image.image_shape)
print(image_plane_grids.image.padded_shape)

# Now, lets setup our lens model_galaxy, source model_galaxy and tracer.
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=45.0,
                                                        intensity=1.0, effective_radius=1.0, sersic_index=2.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)

# Lets look at the tracer's images-plane images - this is the images we'll be simulating.
ray_tracing_plotters.plot_image_plane_image(tracer=tracer)

# To simulate the images, we don't use the images-plane images plotted above. Instead, we use an  images-plane images
# which has been generated specifically for simulating an images, using the padded grid above that ensures
# edge-effects do not degrade our simulation's PSF convolution.
print(tracer.image_plane_images.shape)
print(tracer.image_plane_images_for_simulation.shape)

# Now, to simulate the images, we pass the tracer's images-plane images to the imaging module's simulate function. This
# adds the following effects:

# 1) Telescope optics: Using the Point Spread Function above.
# 2) The Background Sky: Although the images that is returned is automatically background sky subtracted.
# 3) Poisson noise: Due to the background sky, lens model_galaxy and source model_galaxy Poisson photon counts.

image_simulated = im.PreparatoryImage.simulate(array=tracer.image_plane_images_for_simulation, pixel_scale=0.1,
                                               exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

# Lets plot the images - we can see the images has been blurred due to the telescope optics and noise has been added.
imaging_plotters.plot_image_subplot(image=image_simulated)

# Finally, lets output these files to.fits files, we'll begin to analyze them in the next tutorial!
# Unfortunately, in a Jupyter notebook you have to manually specify the path to PyAutoLens and this tutorial.
path = '/home/jammy/PyCharm/Projects/AutoLens/workspace/howtolens/chapter_1_introduction'
im.output_imaging_to_fits(image=image_simulated, image_path=path+'/datas/images.fits',
                                                 noise_map_path=path+'/datas/noise_map.fits',
                                                 psf_path=path+'/datas/psf.fits',
                          overwrite=True)

# You've just completed your third tutorial, try the following:

# 1) Change the size (sigma) of the PSF, what happens to the lensed source model_galaxy's appearance?
# 2) Increase the background sky level of the simulation - what happens to the images?
# 3) Increase the expousre time of the simulation - what happens as you change this and the background sky?
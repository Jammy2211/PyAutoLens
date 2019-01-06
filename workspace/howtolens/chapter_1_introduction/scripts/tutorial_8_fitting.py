from autolens.data import ccd
from autolens.data.array import mask as ma
from autolens.lens import ray_tracing, lens_fit
from autolens.model.galaxy import galaxy as g
from autolens.lens import lens_data as ld
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.data.plotters import ccd_plotters
from autolens.lens.plotters import ray_tracing_plotters
from autolens.lens.plotters import lens_fit_plotters

# In this example, we'll fit the ccd imaging data we simulated in the previous exercise. We'll do this using model
# images generated via a tracer, and by comparing to the simulated image we'll get diagostics about the quality of the fit.

# If you are using Docker, the path you should use to output these images is (e.g. comment out this line)
# path = '/home/user/workspace/howtolens/chapter_1_introduction'

# If you arn't using docker, you need to change the path below to the chapter 2 directory and uncomment it
# path = '/path/to/user/workspace/howtolens/chapter_1_introduction'

path = '/home/jammy/PyCharm/Projects/AutoLens/workspace/howtolens/chapter_1_introduction'

ccd_data = ccd.load_ccd_data_from_fits(image_path=path + '/data/image.fits',
                                       noise_map_path=path+'/data/noise_map.fits',
                                       psf_path=path + '/data/psf.fits', pixel_scale=0.1)

# The variable ccd_data is a CCDData object, which is a 'package' of all components of the CCD data of the lens, in particular:

# 1) The image.
#
# 2) The Point Spread Function (PSF).
#
# 3) Its noise-map.
print('Image:')
print(ccd_data.image)
print('Noise-Map:')
print(ccd_data.noise_map)
print('PSF:')
print(ccd_data.psf)

# To fit an image, we first specify a mask. A mask describes the sections of the image that we fit.

# Typically, we want to mask out regions of the image where the lens and source galaxies are not visible, for example
# at the edges where the signal is entirely background sky and noise.

# For the image we simulated, a 3" circular mask will do the job.

mask = ma.Mask.circular(shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale, radius_arcsec=3.0)
print(mask) # 1 = True, which means the pixel is masked. Edge pixels are indeed masked.
print(mask[48:53, 48:53]) # Whereas central pixels are False and therefore unmasked.

# We can use a ccd_plotter to compare the mask and the image - this is useful if we really want to 'tailor' a
# mask to the lensed source's light (which in this example, we won't).
ccd_plotters.plot_image(ccd_data=ccd_data, mask=mask)

# Now we've loaded the ccd data and created a mask, we use them to create a 'lens data' object, which we'll perform
# using the lens_data module (imported as 'ld').

# A lens data object is a 'package' of all parts of a data-set we need in order to fit it with a lens model:

# 1) The ccd-data, e.g. the image, PSF (so that when we compare a tracer's image-plane image to the image data we
#    can include blurring due to the telescope optics) and noise-map (so our goodness-of-fit measure accounts for
#    noise in the observations).

# 2) The mask, so that only the regions of the image with a signal are fitted.

# 3) A grid-stack aligned to the ccd-imaging data's pixels: so the tracer's image-plane image is generated on the same
#    (masked) grid as the image.

lens_data = ld.LensData(ccd_data=ccd_data, mask=mask)
ccd_plotters.plot_image(ccd_data=ccd_data)

# By printing its attribute, we can see that it does indeed contain the image, mask, psf and so on
print('Image:')
print(lens_data.image)
print('Noise-Map:')
print(lens_data.noise_map)
print('PSF:')
print(lens_data.psf)
print('Mask')
print(lens_data.mask)
print('Grid')
print(lens_data.grid_stack.regular)

# The image, noise-map and grids are masked using the mask and mapped to 1D arrays for fast calcuations.
print(lens_data.image.shape) # This is the original 2D image
print(lens_data.image_1d.shape)
print(lens_data.noise_map_1d.shape)
print(lens_data.grid_stack.regular.shape)

# To fit an image, we need to create an image-plane image using a tracer.
# Lets use the same tracer we simulated the ccd data with (thus, our fit should be 'perfect').

# Its worth noting that below, we use the lens_data's grid-stack to setup the tracer. This ensures that our image-plane
# image will be the same resolution and alignment as our image-data, as well as being masked appropriately.

lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=45.0,
                                                        intensity=1.0, effective_radius=1.0, sersic_index=2.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grid_stack=lens_data.grid_stack)
ray_tracing_plotters.plot_image_plane_image(tracer=tracer)

# To fit the image, we pass the lens data and tracer to the fitting module. This performs the following:

# 1) Blurs the tracer's image-plane image with the lens data's PSF, ensuring that the telescope optics are
#    accounted for by the fit. This creates the fit's 'model_image'.

# 2) Computes the difference between this model_image and the observed image-data, creating the fit's 'residual_map'.

# 3) Divides the residuals by the noise-map and squaring each value, creating the fit's 'chi_squared_map'.

# 4) Sums up these chi-squared values and converts them to a 'likelihood', which quantities how good the tracer's fit
#    to the data was (higher likelihood = better fit).

fit = lens_fit.fit_lens_data_with_tracer(lens_data=lens_data, tracer=tracer)
lens_fit_plotters.plot_fit_subplot(fit=fit)

# We can print the fit's attributes - if we don't specify where we'll get all zeros, as the edges were masked:
print('Model-Image Edge Pixels:')
print(fit.model_image)
print('Residuals Edge Pixels:')
print(fit.residual_map)
print('Chi-Squareds Edge Pixels:')
print(fit.chi_squared_map)

# Of course, the central unmasked pixels have non-zero values.
print('Model-Image Central Pixels:')
print(fit.model_image[48:53, 48:53])
print('Residuals Central Pixels:')
print(fit.residual_map[48:53, 48:53])
print('Chi-Squareds Central Pixels:')
print(fit.chi_squared_map[48:53, 48:53])

# It also provides a likelihood, which is a single-figure estimate of how good the model image fitted the
# simulated image (in unmasked pixels only!).
print('Likelihood:')
print(fit.likelihood)

# We used the same tracer to create and fit the image. Therefore, our fit to the image was excellent.
# For instance, by inspecting the residuals and chi-squareds, one can see no signs of the source galaxy's light present,
# indicating a good fit.

# This solution should translate to one of the highest-likelihood solutions possible.

# Lets change the tracer, so that it's near the correct solution, but slightly off.
# Below, we slightly offset the lens galaxy, by 0.005"

lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.005, 0.005), einstein_radius=1.6, axis_ratio=0.7, phi=45.0))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=45.0,
                                                        intensity=1.0, effective_radius=1.0, sersic_index=2.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grid_stack=lens_data.grid_stack)
fit = lens_fit.fit_lens_data_with_tracer(lens_data=lens_data, tracer=tracer)
lens_fit_plotters.plot_fit_subplot(fit=fit)

# We now observe residuals to appear at the locations the source galaxy was observed, which
# corresponds to an increase in chi-squareds (which determines our goodness-of-fit).

# Lets compare the likelihood to the value we computed above (which was 11697.24):
print('Previous Likelihood:')
print(11697.24)
print('New Likelihood:')
print(fit.likelihood)
# It decreases! This model was a worse fit to the data.

# Lets change the tracer, one more time, to a solution that is nowhere near the correct one.
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.005, 0.005), einstein_radius=1.3, axis_ratio=0.8, phi=45.0))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.7, phi=65.0,
                                                        intensity=1.0, effective_radius=0.4, sersic_index=3.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grid_stack=lens_data.grid_stack)
fit = lens_fit.fit_lens_data_with_tracer(lens_data=lens_data, tracer=tracer)
lens_fit_plotters.plot_fit_subplot(fit=fit)

# Clearly, the model provides a terrible fit, and this tracer is not a plausible representation of
# the image-data  (of course, we already knew that, given that we simulated it!)

# The likelihood drops dramatically, as expected.
print('Previous Likelihoods:')
print(11697.24)
print(10319.44)
print('New Likelihood:')
print(fit.likelihood)

# Congratulations, you've fitted your first strong lens with PyAutoLens! Perform the following exercises:

# 1) In this example, we 'knew' the correct solution, because we simulated the lens ourselves. In the real Universe,
#    we have no idea what the correct solution is. How would you go about finding the correct solution?
#    Could you find a solution that fits the data reasonable through trial and error?
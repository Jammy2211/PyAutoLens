from autolens.data.imaging import image as im
from autolens.data.array import mask as ma
from autolens.lensing import lensing_fitting
from autolens.lensing import ray_tracing
from autolens.model.galaxy import galaxy as g
from autolens.lensing import lensing_image as li
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.data.imaging.plotters import imaging_plotters
from autolens.lensing.plotters import ray_tracing_plotters
from autolens.lensing.plotters import lensing_fitting_plotters

# In this example, we'll fit_normal the regular we simulated in the previous exercise. We'll do this using model regular generated
# via a tracer_without_subhalo, and by compairing to the simulated regular we'll get diagostics about the quality of the fit_normal.

# First, we load the regular-datas we simualted in the previous tutorial.
path = 'path/to/AutoLens/howtolens/1_introduction' # Unfortunately, in a Jupyter notebook you have to manually specify the path to PyAutoLens and this tutorial.
path = '/home/jammy/PyCharm/Projects/AutoLens/workspace/howtolens/chapter_1_introduction'
image = im.load_imaging_from_fits(image_path=path + '/data/regular.fits',
                                  noise_map_path=path+'/data/noise_map.fits',
                                  psf_path=path + '/data/psf.fits', pixel_scale=0.1)

# To fit_normal an regular, we first specify a masks. A masks describes the sections of the regular that we fit_normal.

# Typically, we want to masks out regions of the regular where the lens and source galaxies are not visible, for example
# at the edges where the signal is entirely background sky and noise.

# For the regular we simulated, a 3" circular masks will do the job.

mask = ma.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale, radius_mask_arcsec=3.0)
print(mask) # 1 = True, which means the pixel is masked. Edge pixels are indeed masked.
print(mask[48:53, 48:53]) # Whereas central pixels are False and therefore unmasked.

# We can use an imaging_plotter to compare the masks and the regular - this is useful if we really want to 'tailor' a
# masks to the lensed source's light (which in this example, we won't).
imaging_plotters.plot_image(image=image, mask=mask)

# Now we've loaded the regular and created a masks, we use them to create a 'lensing regular', which we'll perform using the
# lensing_module (imported as 'li').

# A lensing regular is a 'package' of all parts of the the regular datas we need in order to fit_normal it:

# 1) The regular.

# 2) The PSF: so that when we compare a tracer_without_subhalo's regular-plane regular to the regular datas we can include blurring due to
#    the telescope optics.

# 3) The noise-map: so our goodness-of-fit_normal measure accounts for noise in the observations.

# 4) The regular's grids: so the tracer_without_subhalo's regular-plane regular is generated on the same (masked) grid as the regular-datas.

lensing_image = li.LensingImage(image=image, mask=mask)
imaging_plotters.plot_image_subplot(lensing_image.image)

# By printing its attribute, we can see that it does indeed contain the regular, masks, psf and so on
print('Image:')
print(lensing_image.image)
print('Noise-Map:')
print(lensing_image.image.noise_map)
print('PSF:')
print(lensing_image.image.psf)
print('Mask')
print(lensing_image.mask)
print('Grid')
print(lensing_image.grids.regular)

# The shapes of these grids reveals they are 1D and have been masked:
print(lensing_image.image.shape) # This is the original 2D regular
print(lensing_image.shape)
print(lensing_image.noise_map.shape)
print(lensing_image.grids.regular.shape)

# To fit_normal an regular, we need to create an regular-plane regular using a tracer_without_subhalo.
# Lets use the same tracer_without_subhalo we simulated the regular with (thus, our fit_normal should be 'perfect').

# Its worth noting that below, we use the sensitivity_image's grids to setup the tracer_without_subhalo. This ensures that our regular-plane
# regular will be the same resolution and alignment as our regular-datas, as well as being masked appropriately.

lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=45.0,
                                                        intensity=1.0, effective_radius=1.0, sersic_index=2.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=[lensing_image.grids])
ray_tracing_plotters.plot_image_plane_image(tracer=tracer)

# To fit_normal the regular, we pass the lensing regular and tracer_without_subhalo to the fitting module. This performs the following:

# 1) Blurs the tracer_without_subhalo's regular-plane regular with the lensing-regular's PSF, ensuring that the telescope optics are
#    accounted for by the fit_normal. This creates the fit_normal's 'model_image'.

# 2) Computes the difference between this model_image and the observed regular-datas, creating the fit_normal's 'residuals'.

# 3) Divides the residuals by the noise-map and squaring each value, creating the fits 'chi-squareds'.

# 4) Sums up these chi-squared values and converts them to a 'likelihood', which quantities how good the tracer_without_subhalo's fit_normal
#    to the datas was (higher likelihood = better fit_normal).

fit = lensing_fitting.fit_lensing_image_with_tracer(lensing_image=lensing_image, tracer=tracer)
lensing_fitting_plotters.plot_fitting_subplot(fit=fit)

# We can print the fit_normal's attributes - if we don't specify where we'll get all zeros, as the edges were masked:
print('Model-Image Edge Pixels:')
print(fit.model_data)
print('Residuals Edge Pixels:')
print(fit.residual)
print('Chi-Squareds Edge Pixels:')
print(fit.chi_squared)

# Of course, the central unmasked pixels have non-zero values.
print('Model-Image Central Pixels:')
print(fit.model_data[48:53, 48:53])
print('Residuals Central Pixels:')
print(fit.residual[48:53, 48:53])
print('Chi-Squareds Central Pixels:')
print(fit.chi_squared[48:53, 48:53])

# It also provides a likelihood, which is a single-figure estimate of how good the model regular fitted the
# simulated regular (in unmasked pixels only!).
print('Likelihood:')
print(fit.likelihood)

# We used the same tracer_without_subhalo to create and fit_normal the regular. Therefore, our fit_normal to the regular was excellent.
# For example, by inspecting the residuals and chi-squareds, one can see no signs of the source model_galaxy's light present
# and we only see the noise that we simulated the regular with.

# This solution should translate to one of the highest-likelihood solutions possible.

# Lets change the tracer_without_subhalo, so that it's near the correct solution, but slightly off.
# All we're going to do is slightly offset the lens model_galaxy, by 0.005"

lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.005, 0.005), einstein_radius=1.6, axis_ratio=0.7, phi=45.0))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=45.0,
                                                        intensity=1.0, effective_radius=1.0, sersic_index=2.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=[lensing_image.grids])
fit = lensing_fitting.fit_lensing_image_with_tracer(lensing_image=lensing_image, tracer=tracer)
lensing_fitting_plotters.plot_fitting_subplot(fit=fit)

# We now observe residuals to appear at the locations the source model_galaxy was observed, which
# corresponds to an increase in our chi-squareds (which determines our goodness-of-fit_normal).

# Lets compare the likelihood to the value we computed above (which was 11697.24):
print('Previous Likelihood:')
print(11697.24)
print('New Likelihood:')
print(fit.likelihood)
# It decreases! This model was a worse fit_normal to the datas.

# Lets change the tracer_without_subhalo, one more time, to a solution that is nowhere near the correct one.
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.005, 0.005), einstein_radius=1.3, axis_ratio=0.8, phi=45.0))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.7, phi=65.0,
                                                        intensity=1.0, effective_radius=0.4, sersic_index=3.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=[lensing_image.grids])
fit = lensing_fitting.fit_lensing_image_with_tracer(lensing_image=lensing_image, tracer=tracer)
lensing_fitting_plotters.plot_fitting_subplot(fit=fit)

# Clearly, the model provides a terrible fit_normal, and this tracer_without_subhalo is not a plausible representation of
# the regular-datas  (of course, we already knew that, given that we simulated it!)

# The likelihood drops dramatically, as expected.
print('Previous Likelihoods:')
print(11697.24)
print(10319.44)
print('New Likelihood:')
print(fit.likelihood)

# Congratulations, you've fitted your first strong lens with PyAutoLens! Perform the following exercises:

# 1) In this example, we 'knew' the correct solution, because we simulated the lens ourselves. In the real Universe,
#    we have no idea what the correct solution is. How would you go about finding the correct solution?
#    Could you find a solution that fits the datas reasonable through trial and error?
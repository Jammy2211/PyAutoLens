from autolens.imaging import image as im
from autolens.imaging import mask as ma
from autolens.profiles import mass_profiles as mp
from autolens.profiles import light_profiles as lp
from autolens.lensing import galaxy as g
from autolens.lensing import ray_tracing
from autolens.lensing import lensing_image as li
from autolens.lensing import fitting
from autolens.inversion import pixelizations as pix
from autolens.inversion import regularization as reg
from autolens.plotting import fitting_plotters
from autolens.plotting import inversion_plotters

# So, we've seen that we can use an inversion to reconstruct an image. Furthermore, this reconstruction provides
# us with the 'best-fit' solution. And indeed, when we inspect the fit with the fitting module, we saw that we got a
# good fit.


# Everything sounds pretty good, doesn't it? You're probably thinking, why are there more tutorials? We can use
# invesions now, don't ruin it! Well, there is a problem - which I hid from you in the last tutorial.

# Setup the path
path = '/home/jammy/PyCharm/Projects/AutoLens/workspace/howtolens/4_inversions'

# Lets go back to our simple source
def simulate():

    from autolens.imaging import mask
    from autolens.lensing import galaxy as g
    from autolens.lensing import ray_tracing

    psf = im.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=0.05)

    image_plane_grids = mask.ImagingGrids.grids_for_simulation(shape=(180, 180), pixel_scale=0.05, psf_shape=(11, 11))

    lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0,
                                                        einstein_radius=1.6))
    source_galaxy_0 = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=90.0, intensity=0.2,
                                                         effective_radius=0.3, sersic_index=1.0))

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy],
                                                 source_galaxies=[source_galaxy_0],
                                                 image_plane_grids=image_plane_grids)

    return im.PreparatoryImage.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.05,
                                        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

# We're going to perform a lot of fits using an inversion this tutorial. This would create a lot of code, so to keep
# things tidy, I've setup this function which handles it all for us.
def perform_fit_with_source_galaxy(source_galaxy):

    image = simulate()
    mask = ma.Mask.annular(shape=image.shape, pixel_scale=image.pixel_scale, inner_radius_arcsec=0.5,
                           outer_radius_arcsec=2.2)
    lensing_image = li.LensingImage(image=image, mask=mask)
    lens_galaxy = g.Galaxy(
        mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0, einstein_radius=1.6))
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grids=lensing_image.grids, borders=lensing_image.borders)
    return fitting.fit_lensing_image_with_tracer(lensing_image=lensing_image, tracer=tracer)

# Okay, so lets look at our fit from the previous tutorial in more detail. We'll use a higher resolution grid.
source_galaxy = g.Galaxy(pixelization=pix.Rectangular(shape=(40, 40)), regularization=reg.Constant(coefficients=(1.0,)))
fit = perform_fit_with_source_galaxy(source_galaxy=source_galaxy)
fitting_plotters.plot_fitting_subplot(fit=fit)

# It still looks pretty good! However, this is because I sneakily chose a regularization coefficient that gives a,
# good looking solution, without telling you.

# So, what does regularization actually do? When our inversion reconstructs a source, it doesn't *just* compute the set
# of fluxes that best fit the image. It is also regularized, whereby we go to every pixel on our rectangular grid and
# compare its reconstructed flux with its neighboring pixels. If the difference in flux is large, we penalize the
# solution, reducing its likelihood. You can think of this as us 'smoothing' our solution.

# So, why do we need regularization? Well, lets see what happens if we turn it off, by setting our regularization
# coefficient to zero.
source_galaxy = g.Galaxy(pixelization=pix.Rectangular(shape=(40, 40)), regularization=reg.Constant(coefficients=(0.0,)))
no_regularization_fit = perform_fit_with_source_galaxy(source_galaxy=source_galaxy)
fitting_plotters.plot_fitting_subplot(fit=no_regularization_fit)

# Our source-reconstruction looks like complete garbage! The source has completely disappeared! Well, actually, it
# hasn't - it is there, but the large reconstructed flux values at the exterior of the grid are making it non-visible.
# We can see the source is there by changing the 'normalization' variables of the plotter, such that the color-map
# is restricuted to a range of values.
inversion_plotters.plot_reconstructed_pixelization(inversion=fit.inversion, norm_max=1.0, norm_min=-1.0)

# Clearly, there source is present, so what are all the weird high flux-values at the exterior of the source
# reconstruction? This, my friend, is called noise-fitting, or over-fitting. Basically, when we fit the image with a
# linear inversion, its going to try and fit *everything* so as to provide the best-fit solution. This includes
# the lensed source, which is good, but it also includes the noise in the image, which is really really bad.

# This is why regularization is necessary. We have to smooth our source reconstruction to ensure it doesn't fit the
# noise in the image. If we set a really high regularization coefficient, we can completely remove overfitting, at
# the expense of also fitting the lensed source less accurately.
source_galaxy = g.Galaxy(pixelization=pix.Rectangular(shape=(40, 40)),
                         regularization=reg.Constant(coefficients=(100.0,)))
high_regularization_fit = perform_fit_with_source_galaxy(source_galaxy=source_galaxy)
fitting_plotters.plot_fitting_subplot(fit=high_regularization_fit)

# So there we have it, we now understand regularization and its purpose. But theres one nagging question that remains,
# how do I choose the regularization coefficient? We can't use our likelihood, as increasing the regularization
# coefficiently will always decrease the likelihood -  if we use the likelihood we'll choose a coefficient of 0!
print('Likelihood Without Regularization:')
print(no_regularization_fit.likelihood)
print('Likelihood With Normal Regularization:')
print(fit.likelihood)
print('Likelihood With High Regularization:')
print(high_regularization_fit.likelihood)

# Regularization is built into the linear-inversion. When the inversion provides the 'best-fit' solution, it includes
# the penalty term due to this smoothing. That is, an additional 'regularization term' is subtracted from our
# likelihood, which increases as neighboring pixels have larger flux difference or if the regularization coefficient
# is increased.
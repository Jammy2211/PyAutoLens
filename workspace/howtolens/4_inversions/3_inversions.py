from autolens.imaging import image as im
from autolens.imaging import mask as ma
from autolens.profiles import mass_profiles as mp
from autolens.profiles import light_profiles as lp
from autolens.lensing import galaxy as g
from autolens.lensing import ray_tracing
from autolens.lensing import lensing_image as li
from autolens.inversion import pixelizations as pix
from autolens.inversion import regularization as reg
from autolens.inversion import inversions as inv
from autolens.plotting import imaging_plotters
from autolens.plotting import mapper_plotters
from autolens.plotting import inversion_plotters

# We've covered mappers, which map source-pixels to an image, and visa versa. Now, we're look at how we can use a
# mapper to reconstruct the source galaxy - I hope you're excited!

# Setup the path
path = '/home/jammy/PyCharm/Projects/AutoLens/workspace/howtolens/4_inversions'

# We'll simulate two lenses in this tutorial, one with a simple source and one with a complex source.

def simulate():

    from autolens.imaging import mask
    from autolens.lensing import galaxy as g
    from autolens.lensing import ray_tracing

    psf = im.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=0.05)

    image_plane_grids = mask.ImagingGrids.grids_for_simulation(shape=(180, 180), pixel_scale=0.05, psf_shape=(11, 11))

    lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0,
                                                        einstein_radius=1.6))
    source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0., 0.), axis_ratio=0.8, phi=90.0, intensity=0.2,
                                                         effective_radius=1.0, sersic_index=1.5))
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grids=image_plane_grids)

    return im.PreparatoryImage.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.05,
                                        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

# Lets siulate the simple source, mask it, and use a plot to check the mask covers the lensed source.
image = simulate()
mask = ma.Mask.annular(shape=image.shape, pixel_scale=image.pixel_scale,
                       inner_radius_arcsec=1.0, outer_radius_arcsec=2.2)
imaging_plotters.plot_image(image=image, mask=mask)

# Now, lets set this image up as a lensing image, and setup a tracer using the input lens galaxy model (we don't need
# to provide the source's light profile, as we're using a mapper to reconstruct it).
lensing_image = li.LensingImage(image=image, mask=mask, sub_grid_size=1)
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0, einstein_radius=1.6))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[g.Galaxy()],
                                             image_plane_grids=lensing_image.grids)

# Now, lets setup the pixelization and mapper we'll use to perform the reconstruction
rectangular = pix.Rectangular(shape=(25, 25))
mapper = rectangular.mapper_from_grids(grids=tracer.source_plane.grids)
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper)

# We're now going to use our mapper to invert the image, using the 'inversions' module, which is imported as 'inv'.
# We'll think about how this works in a second - but lets perform the inversion first, to see how it looks.
# (Ignore the 'regularization' input below for now, we'll cover this in the next tutorial).
inversion = inv.Inversion(image=lensing_image[:], noise_map=lensing_image.noise_map,
                          convolver=lensing_image.convolver_mapping_matrix, mapper=mapper,
                          regularization=reg.Constant(coeffs=(1.0,)))

# Our inversion has a reconstructed image and pixeilzation, whcih we can plot using an inversion plotter
inversion_plotters.plot_reconstructed_image(inversion=inversion)
inversion_plotters.plot_reconstructed_pixelization(inversion=inversion, should_plot_grid=True)

# We've successfully reconstructed, or, *inverted*, our source using the mapper's rectangular grid. Whilst this source
# was simple (a blob of light in the centre of the source-plane), we'll look at the inversion of a complex source in a
# moment. However, we've first got to ask, how does an inversion actually work?

# Lets look again at the mappings between our mapper's source-pixels and the image
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper, source_pixels=[[8], [12], [16]])

# These mappings are known before the inversion, which means pre-inversion we know two key pieces of information:

# 1) The mappings between every source-pixel and a set of image-pixels.
# 2) The flux values in every observed image-pixel, which are the values we want to fit successfully.

# It turns out that these two pieces of information make it a linear problem to compute the set of source-pixel
# fluxes that best-fit (e.g. maximize the likelihood) our observed image. This process is called a 'linear inversion',
# and it guarantee that the image reconstructiono provides the best-fit solution to the iamge (e.g. the one that
# maximizes the likelihood, or equivalent, minimizes the chi-squareds).

# At this point in the tutorial, I'm going to give you a choice. If you want to dive deeper into the world of linear
# algrebra, you can go to the optional tutorial, 'advanced.py', to understand how this linear inversion works. However,
# the technical details of an inversion works arn't paramount to being good at lens modeling, so don't feel that you
# *need* to do this tutorial.

# Either way, there are three more things about a linear inversion that are worth knowing, before we finish:

# 1) We've discussed the image sub-grid before, which splits each image-pixel into a sub-pixel. Well, if a sub-grid is
#    being used, it is the mapping between every sub-pixel and source-pixel that is computed and used to perform the
#    inversion. This prevents aliasing effects degrading the image reconstruction, and, as a rule of thumb, I would
#    suggest you use sub-gridding of degree 2x2 or 4x4.

# 2) When fitting image's using light profiles, we discussed how a 'model_image' was generated by blurring them with
#    the instrument's PSF. A similar blurring operation is incorporated into the inversion, such that the reconstructed
#    image and source fully account for the telescope optics and effect of the PSF.

# 3) The inversion's solution is regularized. But wait, that's what we'll cover in the next tutorial!

# Before we explore regularization, here are a few questions to get you thinking about it.

# 1) The inversion provides the best-fit to the observed image. Is there any problem with seeking the 'best-fit'? Is
#    there a risk that we're going to fit other things in the image than just the lensed source galaxy? What happens
#    if you reduce the regularization 'coeff' above to zero?

# 2) The exterior pixels in the rectangular grid have no image-pixels in them. However, they are still given a
#    reconstructed flux. If this value isn't' coming from a mapping to an image-pixel, where could it be coming from?




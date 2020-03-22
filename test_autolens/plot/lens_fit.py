from autolens.fit.plotters import masked_imaging_fit_plotters
from test import simulate_util

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
imaging = simulate_util.load_test_imaging(
    data_type="lens_light_dev_vaucouleurs", data_resolution="lsst"
)
mask = al.mask.circular(
    shape_2d=imaging.shape, pixel_scales=imaging.pixel_scales, radius=3.0
)

# The lines of code below do everything we're used to, that is, setup an image and its grid, mask it, trace it
# via a tracer, setup the rectangular mapper, etc.
lens_galaxy = al.Galaxy(
    bulge=al.EllipticalDevVaucouleurs(
        centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0, intensity=0.1, effective_radius=1.0
    )
)

masked_imaging = al.masked_imaging(imaging=imaging, mask=mask)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy])
fit = al.fit(masked_dataset=masked_imaging, tracer=tracer)
aplt_array.fit_imaging.subplot_imaging(fit=fit, mask=True)

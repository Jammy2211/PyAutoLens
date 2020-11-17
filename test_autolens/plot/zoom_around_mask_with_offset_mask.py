import autolens as al


# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
imaging = instrument_util.load_test_imaging(
    dataset_name="light_dev_vaucouleurs", instrument="vro"
)
mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d,
    pixel_scales=imaging.pixel_scales,
    radius=3.0,
    centre=(4.0, 4.0),
)

# The lines of code below do everything we're used to, that is, setup an image and its grid, mask it, trace it
# via a tracer, setup the rectangular mapper, etc.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.EllipticalDevVaucouleurs(
        centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0, intensity=0.1, effective_radius=1.0
    ),
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy])
fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)
aplt.FitImaging.subplot_fit_imaging(fit=fit)

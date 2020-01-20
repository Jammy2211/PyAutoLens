from autolens.fit.plotters import masked_imaging_fit_plotters
from test import simulate_util

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
imaging = simulate_util.load_test_imaging(
    data_type="lens_light_dev_vaucouleurs", data_resolution="lsst"
)
mask = al.mask.elliptical(
    shape=imaging.shape,
    pixel_scales=imaging.pixel_scales,
    major_axis_radius=3.0,
    axis_ratio=1.0,
    phi=0.0,
    centre=(0.0, 0.0),
)

# aplt.imaging.subplot(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='equal')
# aplt.imaging.subplot(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='auto')

# aplt.imaging.image(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='square')
# aplt.imaging.image(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='equal')

# The lines of code below do everything we're used to, that is, setup an image and its grid, mask it, trace it
# via a tracer, setup the rectangular mapper, etc.
lens_galaxy = al.Galaxy(
    mass=al.mp.EllipticalIsothermal(
        centre=(1.0, 1.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    )
)
source_galaxy = al.Galaxy(
    pixelization=al.pix.VoronoiMagnification(shape=(20, 20)),
    regularization=al.reg.instance(coefficient=1.0),
)

masked_imaging = al.masked.imaging(imaging=imaging, mask=mask)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
fit = al.fit(masked_dataset=masked_imaging, tracer=tracer)


aplt_array.fit_imaging.subplot_imaging(
    fit=fit, mask=True, include_image_plane_pix=True, aspect="auto"
)

aplt_array.fit_imaging.subplot_imaging(
    fit=fit, mask=True, include_image_plane_pix=True, aspect="equal"
)

aplt_array.fit_imaging.subplot_imaging(
    fit=fit, mask=True, include_image_plane_pix=True, aspect="square"
)

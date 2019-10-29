from autolens.fit.plotters import masked_imaging_fit_plotters
from test import simulation_util

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
imaging = simulation_util.load_test_imaging(
    data_type="lens_light_dev_vaucouleurs", data_resolution="LSST"
)
mask = al.mask.elliptical(
    shape=imaging.shape,
    pixel_scales=imaging.pixel_scales,
    major_axis_radius_arcsec=3.0,
    axis_ratio=1.0,
    phi=0.0,
    centre=(0.0, 0.0),
)

# imaging_plotters.plot_imaging_subplot(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='equal')
# imaging_plotters.plot_imaging_subplot(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='auto')

# imaging_plotters.plot_image(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='square')
# imaging_plotters.plot_image(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='equal')

# The lines of code below do everything we're used to, that is, setup an image and its al.ogrid, mask it, trace it
# via a tracer, setup the rectangular mapper, etc.
lens_galaxy = al.Galaxy(
    mass=al.mp.EllipticalIsothermal(
        centre=(1.0, 1.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    )
)
source_galaxy = al.Galaxy(
    pixelization=al.pix.VoronoiMagnification(shape=(20, 20)),
    regularization=al.reg.Constant(coefficient=1.0),
)

masked_imaging = al.LensData(imaging=imaging, mask=mask)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
fit = al.LensImageFit.from_masked_data_and_tracer(
    masked_imaging=masked_imaging, tracer=tracer
)


masked_imaging_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask_overlay=True,
    should_plot_image_plane_pix=True,
    aspect="auto",
)

masked_imaging_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask_overlay=True,
    should_plot_image_plane_pix=True,
    aspect="equal",
)

masked_imaging_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask_overlay=True,
    should_plot_image_plane_pix=True,
    aspect="square",
)

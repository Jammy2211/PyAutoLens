import autolens as al


# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
imaging = simulate_util.load_test_imaging(
    data_type="lens_light_dev_vaucouleurs", data_resolution="lsst"
)
mask = al.mask.elliptical(
    shape_2d=imaging.shape_2d,
    pixel_scales=imaging.pixel_scales,
    major_axis_radius=3.0,
    axis_ratio=0.5,
    phi=0.0,
    centre=(0.0, 0.0),
)

# al.plot.imaging.subplot(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='equal')
# al.plot.imaging.subplot(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='auto')

# al.plot.imaging.image(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='square')
# al.plot.imaging.image(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='equal')

# The lines of code below do everything we're used to, that is, setup an image and its grid, mask it, trace it
# via a tracer, setup the rectangular mapper, etc.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
)
source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.Rectangular(shape=(20, 20)),
    regularization=al.reg.Constant(coefficient=1.0),
)

masked_imaging = al.masked.imaging(imaging=imaging, mask=mask)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
fit = al.fit(masked_dataset=masked_imaging, tracer=tracer)

al.plot.fit_imaging.subplot(
    fit=fit, include_mask=True, include_image_plane_pix=True, aspect="auto"
)

al.plot.fit_imaging.subplot(
    fit=fit, include_mask=True, include_image_plane_pix=True, aspect="equal"
)

al.plot.fit_imaging.subplot(
    fit=fit, include_mask=True, include_image_plane_pix=True, aspect="square"
)

from autolens.lens.plotters import lens_imaging_fit_plotters
from test.simulation import simulation_util

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
imaging_data = simulation_util.load_test_imaging_data(
    data_type="lens_light_dev_vaucouleurs", data_resolution="LSST"
)
mask = al.Mask.elliptical(
    shape=imaging_data.shape,
    pixel_scale=imaging_data.pixel_scale,
    major_axis_radius_arcsec=3.0,
    axis_ratio=0.5,
    phi=0.0,
    centre=(0.0, 0.0),
)

# imaging_plotters.plot_imaging_subplot(imaging_data=imaging_data, mask=mask, zoom_around_mask=True, aspect='equal')
# imaging_plotters.plot_imaging_subplot(imaging_data=imaging_data, mask=mask, zoom_around_mask=True, aspect='auto')

# imaging_plotters.plot_image(imaging_data=imaging_data, mask=mask, zoom_around_mask=True, aspect='square')
# imaging_plotters.plot_image(imaging_data=imaging_data, mask=mask, zoom_around_mask=True, aspect='equal')

# The lines of code below do everything we're used to, that is, setup an image and its al.ogrid, mask it, trace it
# via a tracer, setup the rectangular mapper, etc.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mass_profiles.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
)
source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.light_profiles.EllipticalSersic(
        centre=(1.0, 1.0),
        axis_ratio=0.8,
        phi=60.0,
        intensity=0.4,
        effective_radius=0.5,
        sersic_index=1.0,
    ),
)

lens_data = al.LensData(imaging_data=imaging_data, mask=mask)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
fit = al.LensImageFit.from_lens_data_and_tracer(lens_data=lens_data, tracer=tracer)

lens_imaging_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
    should_plot_image_plane_pix=True,
    aspect="auto",
)

stop

lens_imaging_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
    should_plot_image_plane_pix=True,
    aspect="equal",
)

lens_imaging_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
    should_plot_image_plane_pix=True,
    aspect="square",
)

from autolens.data.instrument import abstract_data
from autolens.data.instrument import ccd
from autolens.array import mask as msk
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.model.galaxy import galaxy as g
from autolens.lens import ray_tracing
from autolens.lens import lens_fit
from autolens.lens import lens_data as ld
from autolens.data.plotters import ccd_plotters
from autolens.lens.plotters import lens_fit_plotters
from test.simulation import simulation_util

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
ccd_data = simulation_util.load_test_ccd_data(
    data_type="lens_light_dev_vaucouleurs", data_resolution="LSST"
)
mask = al.Mask.elliptical(
    shape=ccd_data.shape,
    pixel_scale=ccd_data.pixel_scale,
    major_axis_radius_arcsec=3.0,
    axis_ratio=0.5,
    phi=0.0,
    centre=(0.0, 0.0),
)

# ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, mask=mask, zoom_around_mask=True, aspect='equal')
# ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, mask=mask, zoom_around_mask=True, aspect='auto')

# ccd_plotters.plot_image(ccd_data=ccd_data, mask=mask, zoom_around_mask=True, aspect='square')
# ccd_plotters.plot_image(ccd_data=ccd_data, mask=mask, zoom_around_mask=True, aspect='equal')

# The lines of code below do everything we're used to, that is, setup an image and its al.ogrid, mask it, trace it
# via a tracer, setup the rectangular mapper, etc.
lens_galaxy = al.Galaxy(
    mass=al.mass_profiles.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    )
)
source_galaxy = al.Galaxy(
    pixelization=al.pixelizations.VoronoiMagnification(shape=(20, 20)),
    regularization=al.regularization.Constant(coefficient=1.0),
)

lens_data = al.LensData(ccd_data=ccd_data, mask=mask)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
fit = al.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
    should_plot_image_plane_pix=True,
    aspect="auto",
)

lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
    should_plot_image_plane_pix=True,
    aspect="equal",
)

lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
    should_plot_image_plane_pix=True,
    aspect="square",
)

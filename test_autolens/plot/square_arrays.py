from autolens.fit.plotters import masked_imaging_fit_plotters
from test_autolens.simulators.imaging import instrument_util

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
imaging = instrument_util.load_test_imaging(
    dataset_name="light_dev_vaucouleurs", instrument="vro"
)
mask = al.Mask2D.elliptical(
    shape=imaging.shape_2d,
    pixel_scales=imaging.pixel_scales,
    major_axis_radius=3.0,
    axis_ratio=1.0,
    phi=0.0,
    centre=(0.0, 0.0),
)

# aplt.Imaging.subplot(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='equal')
# aplt.Imaging.subplot(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='auto')

# aplt.Imaging.image(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='square')
# aplt.Imaging.image(imaging=imaging, mask=mask, zoom_around_mask=True, aspect='equal')

# The lines of code below do everything we're used to, that is, setup an image and its grid, mask it, trace it
# via a tracer, setup the rectangular mapper, etc.
lens_galaxy = al.PipelineLight(
    mass=al.mp.EllipticalIsothermal(
        centre=(1.0, 1.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    )
)
source_galaxy = al.PipelineLight(
    pixelization=al.pix.VoronoiMagnification(shape=(20, 20)),
    regularization=al.reg.instance(coefficient=1.0),
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)


aplt.FitImaging.subplot_fit_imaging(
    fit=fit, mask=True, include_image_plane_pix=True, aspect="auto"
)

aplt.FitImaging.subplot_fit_imaging(
    fit=fit, mask=True, include_image_plane_pix=True, aspect="equal"
)

aplt.FitImaging.subplot_fit_imaging(
    fit=fit, mask=True, include_image_plane_pix=True, aspect="square"
)

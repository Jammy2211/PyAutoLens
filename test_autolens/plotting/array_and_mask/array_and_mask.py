from test import simulate_util
from autolens.plotters import array_plotters

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
imaging = simulate_util.load_test_imaging(
    data_type="lens_light_dev_vaucouleurs", data_resolution="lsst"
)
array = imaging.image

mask = al.mask.elliptical(
    shape=imaging.shape,
    pixel_scales=imaging.pixel_scales,
    major_axis_radius_arcsec=3.0,
    axis_ratio=1.0,
    phi=0.0,
    centre=(0.0, 0.0),
)
al.plot.array(array=array)

from autolens.array import mask as msk
from test.simulation import simulation_util
from autolens.plotters import array_plotters

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses irregular 'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
ccd_data = simulation_util.load_test_ccd_data(
    data_type="lens_light_dev_vaucouleurs", data_resolution="LSST"
)
array = ccd_data.image

mask = msk.Mask.elliptical(
    shape=ccd_data.shape,
    pixel_scale=ccd_data.pixel_scale,
    major_axis_radius_arcsec=3.0,
    axis_ratio=1.0,
    phi=0.0,
    centre=(0.0, 0.0),
)
array_plotters.plot_array(array=array)

from autolens.array import mask as msk
from test.simulation import simulation_util
from autolens.plotters import array_plotters

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
ccd_data = simulation_util.load_test_ccd_data(
    data_type="lens_light_dev_vaucouleurs", data_resolution="LSST"
)
array = ccd_data.image

mask = msk.Mask.circular(
    shape=ccd_data.shape,
    pixel_scale=ccd_data.pixel_scale,
    radius_arcsec=5.0,
    centre=(0.0, 0.0),
)
array_plotters.plot_array(
    array=array,
    mask=mask,
    positions=[[[1.0, 1.0]]],
    centres=[[(0.0, 0.0)]],
    zoom_around_mask=True,
    extract_array_from_mask=True,
)

ccd_data = simulation_util.load_test_ccd_data(
    data_type="lens_sis__source_smooth__offset_centre", data_resolution="LSST"
)
array = ccd_data.image

mask = msk.Mask.circular(
    shape=ccd_data.shape,
    pixel_scale=ccd_data.pixel_scale,
    radius_arcsec=5.0,
    centre=(1.0, 1.0),
)
array_plotters.plot_array(
    array=array, mask=mask, positions=[[[2.0, 2.0]]], centres=[[(1.0, 1.0)]]
)
array_plotters.plot_array(
    array=array,
    mask=mask,
    positions=[[[2.0, 2.0]]],
    centres=[[(1.0, 1.0)]],
    zoom_around_mask=True,
    extract_array_from_mask=True,
)

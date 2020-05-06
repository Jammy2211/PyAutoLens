from test_autolens.simulate.imaging import simulate_util
from autolens.plot import plotters

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
imaging = simulate_util.load_test_imaging(
    data_type="lens_light_dev_vaucouleurs", data_resolution="lsst"
)
array = imaging.image

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d,
    pixel_scales=imaging.pixel_scales,
    radius=5.0,
    centre=(0.0, 0.0),
)
aplt.Array(array=array, mask=mask, positions=[[(1.0, 1.0)]], centres=[[(0.0, 0.0)]])

imaging = simulate_util.load_test_imaging(
    data_type="lens_sis__source_smooth__offset_centre", data_resolution="lsst"
)
array = imaging.image

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d,
    pixel_scales=imaging.pixel_scales,
    radius=5.0,
    centre=(1.0, 1.0),
)
aplt.Array(array=array, mask=mask, positions=[[(2.0, 2.0)]], centres=[[(1.0, 1.0)]])
aplt.Array(array=array, mask=mask, positions=[[(2.0, 2.0)]], centres=[[(1.0, 1.0)]])

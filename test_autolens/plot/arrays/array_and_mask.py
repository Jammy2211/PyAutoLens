import autolens as al
import autolens.plot as aplt
from test_autolens.simulators.imaging import instrument_util

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
imaging = instrument_util.load_test_imaging(
    dataset_name="light_dev_vaucouleurs", instrument="vro"
)

array = imaging.image

mask = al.Mask2D.elliptical(
    shape_2d=imaging.shape_2d,
    pixel_scales=imaging.pixel_scales,
    major_axis_radius=3.0,
    axis_ratio=1.0,
    phi=0.0,
    centre=(3.0, 0.0),
)

aplt.Array(array=array, mask=mask)

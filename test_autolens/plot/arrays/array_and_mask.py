import autolens as al
import autolens.plot as aplt
from test_autolens.simulators.imaging import instrument_util

imaging = instrument_util.load_test_imaging(
    dataset_name="light_sersic__source_sersic", instrument="vro"
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

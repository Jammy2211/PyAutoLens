import autolens as al
import autolens.plot as aplt
from test_autolens.simulators.imaging import instrument_util
import numpy as np

imaging = instrument_util.load_test_imaging(
    dataset_name="light_sersic__source_sersic", instrument="vro"
)

array = imaging.image

plotter = aplt.Plotter(
    figure=aplt.Figure(figsize=(10, 10)),
    cmap=aplt.ColorMap(
        cmap="gray", norm="symmetric_log", norm_min=-0.13, norm_max=20, linthresh=0.02
    ),
    grid_scatterer=aplt.GridScatterer(marker="+", colors="cyan", size=450),
)

grid = al.GridIrregular(grid=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

print(grid)

vector_field = al.VectorFieldIrregular(
    vectors=[(1.0, 2.0), (2.0, 1.0)], grid=[(-1.0, 0.0), (-2.0, 0.0)]
)

aplt.Array(
    array=array.in_2d,
    grid=grid,
    positions=al.GridIrregularGrouped([(0.0, 1.0), (0.0, 2.0)]),
    vector_field=vector_field,
    patches=vector_field.elliptical_patches,
    plotter=plotter,
)

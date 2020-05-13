import os

import autolens as al
import autolens.plot as aplt

plot_path = "{}/../images/lensing/".format(os.path.dirname(os.path.realpath(__file__)))

grid = al.Grid.uniform(shape_2d=(50, 50), pixel_scales=0.05)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Cartesian Grid of (y,x) GridCoordinates"),
    output=aplt.Output(path=plot_path, filename="grid", format="png"),
)

aplt.Grid(grid=grid, plotter=plotter)

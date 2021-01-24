import autolens as al

grid = al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=1.0)

grid_plotter = aplt.Grid2DPlotter(grid=grid)
grid_plotter.figure()

grid = al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=1.0, origin=(5.0, 5.0))

aplt.Grid2D(grid=grid, symmetric_around_centre=False)

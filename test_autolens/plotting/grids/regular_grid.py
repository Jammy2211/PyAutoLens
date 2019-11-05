import autolens as al

grid = al.grid.uniform(shape_2d=(10, 10), pixel_scales=1.0)

al.plot.grid(grid=grid)

grid = al.grid.uniform(shape_2d=(10, 10), pixel_scales=1.0, origin=(5.0, 5.0))

al.plot.grid(grid=grid, symmetric_around_centre=False)

import autolens as al

grid = al.Grid.uniform(shape_2d=(10, 10), pixel_scales=1.0)

aplt.grid(grid=grid)

grid = al.Grid.uniform(shape_2d=(10, 10), pixel_scales=1.0, origin=(5.0, 5.0))

aplt.grid(grid=grid, symmetric_around_centre=False)

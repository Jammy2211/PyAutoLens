from autolens.data.array import grids
from autolens.data.array.plotters import grid_plotters

# In this example, we'll create a grid of Cartesian (y,x) coordinates, representing the arc second coordinate grid of
# an observed data-set (e.g. an image).

# In AutoLens, a grid is a set of two-dimensional (y,x) coordinates (in arc-seconds) that are deflected
# and traced by a strong lensing system.

# This coordinate grid is aligned with the image we analyze, thus each coordinate on the regular grid maps to the
# centre of each image-pixel. Lets make a grid using 100 x 100 pixels, with a pixel scale (arcsecond-to-pixel
# conversion factor) of 0.05", giving us a a 5" x 5" grid.
data_grids = grids.DataGrids.from_shape_and_pixel_scale(shape=(10, 100), pixel_scale=0.05,
                                                         sub_grid_size=2)

# If we print the regular grid, we see it consists of a set of arc-second coordinates
# (noting that the difference between each coordinate is the 'pixel_scales' of 0.05" defined above)
print('(y,x) regular-pixel 1:')
print(data_grids.regular[0])
print('(y,x) regular-pixel 2:')
print(data_grids.regular[1])
print('(y,x) regular-pixel 3:')
print(data_grids.regular[2])
print('(y,x) regular-pixel 101:')
print(data_grids.regular[100])
print('etc.')

# (PyAutoLens actually flattens the NumPy array to 1D to save memory, however it consists of all
# 10000 pixel coordinates on our 2D grid)
print(data_grids.regular.shape)

# We can also plot the grid, albeit it's a fairly bland uniform grid of dots.
grid_plotters.plot_grid(grid=data_grids.regular, title='Fairly Bland Uniform Grid Of Dots')

# A quick aside - we've tried our hardest to make figures like the one above appear a good size / have no overlapping
# labels. However, this depends somewhat on the display setting of your computer. If a figure doesn't look great,
# You can customize its appearance using standard Matplotlib variables. For example, we can change the grid's appearance
# above as follows:
grid_plotters.plot_grid(grid=data_grids.regular, title='Custom Grid', xlabelsize=20, ylabelsize=20, xyticksize=20,
                        pointsize=14, figsize=(8, 4))

# You can customize every plotter in AutoLens in this way!

# You might be wondering why the data_grids are using the attribute 'regular' (e.g. data_grids.regular).

# This is because the data-grids consist of multiple grids. The 'regular' grid is the grid where each (y,x) coordinate
# is located at the centre of each pixel on the grid.

# There is also a sub-grid, which splits each pixel into a (sub_grid_size x sub_grid_size) pixel. This is
# used for evaluating light and mass profiles at a higher resolution and thus more accurately.

# We specified a sub_grid_size of 2 above, therefore we expect 4 (2 x 2) times more sub-pixels than regular-pixels
# (still flattened to 1D).
print(data_grids.regular.shape)
print(data_grids.sub.shape)

# The sub-pixels coordinates can also be printed:
print('(y,x) sub-pixel 1 (of regular pixel 1):')
print(data_grids.sub[0])
print('(y,x) sub-pixel 2 (of regular pixel 1):')
print(data_grids.sub[1])
print('(y,x) sub-pixel 3 (of regular pixel 1):')
print(data_grids.sub[2])
print('(y,x) sub-pixel 4 (of regular pixel 1):')
print(data_grids.sub[3])
print('(y,x) sub-pixel 1 (of regular pixel 2):')
print(data_grids.sub[4])
print('etc.')

# Congratulations, you've completed your first PyAutoLens tutorial! Before moving on to the next one, experiment with
# PyAutoLens by doing the following:
#
# 1) Change the pixel-scale of the grid - what happens to the Cartesian coordinates?
# 2) Change the resolution of the grid - what happens Cartesian coordinates?
# 3) Change the sub-grid size - what do you note about the pixel-scale of sub pixels?
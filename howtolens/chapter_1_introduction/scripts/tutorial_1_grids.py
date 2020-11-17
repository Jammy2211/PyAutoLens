# %%
"""
Tutorial 1: Grids
=================

In this example, we'll create `Grid`'s of Cartesian $(y,x)$ coordinates, representing the arc-second coordinate `Grid`
of an observed data-set (e.g. imaging).
"""

# %%
#%matplotlib inline

from pyprojroot import here

workspace_path = str(here())
#%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt

# %%
"""
In **PyAutoLens**, a `Grid` is a set of two-dimensional $(y,x)$ coordinates (in arc-seconds) that are deflected and traced by 
a strong lensing system.

This coordinate `Grid` is aligned with the image we analyze, such that each coordinate on a  `Grid` maps to the centre 
of each image-pixel. Lets make a `Grid` using 100 x 100 pixels, with a pixel scale (arcsecond-to-pixel conversion 
factor) of 0.05", giving us a 5" x 5" grid.
"""

# %%
grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05)

# %%
"""
First, lets plot this `Grid`, which shows that it is a fairly bland uniform `Grid` of dots.
"""

# %%
plotter = aplt.Plotter(labels=aplt.Labels(title="Fairly Bland Uniform Grid Of Dots"))

aplt.Grid(grid=grid, plotter=plotter)

# %%
"""
We can print each coordinate of this `Grid`, revealing that it consists of a set of arc-second coordinates (where the 
spacing between each coordinate corresponds to the `pixel_scales` of 0.05" defined above)
"""

# %%
print("(y,x) pixel 0:")
print(grid.in_2d[0, 0])
print("(y,x) pixel 1:")
print(grid.in_2d[0, 1])
print("(y,x) pixel 2:")
print(grid.in_2d[0, 2])
print("(y,x) pixel 100:")
print(grid.in_2d[1, 0])
print("etc.")

# %%
"""
_Grid_`s in **PyAutoLens** are stored as both 1D and 2D ndarrays, because different calculations benefit from us using 
the array in different formats. We can access both the 1D and 2D `Array`'s automatically by specifying the input as a 1D 
or 2D.
"""

# %%
print("(y,x) pixel 0 (accessed in 2D):")
print(grid.in_2d[0, 0])
print("(y,x) pixel 0 (accessed in 1D):")
print(grid.in_1d[0])

# %%
"""
The shape of the `Grid` is also available in 1D and 2D, consisting of 10000 (100 x 100) coordinates.
"""

# %%
print(grid.shape_2d)
print(grid.shape_1d)

# %%
"""
We can print the entire `Grid` in either 1D or 2D. 
"""

# %%
print(grid.in_2d)
print(grid.in_1d)

# %%
"""
A `Grid` can also have a sub-grid, which splits each pixel on the `Grid` into sub-pixels of size (sub_size x sub_size). 
These additional pixels are used to perform calculations more accurately.
"""

# %%
grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

# %%
"""
We specified a sub_size of 2 above, therefore we expect 4 (2 x 2) times more sub-pixels than pixels. We can see this 
is the case by inspecting the `Grid` sub_shape attributes.
"""

# %%
print(grid.sub_shape_1d)
print(grid.sub_shape_2d)

# %%
"""
The first four pixels of our sub-grid correspond to the first four sub-pixels, which form a sub-grid inside the
first `Grid` pixel, as follows:
"""

# %%
"""
pixel 1

              ``.______
              
              I         I
              
              I         I  o = $(y,x)$ centre of
              
 y = 2.475"   I    o    I       `Grid` coordinate.
 
              I         I
              
              I_________I
              
              x = -2.475


Sub-pixels 1, 2, 3 & 4

               ``.______
               
              I         I
              
              I  o   o  I  o = $(y,x)$ centre of sub
              
 y = 2.475"   I         I       `Grid` coordinates.
 
              I  o   o  I
              
              I_________I
              
              x = -2.475

The sub-pixels coordinate are spaced uniformly between the pixel`s edges
(which are at y = (2.45", 2.5") and x = (-2.5". -2.45") )
"""

# %%
"""
By default, a `Grid` is stored in 1D, and we can access its elements without specifying `in_1d`
"""

# %%
print("(y,x) sub-pixel 0 (of pixel 0):")
print(grid[0])
print("(y,x) sub-pixel 1 (of pixel 0):")
print(grid[1])
print("(y,x) sub-pixel 2 (of pixel 0):")
print(grid[2])
print("(y,x) sub-pixel 3 (of pixel 0):")
print(grid[3])

# %%
"""
The sub-grid then continues on to the next `Grid` pixels (and so on)
"""

# %%
print("(y,x) sub-pixel 0 (of pixel 1):")
print(grid[4])
print("(y,x) sub-pixel 0 (of pixel 1):")
print(grid[5])
print("etc.")

# %%
"""
We can again access these coordinates in 2D.
"""

# %%
print("(y,x) sub-pixel 0 (of pixel 1):")
print(grid.in_2d[0, 2])
print("(y,x) sub-pixel 1 (of pixel 1):")
print(grid.in_2d[0, 3])
print("etc.")

# %%
"""
Congratulations, you`ve completed your first **PyAutoLens** tutorial! Before moving on to the next one, experiment with 
__PyAutoLens__ by doing the following:

1) Change the pixel-scale of the `Grid`'s - what happens to the Cartesian coordinates?
2) Change the resolution of the `Grid`'s - what happens Cartesian coordinates?
3) Change the sub-grid size - what do you note about the pixel-scale of sub pixels?
"""

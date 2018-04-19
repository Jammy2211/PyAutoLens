import numpy as np
import os
import sys
sys.path.append("../")

from auto_lens.profiles import light_profiles, mass_profiles
from auto_lens import galaxy
from auto_lens.imaging import grids
from auto_lens import ray_tracing


# Simple coordinates to show behaviour

coordinates = np.array([[1.0, 1.0],
                        [2.0, 2.0]])

# Setup a simple sersic light and sis mass profile

sersic = light_profiles.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=1.0,
                                         sersic_index=4.0)
sis = mass_profiles.SphericalIsothermal(einstein_radius=1.0)

# Check their values at the coordinates
print('intensity of light profile = ', sersic.intensity_at_coordinates(coordinates))
print('potential of mass profile = ', sis.potential_at_coordinates(coordinates), '\n')
print('deflection angle of mass profile = ', sis.deflection_angles_at_coordinates(coordinates), '\n')

# Associate these values with a lens galaxy
lens_galaxy = galaxy.Galaxy(light_profiles=[sersic], mass_profiles=[sis])

# Lets check its the light and mass profile values are the same as above.
print('intensity of lens galaxy = ', lens_galaxy.intensity_at_coordinates(coordinates))
print('deflecton angle of lens galaxy = ', lens_galaxy.deflection_angles_at_coordinates(coordinates),  '\n')

# And now make a source galaxy, which is just a light profile, and check it values are the same as above als.
source_galaxy = galaxy.Galaxy(light_profiles=[sersic])
print('intensity of source galaxy = ', source_galaxy.intensity_at_coordinates(coordinates),  '\n')

# Lets set up the coordinates as a grid, which is an abstract object for the coordinates we pass through the ray-tracing
# module (checkout the module for more info, basically there are lots of different grids we want to ray-trace for
# fitting data in different ways).
image_grid = grids.GridImage(grid=coordinates)
print('grid coordinates = ', image_grid.grid, '\n')

# Because can have multiple grids (which are different to the image grid we're using here), we need to combine this
# grid into a ray-tracing grids module.
ray_trace_grids = grids.RayTracingGrids(image=image_grid)
print('ray trace grid coordinates = ', ray_trace_grids.image.grid, '\n')

# Now lets pass our lens galaxy, source galaxy and grid through the ray tracing module (This currently assumes just one
# image and source plane, but will be expanded for multiple planes in the future).
ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                            image_plane_grids=ray_trace_grids)

# The ray tracing sets up an image plane, whose coordinates are our original coordinates.
print('image plane grid coordinates = ', ray_trace.image_plane.grids.image.grid,  '\n')

# The image plane is also automatically set up with deflection angles, using the lens galaxy mass profile:
print('image plane deflection angles = ', ray_trace.image_plane.deflection_angles.image.grid,  '\n')

# And a source plane is set up too, which is the image plane coordinates - the image plane deflection angles
print('source plane grid coordinates =', ray_trace.source_plane.grids.image.grid,  '\n')

# If we pass the same lens galaxy to the image plane 3 times, notice that the deflection angles triple (as we are
# basically including the same mass profile 3 times)
ray_trace_x3 = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_galaxy, lens_galaxy, lens_galaxy],
                                            source_galaxies=[source_galaxy], image_plane_grids=ray_trace_grids)
print('image plane x3 grid coordinates = ', ray_trace_x3.image_plane.grids.image.grid,  '\n')
print('image plane x3 deflection angles = ', ray_trace_x3.image_plane.deflection_angles.image.grid,  '\n')
print('source plane x3 grid coordinates =', ray_trace_x3.source_plane.grids.image.grid,  '\n')

# NOT IMPLEMENTED - BUT TO SHOW YOU THE DESIGN

# We can now go to any plane, and compute a model image of the galaxy(s) in that plane using their light profiles.

# e.g., ray_trace.image_plane.compute_model_image()  -  will use the image plane coordinates and galaxy light profiles
# to return a model image of the galaxy.

# e.g. ray_trace.source_plane.compute_model_image()  -  as above, but for the source plane coordinates and galaxy.

# e.g. ray_trace.compute_model_image()
# We could also return the model image of the whole ray tracing plane:


# We will also be able to attach a pixelization to a galaxy, which is used to reconstruct the image via a pixel grid.
# source_galaxy = galaxy.Galaxy(pixelization=AdaptivePixelization)

# Given that the galaxy's (and therefore pixelizations) are attached to a plane, for each plane we can retrieve its
# mapping matrix. The mapping matrix is what is called f in Warren & Dye 2003 and tells us the mapping between every
# image and source pixel):
# e.g. ray_trace.source_plane.compute_mapping_matrix()

# Thus, from a ray_tracing instance, we can extract all the model images / pixelization mapping matrices we could ever
# desire. These will then be used to fit in the image data in what will be the 'analysis' module.


### POTENTIAL GRID ###

# The potential grid is basically a rectangular grid of coordinates (e.g. of size (20, 20)). We could store these
# coordinates as another attribute of the RayTracingGrids class, such that when we call TraceImageAndSource, the
# image_plane has an additional set of coordinates ray_trace.image.plane.grids.potential_grid. In the image plane, the
# galaxy potential at these coordinates will be automatically computed.

# The potential grid is computed in the image-plane, so it doesn't need to know the traced image coordinates in the \
# source-plane. Thus, we'll make sure the potential grid coordinates do not have their deflection angles computed and
# that they are not traced to the source plane.

# Now our image plane knows the potential grid coordinates (and the galaxies in image plane, from which to compute their
# potentials), We could extract the potential grid much like the model images / pixelization above:
# e.g. ray_trace.compute_potential_grid()

# My guess is that this will do something along the lines of:

# 1) Using the image plane, compute the potential on the potential_grid.
# 2) Using the source plane, determine the mapping matrix or whatever quantity to needed for the potential grid.
# 3) Use the above quantities to set up the potential grid mapping matrix, which is then passed to the analyis_image
# module for fitting the data.
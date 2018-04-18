import numpy as np
import os
import sys
sys.path.append("../")

from auto_lens.profiles import light_profiles, mass_profiles
from auto_lens import galaxy
from auto_lens.imaging import grids
from auto_lens import ray_tracing


# Uncommment this to see behaviour for many coordinates

coordinates = np.array([[1.0, 1.0],
                        [2.0, 2.0]])

# Setup a simple light and mass profile

sersic = light_profiles.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=1.0,
                                         sersic_index=4.0)
sis = mass_profiles.SphericalIsothermal(einstein_radius=1.0)

# Check their values

print('intensity of light profile = ', sersic.intensity_at_coordinates(coordinates))
print('deflection angle of mass profile = ', sis.deflection_angles_at_coordinates(coordinates), '\n')

# Associate these values with a lens galaxy - the ligth and mass profile values are the same

lens_galaxy = galaxy.Galaxy(light_profiles=[sersic], mass_profiles=[sis])

print('intensity of lens galaxy = ', lens_galaxy.intensity_at_coordinates(coordinates))
print('deflecton angle of lens galaxy = ', lens_galaxy.deflection_angles_at_coordinates(coordinates),  '\n')

# And now make a source galaxy, which is just a light profile.

source_galaxy = galaxy.Galaxy(light_profiles=[sersic])

print('intensity of source galaxy = ', source_galaxy.intensity_at_coordinates(coordinates),  '\n')

# Lets set up the coordinates as a grid, which is an abstract object for the coordinates we pass through the ray-tracing
# module

image_grid = grids.GridImage(grid=coordinates)
print('grid coordinates = ', image_grid.grid, '\n')

# We can have grids other than the image plane grid, so we need to combine them before ray-tracing so that they are all
# ray traced together

ray_trace_grids = grids.RayTracingGrids(image=image_grid)
print('ray trace grid coordinates = ', ray_trace_grids.image.grid, '\n')

# Now lets pass our lens galaxy, source galaxy and grid through the ray tracing module (This currently assumes just one
# image and source plane, but will be expanded for multiple planes in the future).

ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                            image_plane_grids=ray_trace_grids)

# This sets up an image plane, whose coordinates are our original coordinates.

print('image plane grid coordinates = ', ray_trace.image_plane.grids.image.grid,  '\n')

# The image plane is also automatically set up with deflectiono angles, using the lens galaxy deflection angles:
print('image plane deflection angles = ', ray_trace.image_plane.deflection_angles.image.grid,  '\n')

# And a source plane is set up too, which is the image plane coordinates - the image plane deflectiono angles
print('source plane grid coordinates =', ray_trace.source_plane.grids.image.grid,  '\n')

# If we pass the same lens galaxy to the image plane 3 times, notice that the deflection angles triple (as we are
# basically including the same mass profile 3 times)

ray_trace_x3 = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_galaxy, lens_galaxy, lens_galaxy],
                                            source_galaxies=[source_galaxy], image_plane_grids=ray_trace_grids)

print('image plane x3 grid coordinates = ', ray_trace_x3.image_plane.grids.image.grid,  '\n')
print('image plane x3 deflection angles = ', ray_trace_x3.image_plane.deflection_angles.image.grid,  '\n')
print('source plane x3 grid coordinates =', ray_trace_x3.source_plane.grids.image.grid,  '\n')
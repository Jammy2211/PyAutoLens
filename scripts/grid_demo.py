import numpy as np
import sys

sys.path.append("../")

from auto_lens.profiles import light_profiles, mass_profiles
from auto_lens import galaxy
from auto_lens.imaging import grids
from auto_lens import ray_tracing

# Simple coordinates to show behaviour
coordinates = np.array([[1.0, 1.0]])

# Setup a simple sersic light and sis mass profile
sersic = light_profiles.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1,
                                         effective_radius=1.0, sersic_index=4.0)
sis = mass_profiles.SphericalIsothermal(einstein_radius=1.0)

# Check their values at the coordinates
print('--- VALUES OF PROFILES AT COORDINATES (1.0, 1.0) ---')
print('intensity of light profile = ', sersic.intensity_at_coordinates(coordinates[0]))
print('potential of mass profile = ', sis.potential_at_coordinates(coordinates[0]))
print('deflection angle of mass profile = ', sis.deflections_at_coordinates(coordinates[0]), '\n')

# Associate these profiles with a lens galaxy
lens_galaxy = galaxy.Galaxy(light_profiles=[sersic], mass_profiles=[sis])

# Lets check the lens galaxy light and mass profile values are the same as above.
print('--- VALUES OF LENS GALAXY AT COORDINATES (1.0, 1.0) ---')
print('intensity of lens galaxy = ', lens_galaxy.intensity_at_coordinates(coordinates[0]))
print('deflecton angle of lens galaxy = ', lens_galaxy.deflections_at_coordinates(coordinates[0]), '\n')

# Now we'll make a source galaxy, which is just a light profile, and check it values are the same as above als.
source_galaxy = galaxy.Galaxy(light_profiles=[sersic])
print('--- VALUES OF SOURCE GALXY AT COORDINATES (1.0, 1.0) ---')
print('intensity of source galaxy = ', source_galaxy.intensity_at_coordinates(coordinates[0]), '\n')

# Using just one set of coordinates and having to specify their index isn't ideal. Lets set up the coordinates as a \
# grid_coords, which is an abstract object containing a set of coordinates which we will then pass through the ray-tracing
# module.
# (checkout the grids module for more info, basically we can define whatever grids we fancy for ray-tracing.
coordinates = np.array([[1.0, 1.0],
                        [2.0, 2.0]])
image_grid = grids.GridCoordsImage(grid_coords=coordinates)
print('--- VALUES OF GRID COORDINATES [(1.0, 1.0), (2.0, 2.0)] ---')
print('grid_coords coordinates = ', image_grid, '\n')

# We can have multiple grids (which are different to the image grid_coords we're using here). Thus, before ray-tracing, we
# combine all of our grids into a grid_coords collection. This doesn't change any of aspect of this tutorial.
grid_collection = grids.GridCoordsCollection(image=image_grid)
print('--- VALUES OF GRID COLLECTION COORDINATES [(1.0, 1.0), (2.0, 2.0)] ---')
print('grid_coords collection coordinates = ', grid_collection.image, '\n')

# Now lets pass our lens galaxy, source galaxy and grid_coords collection through the ray tracing module.
# We'll assume only one image-plane and one source-plane (multiple planes will be added to the code in the future).
ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                            image_plane_grids=grid_collection)

# The ray tracing sets up an image plane, whose coordinates are our original coordinates.
print('--- VALUES OF IMAGE COORDINATES, DEFLS AND SOURCE COORDINATES ---')
print('image plane grid_coords coordinates = ', ray_trace.image_plane.grids.image)

# The image plane is also automatically set up with deflection angles, using the lens galaxy mass profile:
print('image plane deflection angles = ', ray_trace.image_plane.deflections.image)

# And a source plane is set up too, which is the image plane coordinates - the image plane deflection angles
print('source plane grid_coords coordinates =', ray_trace.source_plane.grids.image, '\n')

# If we pass the same lens galaxy to the image plane 3 times, notice that the deflection angles triple (as we are
# basically including the same mass profile 3 times)
print('--- VALUES OF IMAGE COORDINATES, DEFLS AND SOURCE COORDINATES FOR X3 LENS GALAXY ---')
ray_trace_x3 = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_galaxy, lens_galaxy, lens_galaxy],
                                               source_galaxies=[source_galaxy], image_plane_grids=grid_collection)
print('image plane x3 grid_coords coordinates = ', ray_trace_x3.image_plane.grids.image)
print('image plane x3 deflection angles = ', ray_trace_x3.image_plane.deflections.image)
print('source plane x3 grid_coords coordinates =', ray_trace_x3.source_plane.grids.image, '\n')

# We can now go to any plane, and compute a model image of the galaxy(s) in that plane using their light profiles.
galaxy_image_plane = ray_trace.image_plane.generate_image_of_galaxies()
print('--- PLANE GALAXY IMAGES ---')
print('galaxy image (intensities) from image plane = ', galaxy_image_plane)

# This is the same as just feeding the coordinates into the light profile.
profile_intensity_0 = lens_galaxy.intensity_at_coordinates(coordinates[0])
profile_intensity_1 = lens_galaxy.intensity_at_coordinates(coordinates[1])
print('intensity of lens galaxy light profile = ', [profile_intensity_0, profile_intensity_1])

# We can also do this for the source-plane and for the entire ray-tracing plane (for the image of all galaxies):
galaxy_source_plane = ray_trace.source_plane.generate_image_of_galaxies()
galaxy_ray_trace = ray_trace.generate_image_of_galaxies()
print('galaxy image (intensities) from source plane = ', galaxy_source_plane)
print('galaxy image (intensities) from all of ray-tracing (e.g. image plane + source plane) = ', galaxy_ray_trace)

# We will also be able to attach a pixelization to a galaxy, which is used to reconstruct the image via a pixel grid_coords.
# source_galaxy = galaxy.Galaxy(pixelization=AdaptivePixelization)

# Given that the galaxy's (and therefore pixelizations) are attached to a plane, for each plane we can retrieve its
# mapping matrix. The mapping matrix is what is called f in Warren & Dye 2003 and tells us the mapping between every
# image and source pixel):
# e.g. ray_trace.source_plane.compute_mapping_matrix()

# Thus, from a ray_tracing instance, we can extract all the model images / pixelization mapping matrices we could ever
# desire. These will then be used to fit in the image data in what will be the 'analysis' module.


### POTENTIAL GRID ###

# The potential grid_coords is basically a rectangular grid_coords of coordinates (e.g. of size (20, 20)). We could store these
# coordinates as another attribute of the GridCoordsCollection class, such that when we call TraceImageAndSource, the
# image_plane has an additional set of coordinates ray_trace.image.plane.grids.potential_grid. In the image plane, the
# galaxy potential at these coordinates will be automatically computed.

# The potential grid_coords is computed in the image-plane, so it doesn't need to know the traced image coordinates in the \
# source-plane. Thus, we'll make sure the potential grid_coords coordinates do not have their deflection angles computed and
# that they are not traced to the source plane.

# Now our image plane knows the potential grid_coords coordinates (and the galaxies in image plane, from which to compute their
# potentials), We could extract the potential grid_coords much like the model images / pixelization above:
# e.g. ray_trace.compute_potential_grid()

# My guess is that this will do something along the lines of:

# 1) Using the image plane, compute the potential on the potential_grid.
# 2) Using the source plane, determine the mapping matrix or whatever quantity to needed for the potential grid_coords.
# 3) Use the above quantities to set up the potential grid_coords mapping matrix, which is then passed to the analyis_image
# module for fitting the data.

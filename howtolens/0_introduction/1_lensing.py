from autolens.profiles import mass_profiles
from autolens.profiles import light_profiles
from autolens.lensing import galaxy
from autolens.lensing import ray_tracing
from autolens.imaging import mask
from autolens.plotting import ray_tracing_plotters

# In this example, we'll perform ray-tracing in AutoLens, using an image-plane + source-plane configuration.

# We'll use the 'galaxy', 'mass_profiles' and 'light_profiles' modules, to create:
# 1) A lens galaxy with an isothermal mass distribution
# 2) A source galaxy with a Sersic surface brightness.

# For the lens galaxy, we use a singular isothermal sphere (SIS) mass-profile from the 'mass-profiles' module.
sis_mass_profile = mass_profiles.SphericalIsothermal(centre=(0.1, 0.1), einstein_radius=1.6)

# For the source galaxy, we use a Sersic light-profile from the 'light_profiles' module.
sersic_light_profile = light_profiles.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                       intensity=1.0, effective_radius=1.0, sersic_index=2.5)

# We can print these profiles to confirm their parameters.
print(sis_mass_profile)
print(sersic_light_profile)

# We next use the 'galaxy' module to create our lens galaxy and source galaxy.
lens_galaxy = galaxy.Galaxy(mass=sis_mass_profile)
source_galaxy = galaxy.Galaxy(light=sersic_light_profile)

# We can print the lens and source galaxies and confirm they posses the profiles above.
print(lens_galaxy)
print(source_galaxy)

# Before performing ray-tracing, we must define our image-plane grid.
# In AutoLens, a grid is a set of two-dimensional (x,y) coordinates (in arc-seconds) that are deflected and traced by
# strong lensing system.
# Lets use a 5" x 5" grid on a 2D grid of 100 x 100 pixels.
image_plane_grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05)

# If we print the grid, we see it consists of a set of arc-second coordinates
# (PyAutoLens flattens the NumPy array to 1D to save memory, however it consists of every coordinate on our 2D grid)
print(image_plane_grids.image)

# This array infact consists of multiple grids, for example a sub-grid, which is used for over-sampling profiles to p
# provide more accurate results.
print(image_plane_grids.sub)

# We can use this grid to compute the mass profile / lens galaxy deflection angles at each coordinate of the grid.
# (Again, still in 1D NumPy arrays, for memory efficiency)
print(sis_mass_profile.deflections_from_grid(grid=image_plane_grids.image))
print(lens_galaxy.deflections_from_grid(grid=image_plane_grids.image))

# As well as the intensities of a light-profile at each grid coordinate (Yep, still in 1D).
print(sersic_light_profile.intensities_from_grid(grid=image_plane_grids.image))
print(source_galaxy.intensities_from_grid(grid=image_plane_grids.image))

# Finally, we use the lens galaxy and source galaxy to ray-trace our grids. When we pass our galaxies and grids into the
# Tracer below, the following happens:

# 1) Using the lens-galaxy's mass-profile, the deflection angle of every image-plane grid coordinate is computed.
# 2) These deflection angles are used to trace every image-plane coordinate to a source-plane coordinate.
# 3) This creates a source-plane grid of lensed coordinates.

# We use the 'ray_tracing' module to perform this operation.
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)

# The tracer is composed of two planes and printing their grids shows that the source-plane's grid has been deflected.
print(tracer.image_plane.grids.image)
print(tracer.source_plane.grids.image)

# AutoLens has tools for plotting a tracer. This plots the following:
# 1) The image-plane image, computed by tracing the source galaxy's light 'forwards' through the tracer.
# 2) The source-plane image, showing the source galaxy's true appearance (i.e. if it were not lensed).
# 3) The image-plane surface density, computed using the lens galaxy's mass profile.
# 4) The image-plane gravitational potential, computed using the lens galaxy's mass profile.
# 5) The image-plane deflection angles, computed using the lens galaxy's mass profile.
ray_tracing_plotters.plot_ray_tracing(tracer=tracer)

# These attributes can be assessed by print statements (Which you might notice have 'magically' been converted to 2D NumPy which
# are the same dimensionos as our input image!).
print('Surface Density:')
print(tracer.surface_density)
print('Potential:')
print(tracer.potential)
print('Deflections:')
print(tracer.deflections_x)
print(tracer.deflections_y)
print('Image-plane Image:')
print(tracer.image_plane_image)
print('Source-plane Image:')
print(tracer.source_plane_image)

# You can plot the above attributes on individual figures, using the 'individual' ray_tracing_plotter
ray_tracing_plotters.plot_ray_tracing_individual(tracer=tracer, plot_image_plane_image=True, plot_source_plane=True,
                                                 plot_surface_density=False, plot_potential=False,
                                                 plot_deflections=True)

# Congratulations, you've completed your first PyAutoLens tutorial! Before moving on to the next one, experiment with
# PyAutoLens by doing the following:
#
# 1) Change the lens galaxy's einstein radius - what happens to the tracer's image-plane image?
# 2) Change the source galaxy's effective radius - how does the image-plane image's appearance change?
# 3) Experiment with different light-profiles and mass-profiles. In particular, change the SphericalIsothermal
#    mass-profile to an EllipticalIsothermal mass-profile and set its axis_ratio parameter to 0.8. What
#    happens to the number of source images?
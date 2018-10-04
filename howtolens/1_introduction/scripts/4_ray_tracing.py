from autolens.profiles import mass_profiles
from autolens.profiles import light_profiles
from autolens.lensing import galaxy
from autolens.lensing import ray_tracing
from autolens.imaging import mask
from autolens.plotting import profile_plotters
from autolens.plotting import plane_plotters
from autolens.plotting import ray_tracing_plotters

# In this example, we'll use 'mass_profiles', 'light_profiles' and 'galaxy', along with the 'ray-tracing'
# module, to perform ray-tracing of a lens-plane + source-plane strong lens configuration, where:
# 1) The lens galaxy has an isothermal mass distribution
# 2) The source galaxy has a Sersic surface brightness.

# Same grid as always, you should be used to seeing this now!
image_plane_grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05,
                                                                 sub_grid_size=2)

# Lets use a simple SIS mass profile, and use it to create our lens galaxy.
sis_mass_profile = mass_profiles.SphericalIsothermal(centre=(0.1, 0.1), einstein_radius=1.6)
lens_galaxy = galaxy.Galaxy(mass=sis_mass_profile)
print(lens_galaxy)

# And lets make our source-galaxy using a Sersic light profile
sersic_light_profile = light_profiles.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                       intensity=1.0, effective_radius=1.0, sersic_index=2.5)
source_galaxy = galaxy.Galaxy(light=sersic_light_profile)
print(source_galaxy)

# Finally, we use the lens galaxy and source galaxy to ray-trace our grids. When we pass our galaxies and grids into the
# Tracer below, the following happens:

# 1) Using the lens-galaxy's mass-profile, the deflection angle of every image-plane grid coordinate is computed.
# 2) These deflection angles are used to trace every image-plane coordinate to a source-plane coordinate.
# 3) This creates a source-plane grid of lensed coordinates.

# We use the 'ray_tracing' module to perform this operation.
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)

# The tracer is composed of two planes and printing their grids shows that the source-plane's grid has been deflected.
print('Image-pixel 1 image-plane coordinate')
print(tracer.image_plane.grids.image[0])
print('Image-pixel 1 source-plane coordinate')
print(tracer.source_plane.grids.image[0])
print('Image-pixel 2 image-plane coordinate')
print(tracer.image_plane.grids.image[1])
print('Image-pixel 2 source-plane coordinate')
print(tracer.source_plane.grids.image[1])
print('Image-pixel 3 image-plane coordinate')
print(tracer.image_plane.grids.image[2])
print('Image-pixel 3 source-plane coordinate')
print(tracer.source_plane.grids.image[2])
# etc.

# We can use the plane_plotter to plot these grids.
# The image grid is simply a 100 x 100 uniform grid of coordinates.
plane_plotters.plot_plane_grid(plane=tracer.image_plane)
# Clearly, the source-plane grid has been lensed
plane_plotters.plot_plane_grid(plane=tracer.source_plane)
# We can zoom in on the 'centre' of the source-plane (remembering the lens galaxy was centred at (0.1, 0.1)
plane_plotters.plot_plane_grid(plane=tracer.source_plane, xmin=-0.1, xmax=0.3, ymin=-0.1, ymax=0.3)

# AutoLens has tools for plotting a tracer. This plots the following:
# 1) The image-plane image, computed by tracing the source galaxy's light 'forwards' through the tracer.
# 2) The source-plane image, showing the source galaxy's true appearance (i.e. if it were not lensed).
# 3) The image-plane surface density, computed using the lens galaxy's mass profile.
# 4) The image-plane gravitational potential, computed using the lens galaxy's mass profile.
# 5) The image-plane deflection angles, computed using the lens galaxy's mass profile.
ray_tracing_plotters.plot_ray_tracing(tracer=tracer)

# These attributes can be assessed by print statements (Which you might notice have been converted to 2D NumPy arrays which
# are the same dimensions as our input image!).
print('Image-Pixel 1 - Surface Density:')
print(tracer.surface_density[0,0])
print('Image-Pixel 2 - Surface Density:')
print(tracer.surface_density[0,1])
print('Image-Pixel 3 - Surface Density:')
print(tracer.surface_density[0,2])
print('Image-Pixel 101 - Surface Density:')
print(tracer.surface_density[1,0])

# I've left the rest below commented to avoid too much printing, but you can inspect their data if you please!
# print('Potential:')
# print(tracer.potential)
# print('Deflections:')
# print(tracer.deflections_x)
# print(tracer.deflections_y)
# print('Image-plane Image:')
# print(tracer.image_plane_image)
# print('Source-plane Image:')
# print(tracer.source_plane_image)

# You can plot the above attributes on individual figures, using the 'individual' ray_tracing_plotter
ray_tracing_plotters.plot_ray_tracing_individual(tracer=tracer, plot_image_plane_image=True, plot_source_plane=True,
                                                 plot_surface_density=False, plot_potential=False,
                                                 plot_deflections=True)

# And with that, we're done. You've performed your first actual lensing with PyAutoLens!

# 1) Change the lens galaxy's einstein radius - what happens to the tracer's image-plane image?
# 2) Change the source galaxy's effective radius - how does the image-plane image's appearance change?
# 3) Experiment with different light-profiles and mass-profiles. In particular, change the SphericalIsothermal
#    mass-profile to an EllipticalIsothermal mass-profile and set its axis_ratio parameter to 0.8. What
#    happens to the number of source images?

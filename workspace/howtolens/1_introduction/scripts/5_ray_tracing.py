from autolens.profiles import mass_profiles
from autolens.profiles import light_profiles
from autolens.galaxy import galaxy
from autolens.lensing import ray_tracing
from autolens.imaging import mask
from autolens.plotting import plane_plotters
from autolens.plotting import ray_tracing_plotters

# In the previous example, we used light-profiles, mass-profiles, galaxies and planes to create an image-plane +
# source-plane strong lens system. However, this took a lot of lines of code to do, so in this example we'll use the
# 'ray-tracing module to do it a lot faster!

# Lets use the same grid as always, you should be used to seeing this now!
image_plane_grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05,
                                                                 sub_grid_size=2)

# Unlike the previous tutorial, we'll offset our lens galaxy slightly from the source, to get a slightly ray-tracing
# paths.
sis_mass_profile = mass_profiles.SphericalIsothermal(centre=(0.1, 0.1), einstein_radius=1.6)
lens_galaxy = galaxy.Galaxy(mass=sis_mass_profile)
print(lens_galaxy)

# We'll also make our source galaxy ellipctical and increase its Sersic index.
sersic_light_profile = light_profiles.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                       intensity=1.0, effective_radius=1.0, sersic_index=2.5)
source_galaxy = galaxy.Galaxy(light=sersic_light_profile)
print(source_galaxy)

# Finally, we can use a 'tracer', the lens galaxy and the source galaxy to ray-trace our grids. When we pass our
# galaxies and grids into the Tracer below, the following happens:

# 1) Using the lens-galaxy's mass-profile, the deflection angle of every image-plane grid coordinate is computed.
# 2) These deflection angles are used to trace every image-plane coordinate to a source-plane coordinate.
# 3) This creates a source-plane grid of lensed coordinates.

# This is what we did using Plane's in the preious chapter, but its a lot less lines of code!
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)

# The tracer is composed of an image-plane and source-plane, just like in the previous example!
print('Image-pixel 1 image-plane coordinate')
print(tracer.image_plane.grids.image[0])
print('Image-pixel 2 image-plane coordinate')
print(tracer.image_plane.grids.image[1])
print('Image-pixel 3 image-plane coordinate')
print(tracer.image_plane.grids.image[2])

# And the source-plane's grid has been deflected.
print('Image-pixel 1 source-plane coordinate')
print(tracer.source_plane.grids.image[0])
print('Image-pixel 2 source-plane coordinate')
print(tracer.source_plane.grids.image[1])
print('Image-pixel 3 source-plane coordinate')
print(tracer.source_plane.grids.image[2])

# We can use the plane_plotter to plot these grids, like before.
plane_plotters.plot_plane_grid(plane=tracer.image_plane, title='Image-plane Grid')
plane_plotters.plot_plane_grid(plane=tracer.source_plane, title='Source-plane Grid')
plane_plotters.plot_plane_grid(plane=tracer.source_plane, axis_limits=[-0.1, 0.3, -0.1, 0.3], title='Source-plane Grid')

# AutoLens has tools for plotting a tracer. This plots the following:
# 1) The image-plane image, computed by tracing the source galaxy's light 'forwards' through the tracer.
# 2) The source-plane image, showing the source galaxy's true appearance (i.e. if it were not lensed).
# 3) The image-plane surface density, computed using the lens galaxy's mass profile.
# 4) The image-plane gravitational potential, computed using the lens galaxy's mass profile.
# 5) The image-plane deflection angles, computed using the lens galaxy's mass profile.
ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer)

# These attributes can be assessed by print statements (Which you might notice have been converted to 2D NumPy arrays
# which are the same dimensions as our input image!).
print('Image-Pixel 1 - Tracer - Surface Density:')
print(tracer.surface_density[0,0])
print('Image-Pixel 2 - Tracer - Surface Density:')
print(tracer.surface_density[0,1])
print('Image-Pixel 3 - Tracer - Surface Density:')
print(tracer.surface_density[0,2])
print('Image-Pixel 101 - Tracer - Surface Density:')
print(tracer.surface_density[1,0])

# Of course, these surface densities are identical to the image-plane surface densities, as it's only the lens galaxy
# that contributes to the overall mass of the system.
print('Image-Pixel 1 - Image-Plane - Surface Density:')
print(tracer.image_plane.surface_density[0,0])
print('Image-Pixel 2 - Image-Plane - Surface Density:')
print(tracer.image_plane.surface_density[0,1])
print('Image-Pixel 3 - Image-Plane - Surface Density:')
print(tracer.image_plane.surface_density[0,2])
print('Image-Pixel 101 - Image-Plane - Surface Density:')
print(tracer.image_plane.surface_density[1,0])

# I've left the rest below commented to avoid too much printing, but you can inspect their data if you please!
# print('Potential:')
# print(tracer.potential)
# print(tracer.image_plane.potential)
# print('Deflections:')
# print(tracer.deflections_x)
# print(tracer.deflections_y)
# print(tracer.image_plane.deflections_x)
# print(tracer.image_plane.deflections_y)
# print('Image-plane Image:')
# print(tracer.image_plane_image)
# print(tracer.image_plane.image_plane_image)
# print('Source-plane Image:')
# print(tracer.source_plane_image)
# print(tracer.image_plane.source_plane_image)

# You can also plot the above attributes on individual figures, using appropriate ray-tracing plotter (I've left most
# commented out again for convinience)
ray_tracing_plotters.plot_surface_density(tracer=tracer)
# ray_tracing_plotters.plot_potential(tracer=tracer)
# ray_tracing_plotters.plot_deflections_y(tracer=tracer)
#ray_tracing_plotters.plot_deflections_x(tracer=tracer)
# ray_tracing_plotters.plot_image_plane_image(tracer=tracer)

# Before we finish, the attentive amongst you might be wondering 'why do both the image-plane and tracer have the
# attributes surface density / potential / deflection angles, when, by definition, the two are identical'. Think about
# it - only mass profiles contribute to these quantities, and only the image-plane should have galaxies with
# mass profiles! There are two reasons:

# 1) Convinience - You could always write 'tracer.image_plane.surface_density' and
#                  'plane_plotters.surface_density(plane=tracer.image_plane). However, code appears neater if you can
#                   just write 'tracer.surface_density' and 'ray_tracing_plotters.plot_surface_density(tracer=tracer).

# 2) Multi-plane lensing - For now, we're focused on the image-plane + source-plane lensing configuration. However,
#                          there are strong lens system where there are more than 2 planes! In these instances, the
#                          surface density, potential and deflections of each plane is different to that of the tracer.
#                          This is way beyond the scope of this chapter, but be reassured that what you're learning now
#                          will prepare you for the advanced chapters later on!

# And with that, we're done. You've performed your first actual lensing with PyAutoLens!

# 1) Change the lens galaxy's einstein radius - what happens to the tracer's image-plane image?
# 2) Change the source galaxy's effective radius - how does the image-plane image's appearance change?
# 3) Experiment with different light-profiles and mass-profiles. In particular, change the SphericalIsothermal
#    mass-profile to an EllipticalIsothermal mass-profile and set its axis_ratio parameter to 0.8. What
#    happens to the number of source images?

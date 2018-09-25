from autolens.profiles import mass_profiles
from autolens.profiles import light_profiles
from autolens.lensing import galaxy
from autolens.lensing import ray_tracing
from autolens.imaging import mask
from autolens.plotting import ray_tracing_plotters

# In this example, we'll perform ray-tracing in AutoLens. We'll set up a simple lens + source configuration.
# To do this, we'll use the 'galaxy', 'mass_profiles' and 'light_profiles' modules, to create a lens galaxy with an
# isothermal mass distribution and a source-galaxy with an exponential disk surface brightness.

# First, we need to choose our lens galaxy's mass-profile from the 'mass_profiles' module. Lets keep it simple, and
# use a singular isothermal sphere (SIS).
sis_mass_profile = mass_profiles.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6)

# We also need to choose our source's light profile from the 'light_profiles' module. We'll use an elliptical
# exponential profile.
exponential_light_profile = light_profiles.EllipticalExponential(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                                 intensity=1.0, effective_radius=1.0)

# Next, we create our lens and source galaxies using these profiles. We simply pass these profiles to Galaxy objects in
# 'galaxy' module.
lens_galaxy = galaxy.Galaxy(mass=sis_mass_profile)
source_galaxy = galaxy.Galaxy(light=exponential_light_profile)

# Before we can perform ray-tracing, we define our image-plane grid. This grid is the two-dimensional set
# of (x,y) arc-second coordinates that are traced through the strong lensing configuration.
# Lets use a 5" x 5" grid on a 2D grid of 50 x 50 pixels.
image_plane_grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=(50, 50), pixel_scale=0.1)

# (If you're wondering why this uses the 'mask' module, and why it says 'Grids' plural, we'll cover this later in the
# tutorials!)

# Finally, we can use our lens and source galaxies to trace our coordinate-grid!
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)

# AutoLens has plotting tools specifically for a tracer, if we plot the tracer we'll see its surface-density,
# gravitational potential, deflection-angle map, image-plane image and source-plane image.
ray_tracing_plotters.plot_ray_tracing(tracer=tracer)

# The tracer has everything we just plotted as attributes, feel free to uncomment these lines to inspect the numerical
# values of these arrays.
# print(tracer.surface_density)
# print(tracer.potential)
# print(tracer.deflections_x)
# print(tracer.deflections_y)
# print(tracer.image_plane_image)
# print(tracer.plane_images[1])
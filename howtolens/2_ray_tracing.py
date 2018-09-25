from autolens.profiles import mass_profiles as mp
from autolens.profiles import light_profiles as lp
from autolens.lensing import galaxy as g
from autolens.lensing import ray_tracing
from autolens.imaging import mask
from autolens.plotting import ray_tracing_plotters

# In this example, we'll reinforce what we learnt about ray-tracing in the last tutorial, demonstrating the following
# 3 things:
# 1) Short-hand notation for setting up profiles and galaxies.
# 2) That galaxies can be given any number of profiles.
# 3) That a tracer can be given any number of galaxies.

# Lets begin by setting up our lens galaxy. Instead of setting up each profiles first, we'll combine it into one
# command. We'll also give it a light profile, so its light will now appear in the tracer.
lens_galaxy = g.Galaxy(light=lp.SphericalDevVaucouleurs(centre=(0.0, 0.0), intensity=1.0, effective_radius=0.5),
                       mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, einstein_radius=1.6),
                       shear=mp.ExternalShear(magnitude=0.05, phi=45.0))

# As you can see, we can put as many light and mass profiles in the galaxy and they'll be correctly handled by the
# ray-tracing once we pass it the galaxy!
# Lets also create a small satellite galaxy nearby this lens.
lens_satellite = g.Galaxy(light=lp.SphericalDevVaucouleurs(centre=(1.0, 0.0), intensity=0.5, effective_radius=0.2),
                          mass=mp.SphericalIsothermal(centre=(1.0, 0.0), einstein_radius=0.2))

# Finally, lets make two source galaxies. We don't need to use the terms 'light' and 'mass' to setup profiles, we should
# choose the most descriptive terms for the galaxy, like we have below for the source's bulge and disk.
source_galaxy_0 = g.Galaxy(bulge=lp.SphericalDevVaucouleurs(centre=(0.1, 0.2), intensity=1.0, effective_radius=0.3),
                           disk=lp.EllipticalExponential(centre=(0.1, 0.2), axis_ratio=0.8, phi=45.0, intensity=1.0,
                                                       effective_radius=1.0))

source_galaxy_1 = g.Galaxy(disk=lp.EllipticalExponential(centre=(-0.3, -0.5), axis_ratio=0.6, phi=80.0, intensity=1.0,
                                                         effective_radius=1.0))

# Again, we need to setup the grids we ray-trace using. Lets use a higher resolution grid then before!
image_plane_grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=(200, 200), pixel_scale=0.025)

# Now we just pass these galaxies to ray_tracing, to perform all lensing calculations like before. Note that we're now
# passing multiple lens galaxies and source galaxies, there is no limit to how many you pass.
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy, lens_satellite],
                                             source_galaxies=[source_galaxy_0, source_galaxy_1],
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
from autolens.profiles import mass_profiles
from autolens.profiles import light_profiles
from autolens.lensing import galaxy
from autolens.lensing import ray_tracing
from autolens.imaging import mask

# In this example, we'll perform ray-tracing in AutoLens in a simple lens + source configuration. We'll use the
# AutoLens 'galaxy', 'mass_profiles' and 'light_profiles' modules to create a lens galaxy with an isothermal
# mass distribution and a source-galaxy with an exponential disk surface brightness.


# First, we need to choose our lens galaxy's mass-profile from the 'mass_profiles' module. Lets keep it simple, and
# use a singular isothermal sphere (SIS).

sis_mass_profile = mass_profiles.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6)

# We also need to choose our source's light profile. we'll use an elliptical exponential profile from the
# 'light_profiles' module.

exponential_light_profile = light_profiles.EllipticalExponential(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                                 intensity=1.0, effective_radius=1.0)

# Next, we use these profiles to create our lens and source galaxies. We simply pass these profiles to galaxies that
# are created using the 'galaxy' module.

lens_galaxy = galaxy.Galaxy(mass=sis_mass_profile)
source_galaxy = galaxy.Galaxy(light=exponential_light_profile)

# Before we can perform ray-tracing, we need to define our image_plane_grid - which is the two-dimensional set
# of arc-second Cartesian coordinates we'll trace through our strong lensing configuration. Lets use a 5" x 5" grid.
image_plane_grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=(200, 200), pixel_scale=0.1)

# (If you're wondering why this uses the 'mask' module, and why it says 'Grids' plural, we'll cover this later in the
# tutorials!)

# Finally, we can use our lens and source galaxies to trace our coordinate-grid!
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)

# AutoLens has plotting tools specifically for a tracer, if we plot the tracer we'll see its surface-density,
# gravitational potential, deflection-angle map, image-plane image and source-plane image.


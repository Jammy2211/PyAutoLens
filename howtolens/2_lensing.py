from autolens.profiles import mass_profiles as mp
from autolens.profiles import light_profiles as lp
from autolens.lensing import galaxy as g
from autolens.lensing import ray_tracing
from autolens.imaging import mask
from autolens.plotting import ray_tracing_plotters

from astropy import cosmology

# In this example, we'll reinforce what we learnt about ray-tracing in the last tutorial and introduce the following
# new aspects about using the ray-tracing module:
# 1) Short-hand notation for setting up profiles and galaxies, to make code clean and easy to read.
# 2) How galaxies can be assigned any number of profiles.
# 3) How a tracer can be given any number of galaxies.
# 4) How by specifying redshifts and a cosmology, our results are returned in physical units (kpc).

# Lets begin by setting up our lens galaxy. Instead of setting up each profile first, we'll combine it into one
# command. To help us, we've imported the 'light_profiles' and 'mass_profiles' modules as 'lp' and 'mp' -
# this makes our code easier to read.

# We'll also give the lens galaxy a light profile, meaning its light will also appear in the image-plane image. Every
# galaxy also receives a redshift, will means that units can be converted from arcseconds to kpc.

lens_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=2.0, effective_radius=0.5,
                                                sersic_index=2.5),
                       mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, einstein_radius=1.6),
                       shear=mp.ExternalShear(magnitude=0.05, phi=45.0),
                       redshift=0.5)

# Lets also create a small satellite galaxy nearby this lens.

lens_satellite = g.Galaxy(light=lp.SphericalDevVaucouleurs(centre=(1.0, 0.0), intensity=2.0, effective_radius=0.2),
                          mass=mp.SphericalIsothermal(centre=(1.0, 0.0), einstein_radius=0.4),
                          redshift=0.5)

# Finally, lets make two source galaxies.

# We don't have to use the terms 'light' and 'mass' to setup profiles. We should choose descriptive names for each
# component of a galaxy, for example for the sources below we describe their light-components as a bulge and disk.

source_galaxy_0 = g.Galaxy(bulge=lp.SphericalDevVaucouleurs(centre=(0.1, 0.2), intensity=0.3, effective_radius=0.3),
                           disk=lp.EllipticalExponential(centre=(0.1, 0.2), axis_ratio=0.8, phi=45.0, intensity=3.0,
                                                       effective_radius=2.0),
                           redshift=1.0)

source_galaxy_1 = g.Galaxy(disk=lp.EllipticalExponential(centre=(-0.3, -0.5), axis_ratio=0.6, phi=80.0, intensity=8.0,
                                                         effective_radius=1.0),
                           redshift=1.0)

# Again, we need to setup the grids we ray-trace using. Lets use a higher resolution grid then before!

image_plane_grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=(200, 200), pixel_scale=0.025)

# Now we just pass these galaxies to ray_tracing, to perform all lensing calculations. We're now passing multiple lens
# and source galaxies, and there is no limit to how many you pass.
# We've also supplied the tracer with a cosmology.

tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy, lens_satellite],
                                             source_galaxies=[source_galaxy_0, source_galaxy_1],
                                             image_plane_grids=image_plane_grids, cosmology=cosmology.Planck15)

# Lets plot the lensing quantities again. Note that, because we supplied our galaxies with redshifts and tracer with a
# cosmology, our units have been conerted to kiloparsecs!

ray_tracing_plotters.plot_ray_tracing(tracer=tracer)

# We saw in the last example the tracer's attributes were the surface density, potential, etc.

# Now we input a cosmology to our tracer, it receives attributes assocaited with its cosmology:W

# print(tracer.image_plane.arcec_per_kpc_proper)
# print(tracer.image_plane.kpc_per_arcsec_proper)
# print(tracer.image_plane.angular_diameter_distance_to_earth)

# print(tracer.source_plane.arcec_per_kpc_proper)
# print(tracer.source_plane.kpc_per_arcsec_proper)
# print(tracer.source_plane.angular_diameter_distance_to_earth)

# print(tracer.angular_diameter_distance_from_image_to_source_plane
# print(tracer.critical_denisty_arcsec)
# print(tracer.critical_density_kpc)
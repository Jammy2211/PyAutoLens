import os

import autofit as af
import autolens as al

"""The pixel scale of dataset to be simulated."""
pixel_scales = 0.1

grid = al.Grid.uniform(shape_2d=(50, 50), pixel_scales=pixel_scales, sub_size=1)

print(grid)

# Setup the lens galaxy's light (elliptical Sersic), mass (SIE+Shear) and source galaxy light (elliptical Sersic) for
# this simulated lens.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.mp.EllipticalSersic(
        mass_to_light_ratio=10.1,
        centre=(-0.001, 0.005),
        axis_ratio=0.772905559673341,
        phi=58.07795357623584,
        intensity=2.699624610354442,
        effective_radius=0.1441552587870802,
        sersic_index=20.8030328467225003,
    ),
    disk=al.mp.EllipticalExponential(
        mass_to_light_ratio=10.1,
        centre=(0.077, 0.047),
        axis_ratio=0.3,
        phi=69.43012371637823,
        intensity=0.29617161783298507,
        effective_radius=2.3339416498752623,
    ),
    dark=al.mp.SphericalNFW(kappa_s=0.2, scale_radius=30.0),
    shear=al.mp.ExternalShear(elliptical_comps=(0.0, 0.05)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        elliptical_comps=(0.096225, -0.055555),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)


"""Use these galaxies to setup a tracer, which will generate the image for the simulated imaging dataset."""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

aplt.Tracer.subplot_tracer(tracer=tracer, grid=grid, include_critical_curves=True)

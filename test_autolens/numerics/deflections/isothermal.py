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
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=1.0
    ),
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

deflections = tracer.deflections_from_grid(grid=grid)

print(deflections)

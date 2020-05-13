import os

import autolens as al
import autolens.plot as aplt

# In this example, we simulate a complex image of a 3-plane double Einstein ring strong gravitational lens.

# The first lens galaxy is at redshift z=0.5 and has:

# - A Sersic light and mass profile representing a bulge.
# - An Exponential light and mass profile representing a disk.
# - A Spherical NFW profile representing a dark matter halo.

lens_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lmp.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.9,
        phi=45.0,
        intensity=0.5,
        effective_radius=0.3,
        sersic_index=2.5,
        mass_to_light_ratio=0.3,
    ),
    disk=al.lmp.EllipticalExponential(
        centre=(0.0, 0.0),
        axis_ratio=0.6,
        phi=45.0,
        intensity=1.0,
        effective_radius=2.0,
        mass_to_light_ratio=0.2,
    ),
    dark=al.mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=0.08, scale_radius=30.0),
)

# The second lens galaxy is at redshift 1.0 and has:

# - A SIE mass profile representing its total mass distribution.
# - An Elliptical Sersic profile representing its light.

lens_galaxy_1 = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalExponential(
        centre=(0.1, 0.1), axis_ratio=0.8, phi=60.0, intensity=3.0, effective_radius=0.1
    ),
    mass=al.mp.EllipticalIsothermal(
        centre=(0.1, 0.1), axis_ratio=0.8, phi=60.0, einstein_radius=0.4
    ),
)

# The source-galaxy is at z=1.0 and has a Sersic light-profile.

source_galaxy = al.Galaxy(
    redshift=2.0,
    light=al.lp.EllipticalSersic(
        centre=(0.2, 0.2),
        axis_ratio=0.8,
        phi=60.0,
        intensity=2.0,
        effective_radius=0.1,
        sersic_index=1.5,
    ),
)

# We create a tracer using the galaxies above, which uses the galaxy's redshifts to set image-plane and
# source-plane locations and their light and mass profiles to perform lensing calculations.
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy_0, lens_galaxy_1, source_galaxy])

# Ray-tracing calculations are performed using a (y,x) grid of Cartesian coordinates.
grid = al.Grid.uniform(shape_2d=(150, 150), pixel_scales=0.05)

# The tracer and grid are used to calculate the lens system's convergence, potential,
# deflection angles, image-plane image and source-plane image (see figure ?).
aplt.Tracer.profile_image(
    tracer=tracer, grid=grid, include=aplt.Include(critical_curves=True, caustics=True)
)

figures_path = "{}/../figures/".format(os.path.dirname(os.path.realpath(__file__)))

aplt.Tracer.profile_image(
    tracer=tracer,
    grid=grid,
    include=aplt.Include(critical_curves=True, caustics=True),
    plotter=aplt.Plotter(
        labels=aplt.Labels(title="Multi-Plane Strong Lens Image"),
        output=aplt.Output(
            path=figures_path + "/paper/",
            filename="figure_4_complex_source",
            format="png",
        ),
    ),
)

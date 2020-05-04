import autolens as al
import autolens.plot as aplt
import os

import astropy.cosmology as cosmo

plot_path = "{}/../images/lensing/".format(os.path.dirname(os.path.realpath(__file__)))

grid = al.Grid.uniform(shape_2d=(120, 120), pixel_scales=0.05)

sersic_light_profile = al.lp.EllipticalSersic(
    centre=(0.0, 0.0),
    axis_ratio=0.9,
    phi=60.0,
    intensity=0.05,
    effective_radius=2.0,
    sersic_index=4.0,
)

isothermal_mass_profile = al.mp.EllipticalIsothermal(
    centre=(0.0, 0.0), axis_ratio=0.8, phi=120.0, einstein_radius=1.6
)

another_light_profile = al.lp.EllipticalExponential(
    centre=(0.05, 0.1), axis_ratio=0.6, phi=60.0, intensity=1.0, effective_radius=0.5
)

lens_galaxy = al.Galaxy(
    redshift=0.5, light=sersic_light_profile, mass=isothermal_mass_profile
)

source_galaxy = al.Galaxy(redshift=1.0, light=another_light_profile)

tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy, source_galaxy], cosmology=cosmo.Planck15
)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Image of Strong Lensing System (Lens & Source Galaxies)"),
    output=aplt.Output(path=plot_path, filename="tracer_image", format="png"),
)

aplt.Tracer.profile_image(tracer=tracer, grid=grid, plotter=plotter)

from autolens.pipeline import phase as ph
from autolens.lensing import ray_tracing
from autolens.lensing import galaxy as g
from autolens.imaging import image as im
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
import os

# Setup the path of the analysis so we can output the simulated data at the end.
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))

lens_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0, intensity=0.1,
                                                 effective_radius=0.8, sersic_index=3.0),
                       mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=40.0, einstein_radius=1.8))

lens_satellite = g.Galaxy(light=lp.SphericalSersic(centre=(0.5, 1.0), intensity=0.1, effective_radius=0.1,
                                                   sersic_index=2.0),
                          mass=mp.SphericalIsothermal(centre=(0.5, 1.0), einstein_radius=0.2))

source_galaxy = g.Galaxy(bulge=lp.EllipticalDevVaucouleurs(centre=(0.0, 0.0), axis_ratio=0.9, phi=90.0, intensity=0.1,
                                                           effective_radius=0.3),
                         disk=lp.EllipticalExponential(centre=(0.0, 0.0), axis_ratio=0.6, phi=90.0, intensity=0.1,
                                                       effective_radius=1.0))
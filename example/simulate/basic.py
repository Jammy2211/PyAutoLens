from autolens.lensing import ray_tracing
from autolens.lensing import galaxy as g
from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
import os

# In this example, we'll simulate a basic lens + source galaxy system and output the images (as .fits) for modeling
# with AutoLens (see the phase/basic.py example).

# Setup the path of the analysis so we can output the simulated data.
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))

image_grids = mask.GridCollection.for_simulate

# Use the 'galaxy' module (imported as 'g'), 'light_profiles' module (imported as 'lp') and 'mass profiles' module
# (imported as 'mp') to setup the lens galaxy. The lens below has an elliptical Sersic light profile and singular
# isothermal ellipsoid (SIE) mass profile.
lens_galaxy = g.Galaxy(light=lp.EllipticalSersicLP(centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0, intensity=0.1,
                                                   effective_radius=0.8, sersic_index=3.0),
                       mass=mp.EllipticalIsothermalMP(centre=(0.0, 0.0), axis_ratio=0.8, phi=40.0, einstein_radius=1.8))

# Use the above modules to setup the source galaxy, which in this example has an elliptical Exponential profile.
source_galaxy = g.Galaxy(light=lp.EllipticalExponentialLP(centre=(0.0, 0.0), axis_ratio=0.9, phi=90.0, intensity=0.1,
                                                             effective_radius=0.3))

# Pass these galaxies into the 'ray_tracing' module, in this particular case a tracer which has both an image and source
# plane. Using the lens galaxy's mass profiles deflection-angle and ray-tracing calculations will be performed to
# setup the source-plane.
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy])
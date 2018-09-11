from autolens.pipeline import phase as ph
from autolens.autofit import non_linear as nl
from autolens.lensing import galaxy_prior as gp
from autolens.imaging import image as im
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
import os

# In this example, we'll generate a phase which fits a lens + source plane system. The example data we fit is
# generated using the example in 'simulate/basic.py'.

# Setup the path of the analysis so we can load the example data.
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))

# Load an image from the 'phase_basic_data' folder. It is assumed that this folder contains image.fits, noise_map.fits and
# psf.fits - we've included some example data there already.
image = im.load_from_path(path=path + '/../data/basic/', pixel_scale=0.07)

# The GalaxyPrior class represents a galaxy object, where the parameters of its associated profiles are variable and
# fitted for by the lensing.

# Here, we make a lens galaxy with both a light profile (an elliptical Sersic) and mass profile
# (a singular isothermal sphere). These profiles are loaded from the 'light_profile (lp)' and 'mass_profile (mp)'
# modules, check them out in the source code to see all the profiles you can choose from!
lens_galaxy = gp.GalaxyPrior(light=lp.EllipticalSersic, mass=mp.EllipticalIsothermal)

# We make the source galaxy just like the lens galaxy - lets use another Sersic light profile.
source_galaxy = gp.GalaxyPrior(light=lp.EllipticalSersic)

# Finally, we need to set up the 'phase' in which the lensing is performed. Depending on the lensing you can choose
# from 3 phases, which represent the number of planes in the lens system (LensPlanePhase, LensSourcePlanePhase,
# MultiPlanePhase). For this examplle, we need a LensSourcePlanePhase.
phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                optimizer_class=nl.MultiNest, phase_name='ph_basic')

# We run the phase on the image and print the results.
results = phase.run(image)

# As well as these results there will be images and plots in the 'output' folder.
print(results)

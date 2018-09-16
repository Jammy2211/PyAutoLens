from autolens.pipeline import phase as ph
from autolens.autofit import non_linear as nl
from autolens.lensing import galaxy_prior as gp
from autolens.imaging import image as im
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
import os

# In this example, we'll generate a phase which fits a lens + source plane system.
# The example data we fit is generated using the example in 'example/simulate/2_phase.py'.

# Setup the path of the analysis so we can load the example data.
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))

# Load an image from the 'data/1_basic' folder.
image = im.load_from_path(image_path=path + '/../data/1_basic/image.fits',
                          noise_path=path+'/../data/1_basic/noise_map.fits',
                          psf_path=path + '/../data/1_basic/psf.fits', pixel_scale=0.1)

# A quick visual inspection of the image shows we didn't simulate the lens galaxy's light, thus the model in this phase
# need only fit the lens's mass and source's light.
# plt.imshow(image_simulated)
# plt.show()

# To model galaxies in a lensing system, we create GalaxyPrior objects. The parameters of the profiles we pass
# a GalaxyPrior are variable and fitted for by the lensing analysis performed in a phase.

# Lets model the lens galaxy with an SIE mass profile, which we grab from the 'mass_profile (mp)' module.
lens_galaxy = gp.GalaxyPrior(mass=mp.EllipticalIsothermal)

# And model the source galaxy with an elliptical exponential light profile, from the 'light_profile (lp)' module.
source_galaxy = gp.GalaxyPrior(light=lp.EllipticalExponential)

# Finally, we need to set up the 'phase' in which the lensing is performed. For this example, we have both a lens
# plane and source plane, so we need t use a LensSourcePlanePhase.
phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                optimizer_class=nl.MultiNest, phase_name='ph_basic')

# We run the phase on the image and print the results.
results = phase.run(image)

# As well as these results there will be images and plots in the 'output' folder.
print(results)

from autolens.pipeline import phase as ph
from autolens.autofit import non_linear as nl
from autolens.lensing import galaxy_model as gp
from autolens.imaging import image as im
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.visualize import object_plotters
import shutil
import os

# In this example, we'll generate a phase which fits a lens + source plane system.
# The example data we fit is generated using the example in 'example/simulate/2_phase.py'.

# Setup the path of the analysis so we can load the example data.
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))

# Load an image from the 'data/1_basic' folder.
image = im.load_from_path(image_path=path + '/../data/1_basic/image.fits',
                          noise_path=path+'/../data/1_basic/noise_map.fits',
                          psf_path=path + '/../data/1_basic/psf.fits', pixel_scale=0.1)

# A quick visual inspection of the image will remind us that we didn't simulate the lens galaxy's light,
# thus the model in this phase needs to only fit the lens's mass and source's light.
object_plotters.plot_image_data_from_image(image=image)

# To model galaxies in a lensing system, we create 'GalaxyModel' (gp) objects. The profiles we pass a GalaxyModel are
# variable, that is they are optimized and fitted for during the analysis.

# Lets model the lens galaxy with an SIE mass profile (which is what it was simulated using).
# We'll grab from the mass_profile (mp)' module.
lens_galaxy = gp.GalaxyModel(mass=mp.EllipticalIsothermal)

# Lets model the source galaxy with an elliptical exponential light profile (again, what it was simulated using).
# We'll grab this from the 'light_profile (lp)' module.
source_galaxy = gp.GalaxyModel(light=lp.EllipticalExponential)

# Finally, we need to set up a 'phase', which is where the lens and source galaxy models above are fitted to the data.
# In this example, we have a lens plane and source plane, so we use a LensSourcePlanePhase.
# The 'optimizer' class lets us choose the non-linear optimizer we'll use to fit the model. We'll use MultiNest,
# which is nested-sampling non-linear optimizer.
phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                optimizer_class=nl.MultiNest, phase_name='ph_1_basic')

# We run the phase on the image - which fits the image with the lens galaxy and source galaxy models!.
results = phase.run(image)

# We can plot the results, e.g. the model source-galaxy image, the residuals of the fit and the chi-squareds!
object_plotters.plot_results(results=results)

# One can also print the results to see the best-fit model parameters
print(results)

# You can also checkout the 'output/ph_1_basic/ folders to see the model results, images and non-linear sampler chains
# have been output to your hard-disk.

# If a phase has already begun running at its specified path, the analysis resumes from where it was terminated.
# The folder must be manually deleted to start from scratch. So, if you run this script again, you'll notice the
# results appear immediately!
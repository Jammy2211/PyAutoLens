from autolens.pipeline import phase as ph
from autolens.lensing import galaxy_model as gm
from autolens.imaging import image as im
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.visualize import object_plotters
from autolens.visualize import array_plotters
import os

# In this example, we'll generate a phase which fits a lens + source plane system.
# The example data we fit is generated using the example in 'example/1_basic/1_simulate.py

# Setup the path of the analysis so we can load the example data.
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))

# Load an image from the 'data/1_basic' folder.
image = im.load_from_path(image_path=path + '/../data/1_basic/image.fits',
                          noise_path=path+'/../data/1_basic/noise_map.fits',
                          psf_path=path + '/../data/1_basic/psf.fits', pixel_scale=0.1)

# A quick visual inspection of the image will remind us that we didn't simulate the lens galaxy's light, thus the
# model used in this phase needs to only represent the lens's mass and source's light.
# object_plotters.plot_image_data_from_image(image=image)

# To model galaxies in a lensing system, we create 'GalaxyModel' (gm) objects, which as the name suggests is an object
# representing our model of a galaxy. The profiles we pass a GalaxyModel are variable, such that their parameters are
# varied and optimized to find the best fit to the observed image data.

# Lets model the lens galaxy with an SIE mass profile (which is what it was simulated using).
# We'll grab from the mass_profile (mp)' module.
lens_galaxy = gm.GalaxyModel(mass=mp.EllipticalIsothermal)

# Lets model the source galaxy with an elliptical exponential light profile (again, what it was simulated using).
# We'll grab this from the 'light_profile (lp)' module.
source_galaxy = gm.GalaxyModel(light=lp.EllipticalExponential)

# Finally, we need to set up a 'phase', which is where the analysis using these galaxy models is performed.
# In this example, we have a lens plane and source plane, so we use a LensSourcePlanePhase.
phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy], phase_name='ph_basic')

# When we run this phase, we pass it the image it'll fit, whih is the one we just simulated and loaded.
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
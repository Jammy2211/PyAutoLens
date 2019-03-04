import os
import numpy as np

from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy as g, galaxy_model as gm
from autolens.model.galaxy import galaxy_data as gd
from autolens.model.galaxy.util import galaxy_util
from autolens.data.array import grids, scaled_array
from autolens.pipeline import phase as ph
from autolens.model.profiles import mass_profiles as mp

# In this script, we're going to simulate a surface-density profile using a set of mass-profiles, and perform a direct
# fit to this surface density using a model galaxy. This uses a non-linear search and the fitting process is equivalent
# to fitting an image, but it bypasses the lens modeling process (e.g. the source reconstruction and usse of 'real'
# imaging data).

# Whilst this may sound like an odd thing to do, there are reasons why one may wish to perform a direct fit to a
# derived light or mass profile quantity, like the surface density:

# 1) If the mass-profile(s) used to generate the galaxy that is fitted and the model galaxy are different, this fit
#    will inform us of how the mismatch between profiles leads to a different estimate of the inferred mass profile
#    properties.

# 2) By bypassing the lens modeling process, we can test what results we get whilst bypass the potential systematics
#    that arise from a lens model fit (e.g due to the source reconstruction or quality of data).

# Get the relative path to the config files and output folder in our workspace.
path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

# First, we'll setup the grid stack we use to simulate a surface density profile.
pixel_scale = 0.05
image_shape = (250, 250)
grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=image_shape, pixel_scale=pixel_scale,
                                                                      sub_grid_size=4)

# Now lets create a galaxy, using a simple singular isothermal sphere.
galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0))

# Next, we'll generate its surface density profile. Note that, because we're using the galaxy_util surface density
# function, the sub-grid of the grid-stack that is passed to this function is used to over-sample the surface density.
# The surface density averages over this sub-grid, such that it is the shape of the image (250, 250).
surface_density = galaxy_util.surface_density_of_galaxies_from_grid(galaxies=[galaxy], grid=grid_stack.sub)
surface_density = grid_stack.regular.scaled_array_from_array_1d(array_1d=surface_density)

# Now, we'll set this surface density up as our 'galaxy-data', meaning that it is what we'll fit via a non-linear
# search phase. To perform a fit we need a noise-map to help define our chi-squared. Given we are fitting a direct
# lensing quantity the actual values of this noise-map arn't particularly important, so we'll just use a noise-map of
# all 0.1's
noise_map = scaled_array.ScaledSquarePixelArray(array=0.1*np.ones(surface_density.shape), pixel_scale=pixel_scale)
data = gd.GalaxyData(image=surface_density, noise_map=noise_map, pixel_scale=pixel_scale)

# The fit will use a mask, which we setup like any other fit. Lets use a circular mask of 2.0"
def mask_function_circular(image):
    return msk.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=2.0)

# We can now use a special phase, called a 'GalaxyFitPhase', to fit this surface density with a model galaxy the
# mass-profiles of which we get to choose. We'll fit it with a singular isothermal sphere and should see we retrieve
# the input model above.
phase = ph.GalaxyFitPhase(galaxies=dict(lens=gm.GalaxyModel(mass=mp.SphericalIsothermal)), use_surface_density=True,
                          sub_grid_size=4, mask_function=mask_function_circular,
                          optimizer_class=nl.MultiNest, phase_name='/galaxy_surface_density_fit')

phase.run(galaxy_data=[data])

# If you check your output folder, you should see that this fit has been performed and visualization specific to a
# surface-density fit is output.

# Fits to an intensity map and gravitational potential are also possible. To do so, simply change the profile quantitiy
# that is simuulated and edit the 'use_surface_density' flag in the GalaxyFitPhase above to the appropriate quantity.
# You can also fit deflection angle maps - however this requires a few small changes to this script, so we've create a
# 'galaxy_fit_deflections.py' example script to show you how.
import os
import numpy as np

from autofit import conf
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy as g, galaxy_model as gm
from autolens.model.galaxy import galaxy_data as gd
from autolens.lens import ray_tracing
from autolens.data.array import grids, scaled_array
from autolens.pipeline import phase as ph
from autolens.model.profiles import mass_profiles as mp

# Before reading this script, you should checkout the 'galaxy_fit_surface_density.py' script first, which shows you
# how to simulate a surface density profile and fit it with a galaxy. In this script, we'll do the same thing with
# deflection angles and using multiple galaxies. There a few benefits to fitting deflection angles instead of a surface
# density profile (or gravitational potential):

# 1) In terms of lensing, the deflection-angle map is closest thing to what we *actually* observe when we image and
#    model a strong lens. Thus fitting deflection angle maps is the best way we can compare the results of a lens model
#    to a theoretical quantity.

# 2) As we do in this example, we can simulate our deflecton angle map using multi-plane lens ray-tracing, and thus
#    investigate the impact assuming a single-lens plane has on the inferred lens model.

# Get the relative path to the config files and output folder in our workspace.
path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

# First, we'll setup the grid stack we use to simulate a deflection profile.
pixel_scale = 0.05
image_shape = (250, 250)
grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=image_shape, pixel_scale=pixel_scale,
                                                                      sub_grid_size=4)

# Now lets create two galaxies, using singular isothermal spheres. We'll put the two galaxies at different redshifts,
# and the second galaxy will be much lower mass as if it is a 'perturber' of the main lens galaxy.
lens_galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0), redshift=0.5)
perturber = g.Galaxy(mass=mp.SphericalIsothermal(centre=(0.5, 0.5), einstein_radius=0.1),
                     redshift=0.2)

# We only need the source galaxy to have a redshift - given we're not fitting an image it doens't need a light profile.
source_galaxy  = g.Galaxy(redshift=1.0)

# We'll use a tracer to compute our multi-plane deflection angles.
tracer = ray_tracing.TracerMultiPlanes(galaxies=[lens_galaxy, perturber, source_galaxy],
                                       image_plane_grid_stack=grid_stack)

# We'll now extract the deflection angles from the tracer - we will extract the two deflection angle maps (y and x)
# separately.
deflections_y = tracer.deflections_y
deflections_x = tracer.deflections_x

# Next, we create each deflection angle map as its own GalaxyData object. Again, this needs a somewhat arbritary
# noise-map to be used in a fit.
noise_map = scaled_array.ScaledSquarePixelArray(array=0.1*np.ones(deflections_y.shape), pixel_scale=pixel_scale)
data_y = gd.GalaxyData(image=deflections_y, noise_map=noise_map, pixel_scale=pixel_scale)
data_x = gd.GalaxyData(image=deflections_x, noise_map=noise_map, pixel_scale=pixel_scale)

# The fit will use a mask, which we setup like any other fit. Lets use a circular mask of 2.0"
def mask_function_circular(image):
    return msk.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=2.5)

# Again, we'll use a special phase, called a 'GalaxyFitPhase', to fit the deflections with our model galaxies. We'll
# fit it with two singular isothermal spheres at the same lens-plane, thus we should see how the absence of multi-plane
# ray tracing impacts the mass of the subhalo.

class DeflectionFitPhase(ph.GalaxyFitPhase):

    def pass_priors(self, previous_results):

        # You may wish to fix the first galaxy to its true centre / einstein radius

      #  self.galaxies.lens.mass.centre_0 = 0.0
      #  self.galaxies.lens.mass.centre_1 = 0.0
      #  self.galaxies.lens.mass.einstein_radius = 1.0

        # Adjusting the priors on the centre of galaxies away from (0.0", 0.0") is also a good idea.

        self.galaxies.subhalo.mass.centre_0 = prior.GaussianPrior(mean=0.5, sigma=0.3)
        self.galaxies.subhalo.mass.centre_1 = prior.GaussianPrior(mean=0.5, sigma=0.3)

phase = DeflectionFitPhase(galaxies=dict(lens=gm.GalaxyModel(mass=mp.SphericalIsothermal),
                                      subhalo=gm.GalaxyModel(mass=mp.SphericalIsothermal)), use_deflections=True,
                          sub_grid_size=4, mask_function=mask_function_circular,
                          optimizer_class=nl.MultiNest, phase_name='/galaxy_deflections_fit')


# Finally, when we run the phase, we now pass both deflection angle data's separately.
phase.run(galaxy_data=[data_y, data_x])

# If you check your output folder, you should see that this fit has been performed and visualization specific to a
# deflections fit is output.
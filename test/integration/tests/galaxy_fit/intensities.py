import os
import numpy as np

import autofit as af
from autolens.model.galaxy import galaxy as g, galaxy_model as gm
from autolens.model.galaxy import galaxy_data as gd
from autolens.model.galaxy.util import galaxy_util
from autolens.data.array import grids, scaled_array
from autolens.pipeline.phase import phase
from autolens.model.profiles import light_profiles as lp
from test.integration import integration_util

test_type = 'galaxy_fit'
test_name = "intensities"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)

def galaxy_fit_phase():

    pixel_scale = 0.1
    image_shape = (150, 150)

    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=image_shape, pixel_scale=pixel_scale,
                                                                          sub_grid_size=4)

    galaxy = g.Galaxy(redshift=0.5, mass=lp.SphericalExponential(centre=(0.0, 0.0), intensity=1.0, effective_radius=0.5))

    intensities = galaxy_util.intensities_of_galaxies_from_grid(galaxies=[galaxy], grid=grid_stack.sub)
    intensities = grid_stack.regular.scaled_array_2d_from_array_1d(array_1d=intensities)

    noise_map = scaled_array.ScaledSquarePixelArray(array=np.ones(intensities.shape), pixel_scale=pixel_scale)

    data = gd.GalaxyData(image=intensities, noise_map=noise_map, pixel_scale=pixel_scale)

    phase1 = phase.GalaxyFitPhase(
        galaxies=dict(gal=gm.GalaxyModel(redshift=0.5, light=lp.SphericalExponential)), use_intensities=True,
        sub_grid_size=4,
        optimizer_class=af.MultiNest, phase_name=test_name+'/')

    phase1.run(galaxy_data=[data])

if __name__ == "__main__":
    galaxy_fit_phase()
import os
import numpy as np

import autofit as af
from autolens.model.galaxy import galaxy as g, galaxy_model as gm
from autolens.model.galaxy import galaxy_data as gd
from autolens.model.galaxy.util import galaxy_util
from autolens.data.array import grids, scaled_array
from autolens.pipeline.phase import phase
from autolens.model.profiles import mass_profiles as mp
from test.integration import integration_util

test_type = 'galaxy_fit'
test_name = "surface_density"

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

    galaxy = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0))

    surface_density = galaxy_util.convergence_of_galaxies_from_grid(galaxies=[galaxy], grid=grid_stack.sub)
    surface_density = grid_stack.regular.scaled_array_2d_from_array_1d(array_1d=surface_density)

    noise_map = scaled_array.ScaledSquarePixelArray(array=np.ones(surface_density.shape), pixel_scale=pixel_scale)

    data = gd.GalaxyData(image=surface_density, noise_map=noise_map, pixel_scale=pixel_scale)

    phase1 = phase.GalaxyFitPhase(
        phase_name=test_name + '/',
        galaxies=dict(gal=gm.GalaxyModel(redshift=0.5, light=mp.SphericalIsothermal)), use_convergence=True,
        sub_grid_size=4,
        optimizer_class=af.MultiNest)

    phase1.run(galaxy_data=[data])

if __name__ == "__main__":
    galaxy_fit_phase()
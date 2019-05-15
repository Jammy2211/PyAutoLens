import os
import numpy as np

from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.model.galaxy import galaxy as g, galaxy_model as gm
from autolens.model.galaxy import galaxy_data as gd
from autolens.model.galaxy.util import galaxy_util
from autolens.data.array import grids, scaled_array
from autolens.pipeline import phase as ph
from autolens.model.profiles import mass_profiles as mp
from test.integration import integration_util

test_type = 'galaxy_fit'
test_name = "deflections"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)

def phase():

    pixel_scale = 0.1
    image_shape = (150, 150)

    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=image_shape, pixel_scale=pixel_scale,
                                                                          sub_grid_size=4)

    galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0))

    deflections = galaxy_util.deflections_of_galaxies_from_grid(galaxies=[galaxy], grid=grid_stack.sub)
    deflections_y = grid_stack.regular.scaled_array_2d_from_array_1d(array_1d=deflections[:, 0])
    deflections_x = grid_stack.regular.scaled_array_2d_from_array_1d(array_1d=deflections[:, 1])

    noise_map = scaled_array.ScaledSquarePixelArray(array=np.ones(deflections_y.shape), pixel_scale=pixel_scale)

    data_y = gd.GalaxyData(image=deflections_y, noise_map=noise_map, pixel_scale=pixel_scale)
    data_x = gd.GalaxyData(image=deflections_x, noise_map=noise_map, pixel_scale=pixel_scale)

    phase = ph.GalaxyFitPhase(galaxies=dict(gal=gm.GalaxyModel(light=mp.SphericalIsothermal)), use_deflections=True,
                              sub_grid_size=4,
                              optimizer_class=nl.MultiNest, phase_name=test_name+'/')

    phase.run(galaxy_data=[data_y, data_x])

if __name__ == "__main__":
    phase()
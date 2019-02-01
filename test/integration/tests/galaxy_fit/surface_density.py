import os
import numpy as np

from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.model.galaxy import galaxy as g, galaxy_model as gm
from autolens.model.galaxy import galaxy_data as gd
from autolens.data.array.util import array_util, mapping_util
from autolens.data.array import grids, scaled_array
from autolens.pipeline import phase as ph
from autolens.model.profiles import mass_profiles as mp
from test.integration import tools

test_type = 'galaxy_fit'
test_name = "surface_density"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)

def phase():

    pixel_scale = 0.1
    image_shape = (150, 150)

    grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=image_shape, pixel_scale=pixel_scale)

    galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0))

    surface_density = galaxy.surface_density_from_grid(grid=grid_stack.regular)
    surface_density = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=surface_density,
                                                                                          shape=image_shape)

  #  data = gd.GalaxyData(image=surface_density, noise_map=np.ones(surface_density.shape), pixel_scale=pixel_scale)

    phase = ph.GalaxyFitPhase(galaxy=dict(gal=gm.GalaxyModel(light=mp.SphericalIsothermal)), use_surface_density=True,
                              optimizer_class=nl.MultiNest, phase_name='phase')

    phase.run(array=surface_density)

if __name__ == "__main__":
    phase()
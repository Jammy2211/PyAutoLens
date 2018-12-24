import os
import numpy as np

from autofit import conf
from autolens.model.galaxy import galaxy as g, galaxy_model as gm
from autolens.data.array.util import array_util, mapping_util
from autolens.data.array import grids, scaled_array
from autolens.pipeline import phase as ph
from autolens.model.profiles import mass_profiles as mp
from test.integration import tools

test_type = 'galaxy_fitting'
test_name = "surface_density"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
data_path = path+'hyper/'+test_type
conf.instance = conf.Config(config_path=config_path, output_path=output_path)

def simulate_surface_density(pixel_scale, galaxy):

    image_shape = (150, 150)

    grid_stack = grids.GridStack.from_shape_and_pixel_scale(shape=image_shape, pixel_scale=pixel_scale)

    surface_density = galaxy.surface_density_from_grid(grid=grid_stack.regular)
    surface_density = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=surface_density,
                                                                                          shape=image_shape)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    array_util.numpy_array_to_fits(surface_density, path=data_path + '.fits', overwrite=True)

def setup_and_run_phase():

    pixel_scale = 0.1

    galaxy = g.Galaxy(sie=mp.EllipticalIsothermal(centre=(0.01, 0.01), axis_ratio=0.8, phi=80.0,
                                                        einstein_radius=1.6))

    simulate_surface_density(pixel_scale=pixel_scale, galaxy=galaxy)

    array_surface_density = \
        scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=data_path + data_name + '.fits',
                                                                       hdu=0, pixel_scale=pixel_scale)

    phase = ph.GalaxyFitSurfaceDensityPhase(dict(galaxy=gm.GalaxyModel(light=mp.EllipticalIsothermal)),
                              phase_name='surface_density')

    result = phase.run(array=array_surface_density, noise_map=np.ones(array_surface_density.shape))
    print(result)


if __name__ == "__main__":
    setup_and_run_phase()

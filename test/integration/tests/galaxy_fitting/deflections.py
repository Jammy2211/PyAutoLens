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
test_name = "deflections"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
data_path = path+'hyper/'+test_type
conf.instance = conf.Config(config_path=config_path, output_path=output_path)

def simulate_deflections(pixel_scale, galaxy):

    image_shape = (150, 150)

    grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=image_shape, pixel_scale=pixel_scale)

    deflections = galaxy.deflections_from_grid(grid=grid_stack.regular)
    deflections_y = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=deflections[:, 0],
                                                                                        shape=image_shape)
    deflections_x = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=deflections[:, 1],
                                                                                        shape=image_shape)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    array_util.numpy_array_to_fits(deflections_y, path=data_path + 'y.fits', overwrite=True)
    array_util.numpy_array_to_fits(deflections_x, path=data_path + 'x.fits', overwrite=True)

def setup_and_run_phase():

    pixel_scale = 0.1

    galaxy = g.Galaxy(sie=mp.EllipticalIsothermal(centre=(0.01, 0.01), axis_ratio=0.8, phi=80.0,
                                                        einstein_radius=1.6))

    simulate_deflections(pixel_scale=pixel_scale, galaxy=galaxy)

    array_y = \
        scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=data_path + 'y.fits',
                                                                       hdu=0, pixel_scale=pixel_scale)

    array_x = \
        scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=data_path + 'x.fits',
                                                                       hdu=0, pixel_scale=pixel_scale)

    phase = ph.GalaxyFitDeflectionsPhase(dict(galaxy=gm.GalaxyModel(light=mp.EllipticalIsothermal)),
                                               phase_name='deflection_stacks')

    result = phase.run(array_y=array_y, array_x=array_x, noise_map=np.ones(array_y.shape))
    print(result)


if __name__ == "__main__":
    setup_and_run_phase()

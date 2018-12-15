import os
import numpy as np

from autolens.model.galaxy import galaxy as g, galaxy_model as gm
from autolens.data.array.util import grid_util
from autolens.data.array import mask, scaled_array
from autolens.pipeline import phase as ph
from autolens.model.profiles import mass_profiles as mp
from test.integration import tools

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
data_path = '{}/../datas/galaxy_fit'.format(dirpath)
output_path = '{}/../output/galaxy_fit'.format(dirpath)

def simulate_deflections(data_name_y, data_name_x, pixel_scale, galaxy):

    image_shape = (150, 150)

    grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=image_shape, pixel_scale=pixel_scale)

    deflections = galaxy.deflections_from_grid(grid=grids.image)
    deflections_y = grid_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=deflections[:, 0],
                                                                                        shape=image_shape)
    deflections_x = grid_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=deflections[:, 1],
                                                                                        shape=image_shape)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    grid_util.numpy_array_to_fits(deflections_y, path=data_path + data_name_y + '.fits', overwrite=True)
    grid_util.numpy_array_to_fits(deflections_x, path=data_path + data_name_x + '.fits', overwrite=True)

def setup_and_run_phase():


    data_name_y = '/deflections_y'
    data_name_x = '/deflections_x'

    pixel_scale = 0.1

    tools.reset_paths(data_name='/deflection_stacks', pipeline_name='', output_path=output_path)

    galaxy = g.Galaxy(sie=mp.EllipticalIsothermal(centre=(0.01, 0.01), axis_ratio=0.8, phi=80.0,
                                                        einstein_radius=1.6))

    simulate_deflections(data_name_y=data_name_y, data_name_x=data_name_x, pixel_scale=pixel_scale, galaxy=galaxy)

    array_y = \
        scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=data_path + data_name_y + '.fits',
                                                                       hdu=0, pixel_scale=pixel_scale)

    array_x = \
        scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=data_path + data_name_x + '.fits',
                                                                       hdu=0, pixel_scale=pixel_scale)

    phase = ph.GalaxyFitDeflectionsPhase(dict(galaxy=gm.GalaxyModel(light=mp.EllipticalIsothermal)),
                                               phase_name='deflection_stacks')

    result = phase.run(array_y=array_y, array_x=array_x, noise_map=np.ones(array_y.shape))
    print(result)


if __name__ == "__main__":
    setup_and_run_phase()

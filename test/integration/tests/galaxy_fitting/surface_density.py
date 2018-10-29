import os
import numpy as np

from autolens.galaxy import galaxy as g
from autolens.galaxy import galaxy_model as gm
from autolens.imaging import imaging_util
from autolens.imaging import scaled_array
from autolens.imaging import mask
from autolens.pipeline import phase as ph
from autolens.profiles import mass_profiles as mp
from test.integration import tools

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
data_path = '{}/../data/galaxy_fit'.format(dirpath)
output_path = '{}/../output/galaxy_fit'.format(dirpath)

def simulate_surface_density(data_name, pixel_scale, galaxy):

    image_shape = (150, 150)

    grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=image_shape, pixel_scale=pixel_scale)

    surface_density = galaxy.surface_density_from_grid(grid=grids.image)
    surface_density = imaging_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=surface_density,
                                                                                             shape=image_shape)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    imaging_util.numpy_array_to_fits(surface_density, path=data_path + data_name +'.fits', overwrite=True)

def setup_and_run_phase():

    data_name = '/surface_density'

    pixel_scale = 0.1

    tools.reset_paths(data_name=data_name, pipeline_name='', output_path=output_path)

    galaxy = g.Galaxy(sie=mp.EllipticalIsothermal(centre=(0.01, 0.01), axis_ratio=0.8, phi=80.0,
                                                        einstein_radius=1.6))

    simulate_surface_density(data_name=data_name, pixel_scale=pixel_scale, galaxy=galaxy)

    array_surface_density = \
        scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=data_path + data_name + '.fits',
                                                                       hdu=0, pixel_scale=pixel_scale)

    phase = ph.GalaxyFitSurfaceDensityPhase(dict(galaxy=gm.GalaxyModel(light=mp.EllipticalIsothermal)),
                              phase_name='surface_density')

    result = phase.run(array=array_surface_density, noise_map=np.ones(array_surface_density.shape))
    print(result)


if __name__ == "__main__":
    setup_and_run_phase()

import os
import numpy as np

from autolens.galaxy import galaxy as g
from autolens.galaxy import galaxy_model as gm
from autolens.imaging.util import grid_util
from autolens.imaging import scaled_array
from autolens.imaging import mask
from autolens.pipeline import phase as ph
from autolens.profiles import light_profiles as lp
from test.integration import tools

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
data_path = '{}/../data/galaxy_fit'.format(dirpath)
output_path = '{}/../output/galaxy_fit'.format(dirpath)

def simulate_intensities(data_name, pixel_scale, galaxy):

    image_shape = (150, 150)

    grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=image_shape, pixel_scale=pixel_scale)

    intensities = galaxy.intensities_from_grid(grid=grids.image)
    intensities = grid_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=intensities,
                                                                                      shape=image_shape)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    grid_util.numpy_array_to_fits(intensities, path=data_path + data_name + '.fits', overwrite=True)

def setup_and_run_phase():

    data_name = '/intensities'

    pixel_scale = 0.1

    tools.reset_paths(data_name=data_name, pipeline_name='', output_path=output_path)

    galaxy = g.Galaxy(sersic=lp.SphericalSersic(centre=(0.01, 0.01), intensity=0.1, effective_radius=0.5,
                                                sersic_index=2.0))

    simulate_intensities(data_name=data_name, pixel_scale=pixel_scale, galaxy=galaxy)

    array_intensities = \
        scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=data_path + data_name + '.fits',
                                                                       hdu=0, pixel_scale=pixel_scale)

    phase = ph.GalaxyFitIntensitiesPhase(dict(galaxy=gm.GalaxyModel(light=lp.SphericalSersic)),
                              phase_name='intensities')

    result = phase.run(array=array_intensities, noise_map=np.ones(array_intensities.shape))
    print(result)


if __name__ == "__main__":
    setup_and_run_phase()

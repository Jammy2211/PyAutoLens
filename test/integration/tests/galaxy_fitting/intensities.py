import os
import numpy as np

from autofit import conf
from autolens.model.galaxy import galaxy as g, galaxy_model as gm
from autolens.data.array.util import array_util, mapping_util
from autolens.data.array import grids, scaled_array
from autolens.pipeline import phase as ph
from autolens.model.profiles import light_profiles as lp
from test.integration import tools

test_type = 'galaxy_fitting'
test_name = "intensities"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
data_path = path+'hyper/'+test_type
conf.instance = conf.Config(config_path=config_path, output_path=output_path)

def simulate_intensities(pixel_scale, galaxy):

    image_shape = (150, 150)

    grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=image_shape, pixel_scale=pixel_scale)

    intensities = galaxy.intensities_from_grid(grid=grid_stack.regular)
    intensities = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=intensities,
                                                                                         shape=image_shape)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    array_util.numpy_array_to_fits(intensities, file_path=data_path + '.fits', overwrite=True)

def setup_and_run_phase():

    pixel_scale = 0.1

    galaxy = g.Galaxy(sersic=lp.SphericalSersic(centre=(0.01, 0.01), intensity=0.1, effective_radius=0.5,
                                                sersic_index=2.0))

    tools.reset_paths(test_name=test_name, output_path=output_path)

    simulate_intensities(pixel_scale=pixel_scale, galaxy=galaxy)

    array_intensities = \
        scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=data_path + '.fits',
                                                                       hdu=0, pixel_scale=pixel_scale)

    phase = ph.GalaxyFitPhase(galaxy=dict(gal=gm.GalaxyModel(light=lp.SphericalSersic)), use_intensities=True,
                              phase_name="{}/phase1".format(test_name))

    phase.run(array=array_intensities, noise_map=np.ones(array_intensities.shape))


if __name__ == "__main__":
    setup_and_run_phase()

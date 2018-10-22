import numpy as np
import pytest

from autolens.fitting import fitting
from autolens.imaging import scaled_array as sca
from autolens.imaging import mask as msk
from autolens.galaxy import galaxy_data
from autolens.galaxy import galaxy as g
from autolens.galaxy import galaxy_fitting
from autolens.profiles import light_profiles as lp
from autolens.profiles import light_profiles as mp
from test.mock.mock_galaxy import MockGalaxy

@pytest.fixture(name="galaxy_data", scope='function')
def make_galaxy_data():
    array = sca.ScaledSquarePixelArray(array=np.ones(1), pixel_scale=1.0)
    mask = msk.Mask(array=np.array([[True, True, True],
                                    [True, False, True],
                                    [True, True, True]]), pixel_scale=1.0)
    return galaxy_data.GalaxyData(data=array, mask=mask)


class TestGalaxyFit:

    class TestLikelihood:

        def test__1x1_image__light_profile_fits_data_perfectly__lh_is_noise_term(self):

            array = sca.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)

            noise_map = sca.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)

            mask = msk.Mask(array=np.array([[True, True, True],
                                           [True, False, True],
                                           [True, True, True]]), pixel_scale=1.0)

            data = galaxy_data.GalaxyData(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)

            g0 = MockGalaxy(value=1.0)

            fit = galaxy_fitting.GalaxyFit(galaxy_data=data, galaxy=g0)

            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        def test__1x1_image__tracing_fits_data_with_chi_sq_5(self):

            array = sca.ScaledSquarePixelArray(array=5.0*np.ones((3, 4)), pixel_scale=1.0)
            array[1,2] = 4.0

            noise_map = sca.ScaledSquarePixelArray(array=np.ones((3, 4)), pixel_scale=1.0)

            mask = msk.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)

            data = galaxy_data.GalaxyData(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)

            # Setup as a ray trace instance, using a light profile for the lens

            data = galaxy_data.GalaxyData(array=array, noise_map=noise_map, mask=mask, sub_grid_size=1)

            g0 = MockGalaxy(value=1.0)

            fit = galaxy_fitting.GalaxyFit(galaxy_data=data, galaxy=g0)

            assert fit.chi_squared_term == 25.0
            assert fit.reduced_chi_squared == 25.0 / 2.0
            assert fit.likelihood == -0.5 * (25.0 + 2.0*np.log(2 * np.pi * 1.0))

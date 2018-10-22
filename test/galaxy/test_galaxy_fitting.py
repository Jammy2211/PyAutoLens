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
from test.mock.mock_profiles import MockLightProfile
from test.mock.mock_profiles import MockMassProfile
from test.mock.mock_galaxy import MockHyperGalaxy


@pytest.fixture(name="mock_galaxy", scope='function')
def make_mock_galaxy():
    return [g.Galaxy(light=MockLightProfile(value=1.0), mass=MockMassProfile(value=1.0))]

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

            data = galaxy_data.GalaxyData(data=array, noise_map=noise_map, mask=mask, sub_grid_size=1)

            g0 = g.Galaxy(mass=MockMassProfile(value=1.0))

            fit = galaxy_fitting.GalaxyFit(galaxy_data=data, galaxy=g0)

            print(fit.model_data)
            print(fit.data)
            print(fit.chi_squared_term)
            print(fit.noise_term)

            assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        def test__1x1_image__tracing_fits_data_with_chi_sq_5(self):
            psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])), pixel_scale=1.0)

            im = image.Image(5.0 * np.ones((3, 4)), pixel_scale=1.0, psf=psf, noise_map=np.ones((3, 4)))
            im[1, 2] = 4.0

            ma = mask.Mask(array=np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, True, True, True]]), pixel_scale=1.0)

            li = lensing_image.LensingImage(im, ma, sub_grid_size=1)

            # Setup as a ray trace instance, using a light profile for the lens

            g0 = g.Galaxy(light_profile=MockLightProfile(value=1.0))
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0], image_plane_grids=li.grids)

            fit = lensing_fitting.LensingProfileFit(lensing_image=li, tracer=tracer)

            assert fit.chi_squared_term == 25.0
            assert fit.reduced_chi_squared == 25.0 / 2.0
            assert fit.likelihood == -0.5 * (25.0 + 2.0 * np.log(2 * np.pi * 1.0))
import pytest
import numpy as np

###############
#### DATA #####
###############

#### CCD #####

from test.unit.mock.data import mock_ccd

@pytest.fixture()
def image_5x5():
    return mock_ccd.MockImage(shape=(5, 5), value=1.0)

@pytest.fixture()
def psf_3x3():
    return mock_ccd.MockPSF(shape=(3, 3), value=1.0)

@pytest.fixture()
def noise_map_5x5():
    return mock_ccd.MockNoiseMap(shape=(5, 5), value=2.0)

@pytest.fixture()
def background_noise_map_5x5():
    return mock_ccd.MockBackgroundNoiseMap(shape=(5, 5), value=3.0)

@pytest.fixture()
def poisson_noise_map_5x5():
    return mock_ccd.MockPoissonNoiseMap(shape=(5, 5), value=4.0)

@pytest.fixture()
def exposure_time_map_5x5():
    return mock_ccd.MockExposureTimeMap(shape=(5, 5), value=5.0)

@pytest.fixture()
def background_sky_map_5x5():
    return mock_ccd.MockBackgrondSkyMap(shape=(5, 5), value=6.0)

@pytest.fixture()
def positions_5x5():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))

@pytest.fixture()
def ccd_data_5x5(image_5x5, psf_3x3, noise_map_5x5, background_noise_map_5x5, poisson_noise_map_5x5,
                 exposure_time_map_5x5, background_sky_map_5x5):

    return mock_ccd.MockCCDData(
        image=image_5x5, pixel_scale=image_5x5.pixel_scale, psf=psf_3x3, noise_map=noise_map_5x5,
        background_noise_map=background_noise_map_5x5, poisson_noise_map=poisson_noise_map_5x5,
        exposure_time_map=exposure_time_map_5x5, background_sky_map=background_sky_map_5x5, name='mock_ccd_data_5x5')


@pytest.fixture()
def ccd_data_6x6():
    image = mock_ccd.MockImage(shape=(6, 6), value=1.0)
    psf = mock_ccd.MockPSF(shape=(3, 3), value=1.0)
    noise_map = mock_ccd.MockNoiseMap(shape=(6, 6), value=2.0)
    background_noise_map = mock_ccd.MockBackgroundNoiseMap(shape=(6, 6), value=3.0)
    poisson_noise_map = mock_ccd.MockPoissonNoiseMap(shape=(6, 6), value=4.0)
    exposure_time_map = mock_ccd.MockExposureTimeMap(shape=(6, 6), value=5.0)
    background_sky_map = mock_ccd.MockBackgrondSkyMap(shape=(6, 6), value=6.0)

    return mock_ccd.MockCCDData(image=image, pixel_scale=1.0, psf=psf, noise_map=noise_map,
                                background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                                exposure_time_map=exposure_time_map, background_sky_map=background_sky_map,
                                name='mock_ccd_data_6x6')

### MASK ####

from test.unit.mock.data import mock_mask

@pytest.fixture()
def mask_5x5():

    array = np.array([[True, True,  True,  True,  True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True,  True,  True,  True, True]])

    return mock_mask.MockMask(array=array)


@pytest.fixture()
def blurring_mask_5x5():

    array = np.array([[False, False, False, False, False],
                      [False, True, True, True, False],
                      [False, True, True, True, False],
                      [False, True, True, True, False],
                      [False, False, False, False, False]])

    return mock_mask.MockMask(array=array)


@pytest.fixture()
def padded_mask_7x7():

    array = np.full(fill_value=False, shape=(7, 7))

    return mock_mask.MockMask(array=array)


@pytest.fixture()
def mask_6x6():

    array = np.array([[True, True, True, True, True, True],
                      [True, True, True, True, True, True],
                      [True, True, False, False, True, True],
                      [True, True, False, False, True, True],
                      [True, True, True, True, True, True],
                      [True, True, True, True, True, True]])

    return mock_mask.MockMask(array=array)

### MASKED DATA ###

@pytest.fixture()
def image_1d_5x5(image_5x5, mask_5x5):
    return mask_5x5.map_2d_array_to_masked_1d_array(array_2d=image_5x5)

@pytest.fixture()
def noise_map_1d_5x5(noise_map_5x5, mask_5x5):
    return mask_5x5.map_2d_array_to_masked_1d_array(array_2d=noise_map_5x5)

#### GRIDS ####

from test.unit.mock.data import mock_grids

@pytest.fixture()
def regular_grid_5x5(mask_5x5):
    return mock_grids.MockRegularGrid(mask=mask_5x5)

@pytest.fixture()
def sub_grid_5x5(mask_5x5):
    return mock_grids.MockSubGrid(mask=mask_5x5)

@pytest.fixture()
def blurring_grid_5x5():

    blurring_mask = np.array([[False, False, False, False, False],
                              [False, True, True, True, False],
                              [False, True, True, True, False],
                              [False, True, True, True, False],
                              [False, False, False, False, False]])

    return mock_grids.MockRegularGrid(mask=blurring_mask)

@pytest.fixture()
def grid_stack_5x5(regular_grid_5x5, sub_grid_5x5, blurring_grid_5x5):
    return mock_grids.MockGridStack(regular=regular_grid_5x5, sub=sub_grid_5x5, blurring=blurring_grid_5x5)

@pytest.fixture()
def grid_stack_simple(regular_grid_5x5, sub_grid_5x5, blurring_grid_5x5):

    # Manually overwrite some sub-grid and blurring grid coodinates for easier deflection angle calculations

    grid_stack = mock_grids.MockGridStack(regular=regular_grid_5x5, sub=sub_grid_5x5, blurring=blurring_grid_5x5)

    grid_stack.regular[0] = np.array([1.0, 1.0])
    grid_stack.sub[0] = np.array([1.0, 1.0])
    grid_stack.sub[1] = np.array([1.0, 0.0])
    grid_stack.sub[2] = np.array([1.0, 1.0])
    grid_stack.sub[3] = np.array([1.0, 0.0])
    grid_stack.blurring[0] = np.array([1.0, 0.0])

    return grid_stack

@pytest.fixture()
def padded_regular_grid_5x5():
    return mock_grids.MockPaddedRegularGrid(image_shape=(5,5), psf_shape=(3,3))

@pytest.fixture()
def padded_sub_grid_5x5():
    return mock_grids.MockPaddedSubGrid(image_shape=(5,5), psf_shape=(3,3))

@pytest.fixture()
def padded_grid_stack_5x5(padded_regular_grid_5x5, padded_sub_grid_5x5):
    return mock_grids.MockPaddedGridStack(regular=padded_regular_grid_5x5, sub=padded_sub_grid_5x5)

### BORDERS ###

@pytest.fixture()
def border_5x5():
    return mock_grids.MockBorders(arr=np.array([0, 1, 2, 3, 5, 6, 7, 8]))

### CONVOLVERS ###

from test.unit.mock.data import mock_convolution

@pytest.fixture()
def convolver_image_5x5(mask_5x5, blurring_mask_5x5, psf_3x3):
    return mock_convolution.MockConvolverImage(mask=mask_5x5, blurring_mask=blurring_mask_5x5, psf=psf_3x3)

@pytest.fixture()
def convolver_mapping_matrix_5x5(mask_5x5, psf_3x3):
    return mock_convolution.MockConvolverMappingMatrix(mask=mask_5x5, psf=psf_3x3)

###############
#### MODEL ####
###############

## PROFILES ###

from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.profiles import light_and_mass_profiles as lmp

@pytest.fixture
def lp_0():
    return lp.SphericalSersic(intensity=1.0, effective_radius=2.0, sersic_index=2.0)

@pytest.fixture
def lp_1():
    return lp.SphericalSersic(intensity=2.0, effective_radius=2.0, sersic_index=2.0)

@pytest.fixture
def mp_0():
    return mp.SphericalIsothermal(einstein_radius=1.0)

@pytest.fixture
def mp_1():
    return mp.SphericalIsothermal(einstein_radius=2.0)

@pytest.fixture
def lmp_0():
    return lmp.EllipticalSersicRadialGradient()


### GALAXY ###

from autolens.model.galaxy import galaxy as g

@pytest.fixture
def gal_x1_lp(lp_0):
    return g.Galaxy(redshift=0.5, lp0=lp_0)


@pytest.fixture
def gal_x2_lp(lp_0, lp_1):
    return g.Galaxy(redshift=0.5, lp0=lp_0, lp1=lp_1)


@pytest.fixture
def gal_x1_mp(mp_0):
    return g.Galaxy(redshift=0.5, mass_profile_0=mp_0)


@pytest.fixture
def gal_x2_mp(mp_0, mp_1):
    return g.Galaxy(redshift=0.5, mass_profile_0=mp_0, mass_profile_1=mp_1)


## GALAXY DATA ##

from autolens.model.galaxy import galaxy_data as gd

@pytest.fixture()
def gal_data_5x5(image_5x5, noise_map_5x5, mask_5x5):
    return gd.GalaxyData(image=image_5x5, noise_map=noise_map_5x5, pixel_scale=image_5x5.pixel_scale)

@pytest.fixture()
def gal_fit_data_5x5_intensities(gal_data_5x5, mask_5x5):
    return gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2, use_intensities=True)

@pytest.fixture()
def gal_fit_data_5x5_convergence(gal_data_5x5, mask_5x5):
    return gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2, use_convergence=True)

@pytest.fixture()
def gal_fit_data_5x5_potential(gal_data_5x5, mask_5x5):
    return gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2, use_potential=True)

@pytest.fixture()
def gal_fit_data_5x5_deflections_y(gal_data_5x5, mask_5x5):
    return gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2, use_deflections_y=True)

@pytest.fixture()
def gal_fit_data_5x5_deflections_x(gal_data_5x5, mask_5x5):
    return gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2, use_deflections_x=True)


## GALAXY FIT ##

from autolens.model.galaxy import galaxy_fit

@pytest.fixture()
def gal_fit_5x5_intensities(gal_fit_data_5x5_intensities, gal_x1_lp):
    return galaxy_fit.GalaxyFit(galaxy_data=gal_fit_data_5x5_intensities, model_galaxies=[gal_x1_lp])

@pytest.fixture()
def gal_fit_5x5_convergence(gal_fit_data_5x5_convergence, gal_x1_mp):
    return galaxy_fit.GalaxyFit(galaxy_data=gal_fit_data_5x5_convergence, model_galaxies=[gal_x1_mp])

@pytest.fixture()
def gal_fit_5x5_potential(gal_fit_data_5x5_potential, gal_x1_mp):
    return galaxy_fit.GalaxyFit(galaxy_data=gal_fit_data_5x5_potential, model_galaxies=[gal_x1_mp])

@pytest.fixture()
def gal_fit_5x5_deflections_y(gal_fit_data_5x5_deflections_y, gal_x1_mp):
    return galaxy_fit.GalaxyFit(galaxy_data=gal_fit_data_5x5_deflections_y, model_galaxies=[gal_x1_mp])

@pytest.fixture()
def gal_fit_5x5_deflections_x(gal_fit_data_5x5_deflections_x, gal_x1_mp):
    return galaxy_fit.GalaxyFit(galaxy_data=gal_fit_data_5x5_deflections_x, model_galaxies=[gal_x1_mp])


### LensData

from test.unit.mock.lens import mock_lens_data

@pytest.fixture()
def lens_data_5x5(ccd_data_5x5, mask_5x5, blurring_mask_5x5, grid_stack_5x5, padded_grid_stack_5x5, border_5x5,
                  convolver_image_5x5, convolver_mapping_matrix_5x5):
    return mock_lens_data.MockLensData(
        ccd_data=ccd_data_5x5, mask=mask_5x5, grid_stack=grid_stack_5x5, padded_grid_stack=padded_grid_stack_5x5,
        border=border_5x5, convolver_image=convolver_image_5x5, convolver_mapping_matrix=convolver_mapping_matrix_5x5)
from os import path

import numpy as np
import pytest

import autofit as af
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy import galaxy_data as gd
from autolens.model.galaxy import galaxy_fit
from autolens.model.profiles import light_and_mass_profiles as lmp
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from test.unit.mock.data import mock_ccd
from test.unit.mock.data import mock_convolution
from test.unit.mock.data import mock_grids
from test.unit.mock.data import mock_mask
from test.unit.mock.lens import mock_lens_data

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    af.conf.instance = af.conf.Config(path.join(directory, "test_files/config"),
                                      path.join(directory, "output"))


#
# DATA #
#
# CCD #

@pytest.fixture(name="image_5x5")
def make_image_5x5():
    return mock_ccd.MockImage(
        shape=(5, 5), value=1.0)


@pytest.fixture(name="psf_3x3")
def make_psf_3x3():
    return mock_ccd.MockPSF(
        shape=(3, 3), value=1.0)


@pytest.fixture(name="noise_map_5x5")
def make_noise_map_5x5():
    return mock_ccd.MockNoiseMap(
        shape=(5, 5), value=2.0)


@pytest.fixture(name="background_noise_map_5x5")
def make_background_noise_map_5x5():
    return mock_ccd.MockBackgroundNoiseMap(
        shape=(5, 5), value=3.0)


@pytest.fixture(name="poisson_noise_map_5x5")
def make_poisson_noise_map_5x5():
    return mock_ccd.MockPoissonNoiseMap(
        shape=(5, 5), value=4.0)


@pytest.fixture(name="exposure_time_map_5x5")
def make_exposure_time_map_5x5():
    return mock_ccd.MockExposureTimeMap(
        shape=(5, 5), value=5.0)


@pytest.fixture(name="background_sky_map_5x5")
def make_background_sky_map_5x5():
    return mock_ccd.MockBackgrondSkyMap(
        shape=(5, 5), value=6.0)


@pytest.fixture(name="positions_5x5")
def make_positions_5x5():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))


@pytest.fixture(name="ccd_data_5x5")
def make_ccd_data_5x5(image_5x5, psf_3x3, noise_map_5x5, background_noise_map_5x5,
                      poisson_noise_map_5x5,
                      exposure_time_map_5x5, background_sky_map_5x5):
    return mock_ccd.MockCCDData(
        image=image_5x5,
        pixel_scale=image_5x5.pixel_scale,
        psf=psf_3x3,
        noise_map=noise_map_5x5,
        background_noise_map=background_noise_map_5x5,
        poisson_noise_map=poisson_noise_map_5x5,
        exposure_time_map=exposure_time_map_5x5,
        background_sky_map=background_sky_map_5x5,
        name='mock_ccd_data_5x5')


@pytest.fixture(name="ccd_data_6x6")
def make_ccd_data_6x6():
    image = mock_ccd.MockImage(shape=(6, 6), value=1.0)
    psf = mock_ccd.MockPSF(shape=(3, 3), value=1.0)
    noise_map = mock_ccd.MockNoiseMap(shape=(6, 6), value=2.0)
    background_noise_map = mock_ccd.MockBackgroundNoiseMap(shape=(6, 6), value=3.0)
    poisson_noise_map = mock_ccd.MockPoissonNoiseMap(shape=(6, 6), value=4.0)
    exposure_time_map = mock_ccd.MockExposureTimeMap(shape=(6, 6), value=5.0)
    background_sky_map = mock_ccd.MockBackgrondSkyMap(shape=(6, 6), value=6.0)

    return mock_ccd.MockCCDData(
        image=image,
        pixel_scale=1.0,
        psf=psf,
        noise_map=noise_map,
        background_noise_map=background_noise_map,
        poisson_noise_map=poisson_noise_map,
        exposure_time_map=exposure_time_map,
        background_sky_map=background_sky_map,
        name='mock_ccd_data_6x6')


# MASK #


@pytest.fixture(name="mask_5x5")
def make_mask_5x5():
    array = np.array([[True, True, True, True, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True, True, True, True, True]])

    return mock_mask.MockMask(array=array)


@pytest.fixture(name="mask_5x5_1_pix")
def make_mask_5x5_1_pix():
    array = np.array([[True, True, True, True, True],
                      [True, True, True, True, True],
                      [True, True, False, True, True],
                      [True, True, True, True, True],
                      [True, True, True, True, True]])

    return mock_mask.MockMask(array=array)


@pytest.fixture(name="blurring_mask_5x5")
def make_blurring_mask_5x5():
    array = np.array([[False, False, False, False, False],
                      [False, True, True, True, False],
                      [False, True, True, True, False],
                      [False, True, True, True, False],
                      [False, False, False, False, False]])

    return mock_mask.MockMask(array=array)


@pytest.fixture(name="padded_mask_7x7")
def make_padded_mask_7x7():
    array = np.full(fill_value=False, shape=(7, 7))

    return mock_mask.MockMask(array=array)


@pytest.fixture(name="mask_6x6")
def make_mask_6x6():
    array = np.array([[True, True, True, True, True, True],
                      [True, True, True, True, True, True],
                      [True, True, False, False, True, True],
                      [True, True, False, False, True, True],
                      [True, True, True, True, True, True],
                      [True, True, True, True, True, True]])

    return mock_mask.MockMask(array=array)


# MASKED DATA #

@pytest.fixture(name="image_1d_5x5")
def make_image_1d_5x5(image_5x5, mask_5x5):
    return mask_5x5.array_1d_from_array_2d(
        array_2d=image_5x5)


@pytest.fixture(name="noise_map_1d_5x5")
def make_noise_map_1d_5x5(noise_map_5x5, mask_5x5):
    return mask_5x5.array_1d_from_array_2d(
        array_2d=noise_map_5x5)


# GRIDS #


@pytest.fixture(name="regular_grid_5x5")
def make_regular_grid_5x5(mask_5x5):
    return mock_grids.MockRegularGrid(
        mask=mask_5x5)


@pytest.fixture(name="sub_grid_5x5")
def make_sub_grid_5x5(mask_5x5):
    return mock_grids.MockSubGrid(
        mask=mask_5x5)


@pytest.fixture(name="blurring_grid_5x5")
def make_blurring_grid_5x5():
    blurring_mask = np.array([[False, False, False, False, False],
                              [False, True, True, True, False],
                              [False, True, True, True, False],
                              [False, True, True, True, False],
                              [False, False, False, False, False]])

    return mock_grids.MockRegularGrid(mask=blurring_mask)


@pytest.fixture(name="cluster_grid_5x5")
def make_cluster_grid_5x5(mask_5x5):
    return mock_grids.MockClusterGrid.from_mask_and_cluster_pixel_scale(
        mask=mask_5x5, cluster_pixel_scale=mask_5x5.pixel_scale)


@pytest.fixture(name="grid_stack_5x5")
def make_grid_stack_5x5(regular_grid_5x5, sub_grid_5x5, blurring_grid_5x5):
    return mock_grids.MockGridStack(
        regular=regular_grid_5x5, sub=sub_grid_5x5, blurring=blurring_grid_5x5)


@pytest.fixture(name="grid_stack_simple")
def make_grid_stack_simple(regular_grid_5x5, sub_grid_5x5, blurring_grid_5x5):
    # Manually overwrite some sub-grid and blurring grid coodinates for easier deflection angle calculations

    grid_stack = mock_grids.MockGridStack(
        regular=regular_grid_5x5, sub=sub_grid_5x5, blurring=blurring_grid_5x5)

    grid_stack.regular[0] = np.array([1.0, 1.0])
    grid_stack.sub[0] = np.array([1.0, 1.0])
    grid_stack.sub[1] = np.array([1.0, 0.0])
    grid_stack.sub[2] = np.array([1.0, 1.0])
    grid_stack.sub[3] = np.array([1.0, 0.0])
    grid_stack.blurring[0] = np.array([1.0, 0.0])

    return grid_stack


@pytest.fixture(name="padded_regular_grid_5x5")
def make_padded_regular_grid_5x5():
    return mock_grids.MockPaddedRegularGrid(
        image_shape=(5, 5), psf_shape=(3, 3))


@pytest.fixture(name="padded_sub_grid_5x5")
def make_padded_sub_grid_5x5():
    return mock_grids.MockPaddedSubGrid(
        image_shape=(5, 5), psf_shape=(3, 3))


@pytest.fixture(name="padded_grid_stack_5x5")
def make_padded_grid_stack_5x5(padded_regular_grid_5x5, padded_sub_grid_5x5):
    return mock_grids.MockPaddedGridStack(
        regular=padded_regular_grid_5x5, sub=padded_sub_grid_5x5)


# BORDERS #

@pytest.fixture(name="border_5x5")
def make_border_5x5():
    return mock_grids.MockBorders(
        arr=np.array([0, 1, 2, 3, 5, 6, 7, 8]))


# CONVOLVERS #


@pytest.fixture(name="convolver_image_5x5")
def make_convolver_image_5x5(mask_5x5, blurring_mask_5x5, psf_3x3):
    return mock_convolution.MockConvolverImage(
        mask=mask_5x5, blurring_mask=blurring_mask_5x5, psf=psf_3x3)


@pytest.fixture(name="convolver_mapping_matrix_5x5")
def make_convolver_mapping_matrix_5x5(mask_5x5, psf_3x3):
    return mock_convolution.MockConvolverMappingMatrix(
        mask=mask_5x5, psf=psf_3x3)


#
# MODEL #
#

# PROFILES #


@pytest.fixture(name="lp_0")
def make_lp_0():
    # noinspection PyTypeChecker
    return lp.SphericalSersic(intensity=1.0, effective_radius=2.0, sersic_index=2.0)


@pytest.fixture(name="lp_1")
def make_lp_1():
    # noinspection PyTypeChecker
    return lp.SphericalSersic(intensity=2.0, effective_radius=2.0, sersic_index=2.0)


@pytest.fixture(name="mp_0")
def make_mp_0():
    # noinspection PyTypeChecker
    return mp.SphericalIsothermal(einstein_radius=1.0)


@pytest.fixture(name="mp_1")
def make_mp_1():
    # noinspection PyTypeChecker
    return mp.SphericalIsothermal(einstein_radius=2.0)


@pytest.fixture(name="lmp_0")
def make_lmp_0():
    return lmp.EllipticalSersicRadialGradient()


# GALAXY #

@pytest.fixture(name="gal_x1_lp")
def make_gal_x1_lp(lp_0):
    return g.Galaxy(
        redshift=0.5,
        light_profile_0=lp_0)


@pytest.fixture(name="gal_x2_lp")
def make_gal_x2_lp(lp_0, lp_1):
    return g.Galaxy(
        redshift=0.5,
        light_profile_0=lp_0,
        light_profile_1=lp_1)


@pytest.fixture(name="gal_x1_mp")
def make_gal_x1_mp(mp_0):
    return g.Galaxy(
        redshift=0.5,
        mass_profile_0=mp_0)


@pytest.fixture(name="gal_x2_mp")
def make_gal_x2_mp(mp_0, mp_1):
    return g.Galaxy(
        redshift=0.5,
        mass_profile_0=mp_0,
        mass_profile_1=mp_1)


@pytest.fixture(name="gal_x1_lp_x1_mp")
def make_gal_x1_lp_x1_mp(lp_0, mp_0):
    return g.Galaxy(
        redshift=0.5,
        light_profile_0=lp_0,
        mass_profile_0=mp_0)


@pytest.fixture(name="hyper_galaxy")
def make_hyper_galaxy():
    return g.HyperGalaxy(
        noise_factor=1.0, noise_power=1.0, contribution_factor=1.0)


# GALAXY DATA #


@pytest.fixture(name="gal_data_5x5")
def make_gal_data_5x5(image_5x5, noise_map_5x5):
    return gd.GalaxyData(image=image_5x5, noise_map=noise_map_5x5,
                         pixel_scale=image_5x5.pixel_scale)


@pytest.fixture(name="gal_fit_data_5x5_intensities")
def make_gal_fit_data_5x5_intensities(gal_data_5x5, mask_5x5):
    return gd.GalaxyFitData(
        galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2, use_intensities=True)


@pytest.fixture(name="gal_fit_data_5x5_convergence")
def make_gal_fit_data_5x5_convergence(gal_data_5x5, mask_5x5):
    return gd.GalaxyFitData(
        galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2, use_convergence=True)


@pytest.fixture(name="gal_fit_data_5x5_potential")
def make_gal_fit_data_5x5_potential(gal_data_5x5, mask_5x5):
    return gd.GalaxyFitData(
        galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2, use_potential=True)


@pytest.fixture(name="gal_fit_data_5x5_deflections_y")
def make_gal_fit_data_5x5_deflections_y(gal_data_5x5, mask_5x5):
    return gd.GalaxyFitData(
        galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2,
        use_deflections_y=True)


@pytest.fixture(name="gal_fit_data_5x5_deflections_x")
def make_gal_fit_data_5x5_deflections_x(gal_data_5x5, mask_5x5):
    return gd.GalaxyFitData(
        galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2,
        use_deflections_x=True)


# GALAXY FIT #


@pytest.fixture(name="gal_fit_5x5_intensities")
def make_gal_fit_5x5_intensities(gal_fit_data_5x5_intensities, gal_x1_lp):
    return galaxy_fit.GalaxyFit(
        galaxy_data=gal_fit_data_5x5_intensities, model_galaxies=[gal_x1_lp])


@pytest.fixture(name="gal_fit_5x5_convergence")
def make_gal_fit_5x5_convergence(gal_fit_data_5x5_convergence, gal_x1_mp):
    return galaxy_fit.GalaxyFit(
        galaxy_data=gal_fit_data_5x5_convergence, model_galaxies=[gal_x1_mp])


@pytest.fixture(name="gal_fit_5x5_potential")
def make_gal_fit_5x5_potential(gal_fit_data_5x5_potential, gal_x1_mp):
    return galaxy_fit.GalaxyFit(
        galaxy_data=gal_fit_data_5x5_potential, model_galaxies=[gal_x1_mp])


@pytest.fixture(name="gal_fit_5x5_deflections_y")
def make_gal_fit_5x5_deflections_y(gal_fit_data_5x5_deflections_y, gal_x1_mp):
    return galaxy_fit.GalaxyFit(
        galaxy_data=gal_fit_data_5x5_deflections_y, model_galaxies=[gal_x1_mp])


@pytest.fixture(name="gal_fit_5x5_deflections_x")
def make_gal_fit_5x5_deflections_x(gal_fit_data_5x5_deflections_x, gal_x1_mp):
    return galaxy_fit.GalaxyFit(
        galaxy_data=gal_fit_data_5x5_deflections_x, model_galaxies=[gal_x1_mp])


############
### LENS ###
############

##### Lens Data ###

@pytest.fixture(name="lens_data_5x5")
def make_lens_data_5x5(ccd_data_5x5, mask_5x5, grid_stack_5x5, padded_grid_stack_5x5,
                       border_5x5,
                       convolver_image_5x5, convolver_mapping_matrix_5x5,
                       cluster_grid_5x5):
    return mock_lens_data.MockLensData(
        ccd_data=ccd_data_5x5, mask=mask_5x5, grid_stack=grid_stack_5x5,
        padded_grid_stack=padded_grid_stack_5x5,
        border=border_5x5, convolver_image=convolver_image_5x5,
        convolver_mapping_matrix=convolver_mapping_matrix_5x5,
        cluster=cluster_grid_5x5)


### Plane ####

from autolens.lens import plane as pl


@pytest.fixture(name='plane_5x5')
def make_plane_5x5(gal_x1_lp_x1_mp, grid_stack_5x5):
    return pl.Plane(
        galaxies=[gal_x1_lp_x1_mp], grid_stack=grid_stack_5x5,
        compute_deflections=False)


### Ray Tracing ####

from autolens.lens import ray_tracing


@pytest.fixture(name='tracer_x1_plane_5x5')
def make_tracer_x1_plane_5x5(gal_x1_lp, grid_stack_5x5):
    return ray_tracing.TracerImagePlane(
        lens_galaxies=[gal_x1_lp], image_plane_grid_stack=grid_stack_5x5)


@pytest.fixture(name='tracer_x2_plane_5x5')
def make_tracer_x2_plane_5x5(gal_x1_lp, gal_x1_mp, grid_stack_5x5):
    return ray_tracing.TracerImageSourcePlanes(
        lens_galaxies=[gal_x1_mp, gal_x1_lp], source_galaxies=[gal_x1_lp],
        image_plane_grid_stack=grid_stack_5x5)


### Lens Fit ####

from autolens.lens import lens_fit


@pytest.fixture(name='lens_fit_x1_plane_5x5')
def make_lens_fit_x1_plane_5x5(lens_data_5x5, tracer_x1_plane_5x5):
    return lens_fit.LensDataFit.for_data_and_tracer(
        lens_data=lens_data_5x5, tracer=tracer_x1_plane_5x5)


@pytest.fixture(name='lens_fit_x2_plane_5x5')
def make_lens_fit_x2_plane_5x5(lens_data_5x5, tracer_x2_plane_5x5):
    return lens_fit.LensDataFit.for_data_and_tracer(
        lens_data=lens_data_5x5, tracer=tracer_x2_plane_5x5)


### Sensitive Fit ###

from autolens.lens import sensitivity_fit


@pytest.fixture(name='sensitivity_fit_5x5')
def make_sensitivity_fit_5x5(lens_data_5x5):
    lens_galaxy = g.Galaxy(
        redshift=0.5,
        mass=mp.SphericalIsothermal(einstein_radius=1.0))

    lens_subhalo = g.Galaxy(
        redshift=0.5,
        mass=mp.SphericalIsothermal(einstein_radius=0.1))

    source_galaxy = g.Galaxy(
        redshift=0.5,
        light=lp.EllipticalSersic(intensity=1.0))

    tracer_normal = ray_tracing.TracerImageSourcePlanes(
        lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
        image_plane_grid_stack=lens_data_5x5.grid_stack)

    tracer_sensitivity = ray_tracing.TracerImageSourcePlanes(
        lens_galaxies=[lens_galaxy, lens_subhalo], source_galaxies=[source_galaxy],
        image_plane_grid_stack=lens_data_5x5.grid_stack)

    return sensitivity_fit.SensitivityProfileFit(
        lens_data=lens_data_5x5, tracer_normal=tracer_normal,
        tracer_sensitive=tracer_sensitivity)


##############
## PIPELINE ##
#############

### Phase ###

from autolens.pipeline.phase import phase_imaging
from test.unit.mock.pipeline import mock_pipeline


@pytest.fixture(name="mask_function_5x5_1_pix")
def make_mask_function_5x5_1_pix():
    def mask_function_5x5_1_pix(image):
        array = np.array([[True, True, True, True, True],
                          [True, True, True, True, True],
                          [True, True, False, True, True],
                          [True, True, True, True, True],
                          [True, True, True, True, True]])

        return mock_mask.MockMask(array=array)

    return mask_function_5x5_1_pix


@pytest.fixture(name="mask_function_5x5")
def make_mask_function_5x5():
    def mask_function_5x5(image):
        array = np.array([[True, True, True, True, True],
                          [True, False, False, False, True],
                          [True, False, False, False, True],
                          [True, False, False, False, True],
                          [True, True, True, True, True]])

        return mock_mask.MockMask(array=array)

    return mask_function_5x5


@pytest.fixture(name="phase_5x5")
def make_phase_5x5(mask_function_5x5):
    return phase_imaging.LensSourcePlanePhase(
        optimizer_class=mock_pipeline.MockNLO, mask_function=mask_function_5x5,
        phase_name='test_phase')


@pytest.fixture(name="hyper_model_image_5x5")
def make_hyper_model_image_5x5(grid_stack_5x5):
    return grid_stack_5x5.regular.scaled_array_2d_from_array_1d(array_1d=np.ones(9))


@pytest.fixture(name="hyper_galaxy_image_0_5x5")
def make_hyper_galaxy_image_0_5x5(grid_stack_5x5):
    return grid_stack_5x5.regular.scaled_array_2d_from_array_1d(
        array_1d=2.0 * np.ones(9))


@pytest.fixture(name="hyper_galaxy_image_1_5x5")
def make_hyper_galaxy_image_1_5x5(grid_stack_5x5):
    return grid_stack_5x5.regular.scaled_array_2d_from_array_1d(
        array_1d=3.0 * np.ones(9))


@pytest.fixture(name="contribution_map_5x5")
def make_contribution_map_5x5(hyper_model_image_5x5, hyper_galaxy_image_0_5x5,
                              hyper_galaxy):
    return hyper_galaxy.contribution_map_from_hyper_images(
        hyper_model_image=hyper_model_image_5x5,
        hyper_galaxy_image=hyper_galaxy_image_0_5x5)


@pytest.fixture(name="hyper_noise_map_5x5")
def make_hyper_noise_map_5x5(noise_map_5x5, contribution_map_5x5, hyper_galaxy):
    hyper_noise = hyper_galaxy.hyper_noise_map_from_contribution_map(
        noise_map=noise_map_5x5, contribution_map=contribution_map_5x5)
    return noise_map_5x5 + hyper_noise


@pytest.fixture(name="results_5x5")
def make_results(hyper_model_image_5x5, hyper_galaxy_image_0_5x5,
                 hyper_galaxy_image_1_5x5):
    return mock_pipeline.MockResults(
        model_image=hyper_model_image_5x5,
        galaxy_images=[hyper_galaxy_image_0_5x5, hyper_galaxy_image_1_5x5])


@pytest.fixture(name="results_collection_5x5")
def make_results_collection(results_5x5):
    results_collection = af.pipeline.ResultsCollection()
    results_collection.add("phase", results_5x5)
    return results_collection

import autofit as af
import autolens as al
from autofit.optimize.non_linear.non_linear import Paths
from test_autoarray.unit.conftest import *
from test_autoastro.unit.conftest import *
from test_autolens.mock.lens import mock_lens_data
from test_autolens.mock.pipeline import mock_pipeline

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    af.conf.instance = af.conf.Config(
        path.join(directory, "test_files/config"), path.join(directory, "output")
    )


############
# LENS #
############

# Lens Data #


@pytest.fixture(name="lens_imaging_7x7")
def make_lens_imaging_7x7(
        imaging_7x7,
        mask_7x7,
        sub_grid_7x7,
        blurring_grid_7x7,
        convolver_7x7,
        binned_grid_7x7,
):
    return mock_lens_data.MockLensImagingData(
        imaging=imaging_7x7,
        mask=mask_7x7,
        grid=sub_grid_7x7,
        blurring_grid=blurring_grid_7x7,
        convolver=convolver_7x7,
        binned_grid=binned_grid_7x7,
    )


@pytest.fixture(name="lens_interferometer_7")
def make_lens_interferometer_7(
        interferometer_7, mask_7x7, sub_grid_7x7, transformer_7x7_7, binned_grid_7x7
):
    return mock_lens_data.MockLensUVPlaneData(
        interferometer=interferometer_7,
        mask=mask_7x7,
        grid=sub_grid_7x7,
        transformer=transformer_7x7_7,
        binned_grid=binned_grid_7x7,
    )


# Plane #


@pytest.fixture(name="plane_7x7")
def make_plane_7x7(gal_x1_lp_x1_mp):
    return al.Plane(galaxies=[gal_x1_lp_x1_mp])


# Ray Tracing #


@pytest.fixture(name="tracer_x1_plane_7x7")
def make_tracer_x1_plane_7x7(gal_x1_lp):
    return al.Tracer.from_galaxies(galaxies=[gal_x1_lp])


@pytest.fixture(name="tracer_x2_plane_7x7")
def make_tracer_x2_plane_7x7(lp_0, gal_x1_lp, gal_x1_mp):
    source_gal_x1_lp = al.Galaxy(redshift=1.0, light_profile_0=lp_0)

    return al.Tracer.from_galaxies(galaxies=[gal_x1_mp, gal_x1_lp, source_gal_x1_lp])


# Lens Fit #


@pytest.fixture(name="lens_imaging_fit_x1_plane_7x7")
def make_lens_imaging_fit_x1_plane_7x7(lens_imaging_7x7, tracer_x1_plane_7x7):
    return al.LensImagingFit.from_lens_data_and_tracer(
        lens_data=lens_imaging_7x7, tracer=tracer_x1_plane_7x7
    )


@pytest.fixture(name="lens_imaging_fit_x2_plane_7x7")
def make_lens_imaging_fit_x2_plane_7x7(lens_imaging_7x7, tracer_x2_plane_7x7):
    return al.LensImagingFit.from_lens_data_and_tracer(
        lens_data=lens_imaging_7x7, tracer=tracer_x2_plane_7x7
    )


@pytest.fixture(name="mask_function_7x7_1_pix")
def make_mask_function_7x7_1_pix():
    # noinspection PyUnusedLocal
    def mask_function_7x7_1_pix(image, sub_size):
        array = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        return mock_mask.MockMask(array_2d=array, sub_size=sub_size)

    return mask_function_7x7_1_pix


@pytest.fixture(name="mask_function_7x7")
def make_mask_function_7x7():
    # noinspection PyUnusedLocal
    def mask_function_7x7(image, sub_size):
        array = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        return aa.mask(mask_2d=array, pixel_scales=(1.0, 1.0), sub_size=sub_size)

    return mask_function_7x7


@pytest.fixture(name="phase_data_7x7")
def make_phase_data(mask_function_7x7):
    return al.PhaseData(
        optimizer_class=mock_pipeline.MockNLO,
        paths=Paths(
            phase_tag="",
            phase_name="test_phase"
        )
    )


@pytest.fixture(name="phase_imaging_7x7")
def make_phase_imaging_7x7(mask_function_7x7):
    return al.PhaseImaging(
        optimizer_class=mock_pipeline.MockNLO,
        mask_function=mask_function_7x7,
        paths=Paths(
            phase_name="test_phase",
        )
    )


@pytest.fixture(name="hyper_model_image_7x7")
def make_hyper_model_image_7x7(grid_7x7):
    return grid_7x7.mapping.array_from_array_1d(array_1d=np.ones(9))


@pytest.fixture(name="hyper_galaxy_image_0_7x7")
def make_hyper_galaxy_image_0_7x7(grid_7x7):
    return grid_7x7.mapping.array_from_array_1d(array_1d=2.0 * np.ones(9))


@pytest.fixture(name="hyper_galaxy_image_1_7x7")
def make_hyper_galaxy_image_1_7x7(grid_7x7):
    return grid_7x7.mapping.array_from_array_1d(array_1d=3.0 * np.ones(9))


@pytest.fixture(name="contribution_map_7x7")
def make_contribution_map_7x7(
        hyper_model_image_7x7, hyper_galaxy_image_0_7x7, hyper_galaxy
):
    return hyper_galaxy.contribution_map_from_hyper_images(
        hyper_model_image=hyper_model_image_7x7,
        hyper_galaxy_image=hyper_galaxy_image_0_7x7,
    )


@pytest.fixture(name="hyper_noise_map_7x7")
def make_hyper_noise_map_7x7(noise_map_7x7, contribution_map_7x7, hyper_galaxy):
    hyper_noise = hyper_galaxy.hyper_noise_map_from_contribution_map(
        noise_map=noise_map_7x7, contribution_map=contribution_map_7x7
    )
    return noise_map_7x7 + hyper_noise


@pytest.fixture(name="results_7x7")
def make_results(
        mask_7x7, hyper_model_image_7x7, hyper_galaxy_image_0_7x7, hyper_galaxy_image_1_7x7
):
    return mock_pipeline.MockResults(
        model_image=hyper_model_image_7x7,
        galaxy_images=[hyper_galaxy_image_0_7x7, hyper_galaxy_image_1_7x7],
        mask=mask_7x7,
    )


@pytest.fixture(name="results_collection_7x7")
def make_results_collection(results_7x7):
    results_collection = af.ResultsCollection()
    results_collection.add("phase", results_7x7)
    return results_collection

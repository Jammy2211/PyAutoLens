import autofit as af
import autolens as al
from autolens.fit.fit import ImagingFit, InterferometerFit

from test_autolens.mock import mock_pipeline
from test_autoastro.unit.conftest import *

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


@pytest.fixture(name="masked_imaging_7x7")
def make_masked_imaging_7x7(imaging_7x7, sub_mask_7x7):
    return al.masked.imaging.manual(imaging=imaging_7x7, mask=sub_mask_7x7)


@pytest.fixture(name="masked_interferometer_7")
def make_masked_interferometer_7(
    interferometer_7, mask_7x7, visibilities_mask_7x2, sub_grid_7x7, transformer_7x7_7
):
    return al.masked.interferometer.manual(
        interferometer=interferometer_7,
        visibilities_mask=visibilities_mask_7x2,
        real_space_mask=mask_7x7,
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


@pytest.fixture(name="tracer_x2_plane_inversion_7x7")
def make_tracer_x2_plane_inversion_7x7(lp_0, gal_x1_lp, gal_x1_mp):
    source_gal_inversion = al.Galaxy(
        redshift=1.0,
        pixelization=al.pix.Rectangular(),
        regularization=al.reg.Constant(),
    )

    return al.Tracer.from_galaxies(
        galaxies=[gal_x1_mp, gal_x1_lp, source_gal_inversion]
    )


# Lens Fit #


@pytest.fixture(name="masked_imaging_fit_x1_plane_7x7")
def make_masked_imaging_fit_x1_plane_7x7(masked_imaging_7x7, tracer_x1_plane_7x7):
    return ImagingFit(masked_imaging=masked_imaging_7x7, tracer=tracer_x1_plane_7x7)


@pytest.fixture(name="masked_imaging_fit_x2_plane_7x7")
def make_masked_imaging_fit_x2_plane_7x7(masked_imaging_7x7, tracer_x2_plane_7x7):
    return ImagingFit(masked_imaging=masked_imaging_7x7, tracer=tracer_x2_plane_7x7)


@pytest.fixture(name="masked_imaging_fit_x2_plane_inversion_7x7")
def make_masked_imaging_fit_x2_plane_inversion_7x7(
    masked_imaging_7x7, tracer_x2_plane_inversion_7x7
):
    return ImagingFit(
        masked_imaging=masked_imaging_7x7, tracer=tracer_x2_plane_inversion_7x7
    )


@pytest.fixture(name="masked_interferometer_fit_x1_plane_7x7")
def make_masked_interferometer_fit_x1_plane_7x7(
    masked_interferometer_7, tracer_x1_plane_7x7
):
    return InterferometerFit(
        masked_interferometer=masked_interferometer_7, tracer=tracer_x1_plane_7x7
    )


@pytest.fixture(name="masked_interferometer_fit_x2_plane_7x7")
def make_masked_interferometer_fit_x2_plane_7x7(
    masked_interferometer_7, tracer_x2_plane_7x7
):
    return InterferometerFit(
        masked_interferometer=masked_interferometer_7, tracer=tracer_x2_plane_7x7
    )


@pytest.fixture(name="masked_interferometer_fit_x2_plane_inversion_7x7")
def make_masked_interferometer_fit_x2_plane_inversion_7x7(
    masked_interferometer_7, tracer_x2_plane_inversion_7x7
):
    return InterferometerFit(
        masked_interferometer=masked_interferometer_7,
        tracer=tracer_x2_plane_inversion_7x7,
    )


@pytest.fixture(name="mask_7x7_1_pix")
def make_mask_7x7_1_pix():
    # noinspection PyUnusedLocal

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

    return aa.mask.manual(mask_2d=array)


@pytest.fixture(name="phase_dataset_7x7")
def make_phase_data(mask_7x7):
    return al.PhaseDataset(
        optimizer_class=mock_pipeline.MockNLO, phase_tag="", phase_name="test_phase"
    )


@pytest.fixture(name="phase_imaging_7x7")
def make_phase_imaging_7x7():
    return al.PhaseImaging(
        optimizer_class=mock_pipeline.MockNLO, phase_name="test_phase"
    )


@pytest.fixture(name="phase_interferometer_7")
def make_phase_interferometer_7(mask_7x7):
    return al.PhaseInterferometer(
        optimizer_class=mock_pipeline.MockNLO,
        real_space_mask=mask_7x7,
        phase_name="test_phase",
    )


@pytest.fixture(name="hyper_model_image_7x7")
def make_hyper_model_image_7x7(mask_7x7):
    return al.masked.array.full(fill_value=5.0, mask=mask_7x7)


@pytest.fixture(name="hyper_galaxy_image_0_7x7")
def make_hyper_galaxy_image_0_7x7(mask_7x7):
    return al.masked.array.full(fill_value=2.0, mask=mask_7x7)


@pytest.fixture(name="hyper_galaxy_image_1_7x7")
def make_hyper_galaxy_image_1_7x7(mask_7x7):
    return al.masked.array.full(fill_value=3.0, mask=mask_7x7)


@pytest.fixture(name="hyper_galaxy_image_path_dict_7x7")
def make_hyper_galaxy_image_path_dict_7x7(
    hyper_galaxy_image_0_7x7, hyper_galaxy_image_1_7x7
):
    hyper_galaxy_image_path_dict = {}

    hyper_galaxy_image_path_dict[("g0",)] = hyper_galaxy_image_0_7x7
    hyper_galaxy_image_path_dict[("g1",)] = hyper_galaxy_image_1_7x7

    return hyper_galaxy_image_path_dict


@pytest.fixture(name="contribution_map_7x7")
def make_contribution_map_7x7(
    hyper_model_image_7x7, hyper_galaxy_image_0_7x7, hyper_galaxy
):
    return hyper_galaxy.contribution_map_from_hyper_images(
        hyper_model_image=hyper_model_image_7x7,
        hyper_galaxy_image=hyper_galaxy_image_0_7x7,
    )


@pytest.fixture(name="hyper_noise_map_7x7")
def make_hyper_noise_map_7x7(
    masked_imaging_fit_x2_plane_7x7, contribution_map_7x7, hyper_galaxy
):
    hyper_noise = hyper_galaxy.hyper_noise_map_from_contribution_map(
        noise_map=masked_imaging_fit_x2_plane_7x7.noise_map,
        contribution_map=contribution_map_7x7,
    )
    return masked_imaging_fit_x2_plane_7x7.noise_map + hyper_noise


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

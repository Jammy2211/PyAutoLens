from autoconf import conf
import autolens as al
from test_autogalaxy.unit.conftest import *
from test_autolens import mock

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "config"), path.join(directory, "pipeline/output")
    )


############
# AutoLens #
############

# Lens Datasets #


@pytest.fixture(name="masked_imaging_7x7")
def make_masked_imaging_7x7(imaging_7x7, sub_mask_7x7):
    return al.MaskedImaging(imaging=imaging_7x7, mask=sub_mask_7x7)


@pytest.fixture(name="masked_imaging_7x7_grid")
def make_masked_imaging_7x7_grid(imaging_7x7, sub_mask_7x7):
    return al.MaskedImaging(imaging=imaging_7x7, mask=sub_mask_7x7, grid_class=aa.Grid)


@pytest.fixture(name="masked_interferometer_7")
def make_masked_interferometer_7(
    interferometer_7, mask_7x7, visibilities_mask_7x2, sub_grid_7x7, transformer_7x7_7
):
    return al.MaskedInterferometer(
        interferometer=interferometer_7,
        visibilities_mask=visibilities_mask_7x2,
        real_space_mask=mask_7x7,
        transformer_class=aa.TransformerDFT,
    )


@pytest.fixture(name="masked_interferometer_7_grid")
def make_masked_interferometer_7_grid(
    interferometer_7, mask_7x7, visibilities_mask_7x2, sub_grid_7x7, transformer_7x7_7
):
    return al.MaskedInterferometer(
        interferometer=interferometer_7,
        visibilities_mask=visibilities_mask_7x2,
        real_space_mask=mask_7x7,
        grid_class=aa.Grid,
        transformer_class=aa.TransformerDFT,
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
    return al.FitImaging(masked_imaging=masked_imaging_7x7, tracer=tracer_x1_plane_7x7)


@pytest.fixture(name="masked_imaging_fit_x2_plane_7x7")
def make_masked_imaging_fit_x2_plane_7x7(masked_imaging_7x7, tracer_x2_plane_7x7):
    return al.FitImaging(masked_imaging=masked_imaging_7x7, tracer=tracer_x2_plane_7x7)


@pytest.fixture(name="masked_imaging_fit_x2_plane_inversion_7x7")
def make_masked_imaging_fit_x2_plane_inversion_7x7(
    masked_imaging_7x7, tracer_x2_plane_inversion_7x7
):
    return al.FitImaging(
        masked_imaging=masked_imaging_7x7, tracer=tracer_x2_plane_inversion_7x7
    )


@pytest.fixture(name="masked_interferometer_fit_x1_plane_7x7")
def make_masked_interferometer_fit_x1_plane_7x7(
    masked_interferometer_7, tracer_x1_plane_7x7
):
    return al.FitInterferometer(
        masked_interferometer=masked_interferometer_7, tracer=tracer_x1_plane_7x7
    )


@pytest.fixture(name="masked_interferometer_fit_x2_plane_7x7")
def make_masked_interferometer_fit_x2_plane_7x7(
    masked_interferometer_7, tracer_x2_plane_7x7
):
    return al.FitInterferometer(
        masked_interferometer=masked_interferometer_7, tracer=tracer_x2_plane_7x7
    )


@pytest.fixture(name="masked_interferometer_fit_x2_plane_inversion_7x7")
def make_masked_interferometer_fit_x2_plane_inversion_7x7(
    masked_interferometer_7, tracer_x2_plane_inversion_7x7
):
    return al.FitInterferometer(
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

    return aa.Mask.manual(mask=array)


@pytest.fixture(name="phase_dataset_7x7")
def make_phase_data(mask_7x7):
    return al.PhaseDataset(phase_name="test_phase", search=mock.MockSearch())


@pytest.fixture(name="phase_imaging_7x7")
def make_phase_imaging_7x7():
    return al.PhaseImaging(phase_name="test_phase", search=mock.MockSearch())


@pytest.fixture(name="phase_interferometer_7")
def make_phase_interferometer_7(mask_7x7):
    return al.PhaseInterferometer(
        phase_name="test_phase", search=mock.MockSearch(), real_space_mask=mask_7x7
    )

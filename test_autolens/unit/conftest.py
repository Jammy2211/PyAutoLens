from os import path

import numpy as np
import pytest

import autoarray as aa
import autolens as al
from autoarray import (
    MaskedInterferometer,
    SettingsMaskedInterferometer,
    TransformerNUFFT,
)
from autoconf import conf
from autolens import mock

directory = path.dirname(path.realpath(__file__))


############
# AutoLens #
############

# Lens Datasets #


@pytest.fixture(name="config", autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../config"), path.join(directory, "../pipeline/output")
    )
    return conf.instance


@pytest.fixture(name="masked_imaging_7x7")
def make_masked_imaging_7x7(imaging_7x7, sub_mask_7x7):
    return al.MaskedImaging(
        imaging=imaging_7x7,
        mask=sub_mask_7x7,
        settings=al.SettingsMaskedImaging(sub_size=1),
    )


@pytest.fixture(name="masked_interferometer_7")
def make_masked_interferometer_7():
    return MaskedInterferometer(
        interferometer=mock.make_interferometer_7(),
        visibilities_mask=mock.make_visibilities_mask_7x2(),
        real_space_mask=mock.make_mask_7x7(),
        settings=SettingsMaskedInterferometer(
            sub_size=1, transformer_class=TransformerNUFFT
        ),
    )


@pytest.fixture(name="samples_with_result")
def make_samples_with_result():
    return mock.make_samples_with_result()


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7():
    return mock.make_noise_map_7x7()


@pytest.fixture(name="masked_interferometer_7_grid")
def make_masked_interferometer_7_grid(
    interferometer_7, mask_7x7, visibilities_mask_7x2, sub_grid_7x7, transformer_7x7_7
):
    return al.MaskedInterferometer(
        interferometer=interferometer_7,
        visibilities_mask=visibilities_mask_7x2,
        real_space_mask=mask_7x7,
        settings=al.SettingsMaskedInterferometer(
            grid_class=aa.Grid, sub_size=1, transformer_class=aa.TransformerDFT
        ),
    )


@pytest.fixture(name="gal_x1_mp")
def make_gal_x1_mp():
    return mock.make_gal_x1_mp()


@pytest.fixture(name="imaging_7x7")
def make_imaging_7x7():
    return mock.make_imaging_7x7()


@pytest.fixture(name="sub_mask_7x7")
def make_sub_mask_7x7():
    return mock.make_sub_mask_7x7()


@pytest.fixture(name="lp_0")
def make_lp_0():
    return mock.make_lp_0()


@pytest.fixture(name="gal_x1_lp")
def make_gal_x1_lp():
    return mock.make_gal_x1_lp()


@pytest.fixture(name="include_all")
def make_include_all():
    return mock.make_include_all()


@pytest.fixture(name="fit_interferometer_7")
def make_masked_interferometer_fit_x1_plane_7(masked_interferometer_7):
    return mock.make_masked_interferometer_fit_x1_plane_7()


@pytest.fixture(name="mask_7x7")
def make_mask_7x7():
    return mock.make_mask_7x7()


@pytest.fixture(name="positions_7x7")
def make_positions_7x7():
    return mock.make_positions_7x7()


@pytest.fixture(name="transformer_7x7_7")
def make_transformer_7x7_7():
    return mock.make_transformer_7x7_7()


# Ray Tracing #


@pytest.fixture(name="sub_grid_7x7_simple")
def make_sub_grid_7x7_simple():
    return mock.make_sub_grid_7x7_simple()


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


@pytest.fixture(name="interferometer_7")
def make_interferometer_7():
    return mock.make_interferometer_7()


@pytest.fixture(name="visibilities_mask_7x2")
def make_visibilities_mask_7x2():
    return np.full(fill_value=False, shape=(7, 2))


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


@pytest.fixture(name="sub_grid_7x7")
def make_sub_grid_7x7():
    return mock.make_sub_grid_7x7()


@pytest.fixture(name="masked_interferometer_fit_x2_plane_inversion_7x7")
def make_masked_interferometer_fit_x2_plane_inversion_7x7(
    masked_interferometer_7, tracer_x2_plane_inversion_7x7
):
    return al.FitInterferometer(
        masked_interferometer=masked_interferometer_7,
        tracer=tracer_x2_plane_inversion_7x7,
    )


@pytest.fixture(name="phase_interferometer_7")
def make_phase_interferometer_7(mask_7x7):
    return al.PhaseInterferometer(
        search=mock.MockSearch("test_phase"), real_space_mask=mask_7x7
    )


@pytest.fixture(name="grid_iterate_7x7")
def make_grid_iterate_7x7():
    return mock.make_grid_iterate_7x7()


@pytest.fixture(name="psf_3x3")
def make_psf_3x3():
    return mock.make_psf_3x3()


@pytest.fixture(name="phase_imaging_7x7")
def make_phase_imaging_7x7():
    return mock.make_phase_imaging_7x7()


@pytest.fixture(name="mask_7x7_1_pix")
def make_mask_7x7_1_pix():
    return mock.make_mask_7x7_1_pix()


@pytest.fixture(name="convolver_7x7")
def make_convolver_7x7():
    return mock.make_convolver_7x7()


@pytest.fixture(name="visibilities_7x2")
def make_visibilities_7():
    return aa.mock.make_visibilities_7()


@pytest.fixture(name="noise_map_7x2")
def make_noise_map_7():
    return aa.mock.make_noise_map_7()


@pytest.fixture(name="blurring_grid_7x7")
def make_blurring_grid_7x7(blurring_mask_7x7):
    return aa.Grid.from_mask(mask=blurring_mask_7x7)


@pytest.fixture(name="blurring_mask_7x7")
def make_blurring_mask_7x7():
    return aa.mock.make_blurring_mask_7x7()


@pytest.fixture(name="grid_7x7")
def make_grid_7x7():
    return aa.Grid.from_mask(mask=mock.make_mask_7x7())


@pytest.fixture(name="hyper_galaxy_image_0_7x7")
def make_hyper_galaxy_image_0_7x7():
    return mock.make_hyper_galaxy_image_0_7x7()


@pytest.fixture(name="hyper_model_image_7x7")
def make_hyper_model_image_7x7():
    return mock.make_hyper_model_image_7x7()


@pytest.fixture(name="image_7x7")
def make_image_7x7():
    return mock.make_image_7x7()


@pytest.fixture(name="hyper_galaxy_image_path_dict_7x7")
def make_hyper_galaxy_image_path_dict_7x7():
    return mock.make_hyper_galaxy_image_path_dict_7x7()

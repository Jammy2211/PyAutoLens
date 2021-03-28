import autolens as al
from autogalaxy.mock.fixtures import *
from autofit.mock.mock import MockSearch
from autolens.mock.mock import MockPositionsSolver


def make_masked_imaging_7x7():
    return al.MaskedImaging(
        imaging=make_imaging_7x7(),
        mask=make_sub_mask_7x7(),
        settings=al.SettingsMaskedImaging(sub_size=1),
    )


def make_masked_interferometer_7():
    return al.MaskedInterferometer(
        interferometer=make_interferometer_7(),
        visibilities_mask=make_visibilities_mask_7(),
        real_space_mask=make_mask_7x7(),
        settings=al.SettingsMaskedInterferometer(
            sub_size=1, transformer_class=al.TransformerNUFFT
        ),
    )


def make_masked_interferometer_7_grid():
    return al.MaskedInterferometer(
        interferometer=make_interferometer_7(),
        visibilities_mask=make_visibilities_mask_7(),
        real_space_mask=make_mask_7x7(),
        settings=al.SettingsMaskedInterferometer(
            grid_class=al.Grid2D, sub_size=1, transformer_class=al.TransformerDFT
        ),
    )


def make_positions_x2():
    return al.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])


def make_positions_noise_map_x2():
    return al.ValuesIrregular(values=[1.0, 1.0])


def make_fluxes_x2():
    return al.ValuesIrregular(values=[1.0, 2.0])


def make_fluxes_noise_map_x2():
    return al.ValuesIrregular(values=[1.0, 1.0])


def make_point_source_dataset():
    return al.PointSourceDataset(
        name="point_0",
        positions=make_positions_x2(),
        positions_noise_map=make_positions_noise_map_x2(),
        fluxes=make_fluxes_x2(),
        fluxes_noise_map=make_fluxes_noise_map_x2(),
    )


def make_point_source_dict():
    return al.PointSourceDict(point_source_dataset_list=[make_point_source_dataset()])


def make_tracer_x1_plane_7x7():
    return al.Tracer.from_galaxies(galaxies=[make_gal_x1_lp()])


def make_tracer_x2_plane_7x7():

    source_gal_x1_lp = al.Galaxy(redshift=1.0, light_profile_0=make_lp_0())

    return al.Tracer.from_galaxies(
        galaxies=[make_gal_x1_mp(), make_gal_x1_lp(), source_gal_x1_lp]
    )


def make_tracer_x2_plane_inversion_7x7():

    source_gal_inversion = al.Galaxy(
        redshift=1.0,
        pixelization=al.pix.Rectangular(),
        regularization=al.reg.Constant(),
    )

    return al.Tracer.from_galaxies(
        galaxies=[make_gal_x1_mp(), make_gal_x1_lp(), source_gal_inversion]
    )


def make_masked_imaging_fit_x1_plane_7x7():
    return al.FitImaging(
        masked_imaging=make_masked_imaging_7x7(), tracer=make_tracer_x1_plane_7x7()
    )


def make_masked_imaging_fit_x2_plane_7x7():
    return al.FitImaging(
        masked_imaging=make_masked_imaging_7x7(), tracer=make_tracer_x2_plane_7x7()
    )


def make_masked_imaging_fit_x2_plane_inversion_7x7():
    return al.FitImaging(
        masked_imaging=make_masked_imaging_7x7(),
        tracer=make_tracer_x2_plane_inversion_7x7(),
    )


def make_masked_interferometer_fit_x1_plane_7x7():
    return al.FitInterferometer(
        masked_interferometer=make_masked_interferometer_7(),
        tracer=make_tracer_x1_plane_7x7(),
    )


def make_masked_interferometer_fit_x2_plane_7x7():
    return al.FitInterferometer(
        masked_interferometer=make_masked_interferometer_7(),
        tracer=make_tracer_x2_plane_7x7(),
    )


def make_masked_interferometer_fit_x2_plane_inversion_7x7():
    return al.FitInterferometer(
        masked_interferometer=make_masked_interferometer_7(),
        tracer=make_tracer_x2_plane_inversion_7x7(),
    )


def make_analysis_imaging_7x7():
    return al.AnalysisImaging(dataset=make_masked_imaging_7x7())


def make_analysis_interferometer_7():
    return al.AnalysisInterferometer(dataset=make_masked_interferometer_7())


def make_analysis_point_source_x2():
    return al.AnalysisPointSource(
        positions=make_positions_x2(),
        noise_map=make_positions_noise_map_x2(),
        solver=MockPositionsSolver(model_positions=make_positions_x2()),
    )

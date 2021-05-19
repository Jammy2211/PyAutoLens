import autolens as al
from autogalaxy.mock.fixtures import *
from autofit.mock.mock import MockSearch
from autolens.mock.mock import MockPositionsSolver


def make_positions_x2():
    return al.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])


def make_positions_noise_map_x2():
    return al.ValuesIrregular(values=[1.0, 1.0])


def make_fluxes_x2():
    return al.ValuesIrregular(values=[1.0, 2.0])


def make_fluxes_noise_map_x2():
    return al.ValuesIrregular(values=[1.0, 1.0])


def make_point_dataset():
    return al.PointDataset(
        name="point_0",
        positions=make_positions_x2(),
        positions_noise_map=make_positions_noise_map_x2(),
        fluxes=make_fluxes_x2(),
        fluxes_noise_map=make_fluxes_noise_map_x2(),
    )


def make_point_dict():
    return al.PointDict(point_dataset_list=[make_point_dataset()])


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


def make_fit_imaging_x1_plane_7x7():
    return al.FitImaging(
        imaging=make_masked_imaging_7x7(), tracer=make_tracer_x1_plane_7x7()
    )


def make_fit_imaging_x2_plane_7x7():
    return al.FitImaging(
        imaging=make_masked_imaging_7x7(), tracer=make_tracer_x2_plane_7x7()
    )


def make_fit_imaging_x2_plane_inversion_7x7():
    return al.FitImaging(
        imaging=make_masked_imaging_7x7(), tracer=make_tracer_x2_plane_inversion_7x7()
    )


def make_fit_interferometer_x1_plane_7x7():
    return al.FitInterferometer(
        interferometer=make_interferometer_7(), tracer=make_tracer_x1_plane_7x7()
    )


def make_fit_interferometer_x2_plane_7x7():
    return al.FitInterferometer(
        interferometer=make_interferometer_7(), tracer=make_tracer_x2_plane_7x7()
    )


def make_fit_interferometer_x2_plane_inversion_7x7():
    return al.FitInterferometer(
        interferometer=make_interferometer_7(),
        tracer=make_tracer_x2_plane_inversion_7x7(),
    )


def make_analysis_imaging_7x7():
    return al.AnalysisImaging(dataset=make_masked_imaging_7x7())


def make_analysis_interferometer_7():
    return al.AnalysisInterferometer(dataset=make_interferometer_7())


def make_analysis_point_x2():
    return al.AnalysisPoint(
        point_dict=make_point_dict(),
        solver=MockPositionsSolver(model_positions=make_positions_x2()),
    )

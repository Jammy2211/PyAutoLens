import autolens as al

from autogalaxy.fixtures import *


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


def make_point_solver():
    grid = al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.5)
    return al.PointSolver(grid=grid, pixel_scale_precision=0.25)


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


def make_tracer_x2_plane_voronoi_7x7():

    source_gal_inversion = al.Galaxy(
        redshift=1.0,
        pixelization=al.pix.VoronoiMagnification(),
        regularization=al.reg.Constant(),
    )

    return al.Tracer.from_galaxies(
        galaxies=[make_gal_x1_mp(), make_gal_x1_lp(), source_gal_inversion]
    )


def make_tracer_x2_plane_point():

    source_gal_x1_lp = al.Galaxy(redshift=1.0, point_0=al.ps.PointFlux())

    return al.Tracer.from_galaxies(
        galaxies=[make_gal_x1_mp(), make_gal_x1_lp(), source_gal_x1_lp]
    )


def make_fit_imaging_x1_plane_7x7():
    return al.FitImaging(
        dataset=make_masked_imaging_7x7(), tracer=make_tracer_x1_plane_7x7()
    )


def make_fit_imaging_x2_plane_7x7():
    return al.FitImaging(
        dataset=make_masked_imaging_7x7(), tracer=make_tracer_x2_plane_7x7()
    )


def make_fit_imaging_x2_plane_inversion_7x7():
    return al.FitImaging(
        dataset=make_masked_imaging_7x7(), tracer=make_tracer_x2_plane_inversion_7x7()
    )


def make_fit_interferometer_x1_plane_7x7():
    return al.FitInterferometer(
        dataset=make_interferometer_7(),
        tracer=make_tracer_x1_plane_7x7(),
        settings_inversion=aa.SettingsInversion(use_w_tilde=False),
    )


def make_fit_interferometer_x2_plane_7x7():
    return al.FitInterferometer(
        dataset=make_interferometer_7(),
        tracer=make_tracer_x2_plane_7x7(),
        settings_inversion=aa.SettingsInversion(use_w_tilde=False),
    )


def make_fit_interferometer_x2_plane_inversion_7x7():
    return al.FitInterferometer(
        dataset=make_interferometer_7(),
        tracer=make_tracer_x2_plane_inversion_7x7(),
        settings_inversion=aa.SettingsInversion(use_w_tilde=False),
    )


def make_fit_point_dataset_x2_plane():
    return al.FitPointDataset(
        point_dataset=make_point_dataset(),
        tracer=make_tracer_x2_plane_point(),
        point_solver=make_point_solver(),
    )


def make_fit_point_dict_x2_plane():
    return al.FitPointDict(
        point_dict=make_point_dict(),
        tracer=make_tracer_x2_plane_point(),
        point_solver=make_point_solver(),
    )


def make_analysis_imaging_7x7():
    return al.AnalysisImaging(
        dataset=make_masked_imaging_7x7(),
        settings_inversion=aa.SettingsInversion(use_w_tilde=False),
    )


def make_analysis_interferometer_7():
    return al.AnalysisInterferometer(dataset=make_interferometer_7())


def make_analysis_point_x2():
    return al.AnalysisPoint(
        point_dict=make_point_dict(),
        solver=al.m.MockPointSolver(model_positions=make_positions_x2()),
    )

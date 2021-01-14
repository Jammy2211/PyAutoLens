from os import path
import autolens.plot as aplt
import pytest


@pytest.fixture(name="plot_path")
def make_fit_imaging_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test__fit_quantities_are_output(
    masked_interferometer_fit_x2_plane_7x7, plot_path, plot_patch
):

    fit_interferometer_plotter = aplt.FitInterferometerPlotter(
        fit=masked_interferometer_fit_x2_plane_7x7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(path=plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_interferometer_plotter.figures(
        visibilities=True,
        noise_map=True,
        signal_to_noise_map=True,
        model_visibilities=True,
        residual_map_real=True,
        residual_map_imag=True,
        normalized_residual_map_real=True,
        normalized_residual_map_imag=True,
        chi_squared_map_real=True,
        chi_squared_map_imag=True,
        image=True,
    )

    assert path.join(plot_path, "visibilities.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "model_visibilities.png") in plot_patch.paths
    assert (
        path.join(plot_path, "real_residual_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "real_residual_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "real_normalized_residual_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "imag_residual_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "imag_residual_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "imag_normalized_residual_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert path.join(plot_path, "image.png") in plot_patch.paths

    plot_patch.paths = []

    fit_interferometer_plotter.figures(
        visibilities=True,
        noise_map=False,
        signal_to_noise_map=False,
        model_visibilities=True,
        chi_squared_map_real=True,
        chi_squared_map_imag=True,
    )

    assert path.join(plot_path, "visibilities.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "model_visibilities.png") in plot_patch.paths
    assert (
        path.join(plot_path, "real_residual_map_vs_uv_distances.png")
        not in plot_patch.paths
    )
    assert (
        path.join(plot_path, "real_normalized_residual_map_vs_uv_distances.png")
        not in plot_patch.paths
    )
    assert (
        path.join(plot_path, "real_chi_squared_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "imag_residual_map_vs_uv_distances.png")
        not in plot_patch.paths
    )
    assert (
        path.join(plot_path, "imag_normalized_residual_map_vs_uv_distances.png")
        not in plot_patch.paths
    )
    assert (
        path.join(plot_path, "imag_chi_squared_map_vs_uv_distances.png")
        in plot_patch.paths
    )


def test__fit_sub_plot_real_space(
    masked_interferometer_fit_x2_plane_7x7,
    masked_interferometer_fit_x2_plane_inversion_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):

    fit_interferometer_plotter = aplt.FitInterferometerPlotter(
        fit=masked_interferometer_fit_x2_plane_7x7,
        include_2d=include_2d_all,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_interferometer_plotter.subplot_fit_real_space()
    assert path.join(plot_path, "subplot_fit_real_space.png") in plot_patch.paths

    plot_patch.paths = []

    fit_interferometer_plotter = aplt.FitInterferometerPlotter(
        fit=masked_interferometer_fit_x2_plane_inversion_7x7,
        include_2d=include_2d_all,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_interferometer_plotter.subplot_fit_real_space()
    assert path.join(plot_path, "subplot_fit_real_space.png") in plot_patch.paths

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

    aplt.FitInterferometer.visibilities(
        fit=masked_interferometer_fit_x2_plane_7x7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "visibilities.png") in plot_patch.paths

    aplt.FitInterferometer.noise_map(
        fit=masked_interferometer_fit_x2_plane_7x7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "noise_map.png") in plot_patch.paths

    aplt.FitInterferometer.signal_to_noise_map(
        fit=masked_interferometer_fit_x2_plane_7x7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths

    aplt.FitInterferometer.model_visibilities(
        fit=masked_interferometer_fit_x2_plane_7x7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "model_visibilities.png") in plot_patch.paths

    aplt.FitInterferometer.residual_map_vs_uv_distances(
        fit=masked_interferometer_fit_x2_plane_7x7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert (
        path.join(plot_path, "residual_map_vs_uv_distances_real.png")
        in plot_patch.paths
    )

    aplt.FitInterferometer.residual_map_vs_uv_distances(
        fit=masked_interferometer_fit_x2_plane_7x7,
        plot_real=False,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert (
        path.join(plot_path, "residual_map_vs_uv_distances_imag.png")
        in plot_patch.paths
    )

    aplt.FitInterferometer.normalized_residual_map_vs_uv_distances(
        fit=masked_interferometer_fit_x2_plane_7x7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert (
        path.join(plot_path, "normalized_residual_map_vs_uv_distances_real.png")
        in plot_patch.paths
    )

    aplt.FitInterferometer.normalized_residual_map_vs_uv_distances(
        fit=masked_interferometer_fit_x2_plane_7x7,
        plot_real=False,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert (
        path.join(plot_path, "normalized_residual_map_vs_uv_distances_imag.png")
        in plot_patch.paths
    )

    aplt.FitInterferometer.chi_squared_map_vs_uv_distances(
        fit=masked_interferometer_fit_x2_plane_7x7,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert (
        path.join(plot_path, "chi_squared_map_vs_uv_distances_real.png")
        in plot_patch.paths
    )

    aplt.FitInterferometer.chi_squared_map_vs_uv_distances(
        fit=masked_interferometer_fit_x2_plane_7x7,
        plot_real=False,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert (
        path.join(plot_path, "chi_squared_map_vs_uv_distances_imag.png")
        in plot_patch.paths
    )


def test__fit_sub_plot(
    masked_interferometer_fit_x2_plane_7x7, include_all, plot_path, plot_patch
):

    aplt.FitInterferometer.subplot_fit_interferometer(
        fit=masked_interferometer_fit_x2_plane_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert (
        path.join(plot_path, "subplot_fit_interferometer_real.png") in plot_patch.paths
    )


def test__fit_sub_plot_real_space(
    masked_interferometer_fit_x2_plane_7x7, include_all, plot_path, plot_patch
):

    aplt.FitInterferometer.subplot_fit_real_space(
        fit=masked_interferometer_fit_x2_plane_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_fit_real_space.png") in plot_patch.paths


def test__fit_individuals__source_and_lens__depedent_on_input(
    masked_interferometer_fit_x1_plane_7x7,
    masked_interferometer_fit_x2_plane_7x7,
    include_all,
    plot_path,
    plot_patch,
):

    aplt.FitInterferometer.individuals(
        fit=masked_interferometer_fit_x1_plane_7x7,
        plot_visibilities=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_visibilities=True,
        plot_chi_squared_map=True,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "visibilities.png") in plot_patch.paths

    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths

    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths

    assert path.join(plot_path, "model_visibilities.png") in plot_patch.paths

    assert (
        path.join(plot_path, "residual_map_vs_uv_distances_real.png")
        not in plot_patch.paths
    )

    assert (
        path.join(plot_path, "normalized_residual_map_vs_uv_distances_real.png")
        not in plot_patch.paths
    )

    assert (
        path.join(plot_path, "chi_squared_map_vs_uv_distances_real.png")
        in plot_patch.paths
    )

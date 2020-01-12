import pytest
import os

import autolens as al


@pytest.fixture(name="fit_interferometer_plotter_path")
def make_fit_interferometer_plotter_setup():
    return "{}/../../test_files/plotting/fit/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__fit_sub_plot(
    masked_interferometer_fit_x2_plane_7x7, fit_interferometer_plotter_path, plot_patch
):

    al.plot.fit_interferometer.subplot_fit_interferometer(
        fit=masked_interferometer_fit_x2_plane_7x7,
        sub_plotter=al.plotter.SubPlotter(
            output=al.plotter.Output(path=fit_interferometer_plotter_path, format="png")
        ),
    )

    assert fit_interferometer_plotter_path + "subplot_fit_interferometer.png" in plot_patch.paths


def test__fit_sub_plot_real_space(
    masked_interferometer_fit_x2_plane_7x7, fit_interferometer_plotter_path, plot_patch
):

    al.plot.fit_interferometer.subplot_fit_real_space(
        fit=masked_interferometer_fit_x2_plane_7x7,
        sub_plotter=al.plotter.SubPlotter(
            output=al.plotter.Output(path=fit_interferometer_plotter_path, format="png")
        ),
    )

    assert fit_interferometer_plotter_path + "subplot_fit_real_space.png" in plot_patch.paths


def test__fit_individuals__source_and_lens__depedent_on_input(
    masked_interferometer_fit_x1_plane_7x7,
    masked_interferometer_fit_x2_plane_7x7,
    fit_interferometer_plotter_path,
    plot_patch,
):

    al.plot.fit_interferometer.individuals(
        fit=masked_interferometer_fit_x1_plane_7x7,
        plot_visibilities=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_visibilities=True,
        plot_chi_squared_map=True,
        plotter=al.plotter.Plotter(
            output=al.plotter.Output(path=fit_interferometer_plotter_path, format="png")
        )
    )

    assert fit_interferometer_plotter_path + "visibilities.png" in plot_patch.paths

    assert fit_interferometer_plotter_path + "noise_map.png" not in plot_patch.paths

    assert (
        fit_interferometer_plotter_path + "signal_to_noise_map.png"
        not in plot_patch.paths
    )

    assert (
        fit_interferometer_plotter_path + "model_visibilities.png"
        in plot_patch.paths
    )

    assert (
        fit_interferometer_plotter_path + "residual_map_vs_uv_distances_real.png"
        not in plot_patch.paths
    )

    assert (
        fit_interferometer_plotter_path
        + "normalized_residual_map_vs_uv_distances_real.png"
        not in plot_patch.paths
    )

    assert (
        fit_interferometer_plotter_path + "chi_squared_map_vs_uv_distances_real.png"
        in plot_patch.paths
    )

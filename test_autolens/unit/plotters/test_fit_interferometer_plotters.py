import pytest
import os

import autolens as al


@pytest.fixture(name="interferometer_fit_plotter_path")
def make_interferometer_fit_plotter_setup():
    return "{}/../../test_files/plotting/fit/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__fit_sub_plot(
    masked_interferometer_fit_x2_plane_7x7, interferometer_fit_plotter_path, plot_patch
):

    al.plot.fit_interferometer.subplot(
        fit=masked_interferometer_fit_x2_plane_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=interferometer_fit_plotter_path,
        output_format="png",
    )

    assert interferometer_fit_plotter_path + "fit.png" in plot_patch.paths


def test__fit_sub_plot_real_space(
    masked_interferometer_fit_x2_plane_7x7, interferometer_fit_plotter_path, plot_patch
):

    al.plot.fit_interferometer.subplot_real_space(
        fit=masked_interferometer_fit_x2_plane_7x7,
        include_mask=True,
        include_critical_curves=True,
        include_caustics=True,
        include_image_plane_pix=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=interferometer_fit_plotter_path,
        output_format="png",
    )

    assert interferometer_fit_plotter_path + "fit_real_space.png" in plot_patch.paths


def test__fit_individuals__source_and_lens__depedent_on_input(
    masked_interferometer_fit_x1_plane_7x7,
    masked_interferometer_fit_x2_plane_7x7,
    interferometer_fit_plotter_path,
    plot_patch,
):

    al.plot.fit_interferometer.individuals(
        fit=masked_interferometer_fit_x1_plane_7x7,
        plot_visibilities=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_visibilities=True,
        plot_chi_squared_map=True,
        output_path=interferometer_fit_plotter_path,
        output_format="png",
    )

    assert interferometer_fit_plotter_path + "fit_visibilities.png" in plot_patch.paths

    assert interferometer_fit_plotter_path + "fit_noise_map.png" not in plot_patch.paths

    assert (
        interferometer_fit_plotter_path + "fit_signal_to_noise_map.png"
        not in plot_patch.paths
    )

    assert (
        interferometer_fit_plotter_path + "fit_model_visibilities.png"
        in plot_patch.paths
    )

    assert (
        interferometer_fit_plotter_path + "fit_residual_map_vs_uv_distances_real.png"
        not in plot_patch.paths
    )

    assert (
        interferometer_fit_plotter_path
        + "fit_normalized_residual_map_vs_uv_distances_real.png"
        not in plot_patch.paths
    )

    assert (
        interferometer_fit_plotter_path + "fit_chi_squared_map_vs_uv_distances_real.png"
        in plot_patch.paths
    )

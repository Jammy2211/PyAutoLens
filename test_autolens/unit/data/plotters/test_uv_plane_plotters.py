import autolens as al
import os

import pytest


@pytest.fixture(name="uv_plane_plotter_path")
def make_uv_plane_plotter_setup():
    uv_plane_plotter_path = "{}/../../test_files/plotting/uv_plane/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    return uv_plane_plotter_path


def test__individual_attributes_are_output(
    uv_plane_data_7, uv_plane_plotter_path, plot_patch
):
    al.uv_plane_plotters.plot_visibilities(
        uv_plane_data=uv_plane_data_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=uv_plane_plotter_path,
        output_format="png",
    )

    assert uv_plane_plotter_path + "uv_plane_visibilities.png" in plot_patch.paths

    al.uv_plane_plotters.plot_u_wavelengths(
        uv_plane_data=uv_plane_data_7,
        output_path=uv_plane_plotter_path,
        output_format="png",
    )

    assert uv_plane_plotter_path + "uv_plane_u_wavelengths.png" in plot_patch.paths

    al.uv_plane_plotters.plot_v_wavelengths(
        uv_plane_data=uv_plane_data_7,
        output_path=uv_plane_plotter_path,
        output_format="png",
    )

    assert uv_plane_plotter_path + "uv_plane_v_wavelengths.png" in plot_patch.paths

    al.uv_plane_plotters.plot_primary_beam(
        uv_plane_data=uv_plane_data_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=uv_plane_plotter_path,
        output_format="png",
    )

    assert uv_plane_plotter_path + "uv_plane_primary_beam.png" in plot_patch.paths

    al.uv_plane_plotters.plot_uv_plane_subplot(
        uv_plane_data=uv_plane_data_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=uv_plane_plotter_path,
        output_format="png",
    )

    assert uv_plane_plotter_path + "uv_plane_data.png" in plot_patch.paths


def test__uv_plane_individuals__output_dependent_on_input(
    uv_plane_data_7, general_config, uv_plane_plotter_path, plot_patch
):
    al.uv_plane_plotters.plot_uv_plane_individual(
        uv_plane_data=uv_plane_data_7,
        should_plot_visibilities=True,
        should_plot_u_wavelengths=False,
        should_plot_v_wavelengths=True,
        should_plot_primary_beam=True,
        output_path=uv_plane_plotter_path,
        output_format="png",
    )

    assert uv_plane_plotter_path + "uv_plane_visibilities.png" in plot_patch.paths

    assert not uv_plane_plotter_path + "uv_plane_u_wavelengths.png" in plot_patch.paths

    assert uv_plane_plotter_path + "uv_plane_v_wavelengths.png" in plot_patch.paths

    assert uv_plane_plotter_path + "uv_plane_primary_beam.png" in plot_patch.paths

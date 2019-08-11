import os

import pytest

from autolens.data.plotters import interferometer_plotters


@pytest.fixture(name="interferometer_plotter_path")
def make_interferometer_plotter_setup():
    interferometer_plotter_path = "{}/../../test_files/plotting/interferometer/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    return interferometer_plotter_path


def test__individual_attributes_are_output(
    interferometer_data_7, interferometer_plotter_path, plot_patch
):
    interferometer_plotters.plot_visibilities(
        interferometer_data=interferometer_data_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=interferometer_plotter_path,
        output_format="png",
    )

    assert interferometer_plotter_path + "interferometer_visibilities.png" in plot_patch.paths

    interferometer_plotters.plot_u_wavelengths(
        interferometer_data=interferometer_data_7,
        output_path=interferometer_plotter_path,
        output_format="png",
    )

    assert interferometer_plotter_path + "interferometer_u_wavelengths.png" in plot_patch.paths

    interferometer_plotters.plot_v_wavelengths(
        interferometer_data=interferometer_data_7,
        output_path=interferometer_plotter_path,
        output_format="png",
    )

    assert interferometer_plotter_path + "interferometer_v_wavelengths.png" in plot_patch.paths

    interferometer_plotters.plot_primary_beam(
        interferometer_data=interferometer_data_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=interferometer_plotter_path,
        output_format="png",
    )

    assert interferometer_plotter_path + "interferometer_primary_beam.png" in plot_patch.paths
    
    interferometer_plotters.plot_interferometer_subplot(
        interferometer_data=interferometer_data_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=interferometer_plotter_path,
        output_format="png",
    )

    assert interferometer_plotter_path + "interferometer_data.png" in plot_patch.paths

def test__interferometer_individuals__output_dependent_on_input(
    interferometer_data_7, general_config, interferometer_plotter_path, plot_patch
):
    interferometer_plotters.plot_interferometer_individual(
        interferometer_data=interferometer_data_7,
        should_plot_visibilities=True,
        should_plot_u_wavelengths=False,
        should_plot_v_wavelengths=True,
        should_plot_primary_beam=True,
        output_path=interferometer_plotter_path,
        output_format="png",
    )

    assert interferometer_plotter_path + "interferometer_visibilities.png" in plot_patch.paths

    assert not interferometer_plotter_path + "interferometer_u_wavelengths.png" in plot_patch.paths

    assert interferometer_plotter_path + "interferometer_v_wavelengths.png" in plot_patch.paths

    assert interferometer_plotter_path + "interferometer_primary_beam.png" in plot_patch.paths
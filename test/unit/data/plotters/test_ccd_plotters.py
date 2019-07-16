import os

import pytest

from autolens.data.plotters import ccd_plotters


@pytest.fixture(name="ccd_plotter_path")
def make_ccd_plotter_setup():
    ccd_plotter_path = "{}/../../test_files/plotting/ccd/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    return ccd_plotter_path


def test__individual_attributes_are_output(
    ccd_data_7x7, positions_7x7, mask_7x7, ccd_plotter_path, plot_patch
):
    ccd_plotters.plot_image(
        ccd_data=ccd_data_7x7,
        positions=positions_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=ccd_plotter_path,
        output_format="png",
    )

    assert ccd_plotter_path + "ccd_image.png" in plot_patch.paths

    ccd_plotters.plot_noise_map(
        ccd_data=ccd_data_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=ccd_plotter_path,
        output_format="png",
    )

    assert ccd_plotter_path + "ccd_noise_map.png" in plot_patch.paths

    ccd_plotters.plot_psf(
        ccd_data=ccd_data_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=ccd_plotter_path,
        output_format="png",
    )

    assert ccd_plotter_path + "ccd_psf.png" in plot_patch.paths

    ccd_plotters.plot_signal_to_noise_map(
        ccd_data=ccd_data_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=ccd_plotter_path,
        output_format="png",
    )

    assert ccd_plotter_path + "ccd_signal_to_noise_map.png" in plot_patch.paths

    ccd_plotters.plot_ccd_subplot(
        ccd_data=ccd_data_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=ccd_plotter_path,
        output_format="png",
    )

    assert ccd_plotter_path + "ccd_data.png" in plot_patch.paths


def test__ccd_individuals__output_dependent_on_input(
    ccd_data_7x7, general_config, ccd_plotter_path, plot_patch
):
    ccd_plotters.plot_ccd_individual(
        ccd_data=ccd_data_7x7,
        should_plot_image=True,
        should_plot_psf=True,
        should_plot_absolute_signal_to_noise_map=True,
        output_path=ccd_plotter_path,
        output_format="png",
    )

    assert ccd_plotter_path + "ccd_image.png" in plot_patch.paths

    assert not ccd_plotter_path + "ccd_noise_map.png" in plot_patch.paths

    assert ccd_plotter_path + "ccd_psf.png" in plot_patch.paths

    assert not ccd_plotter_path + "ccd_signal_to_noise_map.png" in plot_patch.paths

    assert ccd_plotter_path + "ccd_absolute_signal_to_noise_map.png" in plot_patch.paths

    assert (
        not ccd_plotter_path + "ccd_potential_chi_squared_map.png" in plot_patch.paths
    )

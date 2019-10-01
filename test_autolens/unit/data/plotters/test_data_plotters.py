import autolens as al
import os

import pytest


@pytest.fixture(name="data_plotter_path")
def make_data_plotter_setup():
    data_plotter_path = "{}/../../test_files/plotting/data/".format(
        os.path.dirname(os.path.realpath(__file__))
    )
    return data_plotter_path


def test__all_imaging_data_types_are_output(
    image_7x7,
    noise_map_7x7,
    psf_3x3,
    positions_7x7,
    mask_7x7,
    data_plotter_path,
    plot_patch,
):
    al.data_plotters.plot_image(
        image=image_7x7,
        positions=positions_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "image.png" in plot_patch.paths

    al.data_plotters.plot_noise_map(
        noise_map=noise_map_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "noise_map.png" in plot_patch.paths

    al.data_plotters.plot_psf(
        psf=psf_3x3,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "psf.png" in plot_patch.paths

    al.data_plotters.plot_signal_to_noise_map(
        signal_to_noise_map=image_7x7 / noise_map_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "signal_to_noise_map.png" in plot_patch.paths

    al.data_plotters.plot_absolute_signal_to_noise_map(
        absolute_signal_to_noise_map=image_7x7 / noise_map_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "absolute_signal_to_noise_map.png" in plot_patch.paths

    al.data_plotters.plot_potential_chi_squared_map(
        potential_chi_squared_map=image_7x7 / noise_map_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "potential_chi_squared_map.png" in plot_patch.paths


def test__all_uv_plane_data_types_are_output(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7,
    primary_beam_3x3,
    data_plotter_path,
    plot_patch,
):
    al.data_plotters.plot_visibilities(
        visibilities=visibilities_7,
        noise_map=visibilities_noise_map_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "visibilities.png" in plot_patch.paths

    al.data_plotters.plot_u_wavelengths(
        uv_wavelengths=uv_wavelengths_7,
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "u_wavelengths.png" in plot_patch.paths

    al.data_plotters.plot_v_wavelengths(
        uv_wavelengths=uv_wavelengths_7,
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "v_wavelengths.png" in plot_patch.paths

    al.data_plotters.plot_primary_beam(
        primary_beam=primary_beam_3x3,
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "primary_beam.png" in plot_patch.paths

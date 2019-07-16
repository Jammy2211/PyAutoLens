import os

import pytest

from autolens.pipeline.plotters import hyper_plotters


@pytest.fixture(name="hyper_plotter_path")
def make_hyper_plotter_setup():
    return "{}/../../test_files/plotting/hyper/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__plot_hyper_model_image(hyper_model_image_7x7, hyper_plotter_path, plot_patch):
    hyper_plotters.plot_hyper_model_image(
        hyper_model_image=hyper_model_image_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "hyper_model_image.png" in plot_patch.paths


def test__plot_hyper_galaxy_image(
    hyper_galaxy_image_0_7x7, hyper_plotter_path, plot_patch
):
    hyper_plotters.plot_hyper_galaxy_image(
        hyper_galaxy_image=hyper_galaxy_image_0_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "hyper_galaxy_image.png" in plot_patch.paths


def test__plot_contribution_map(contribution_map_7x7, hyper_plotter_path, plot_patch):
    hyper_plotters.plot_contribution_map(
        contribution_map=contribution_map_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "contribution_map.png" in plot_patch.paths


def test__plot_hyper_noise_map(hyper_noise_map_7x7, hyper_plotter_path, plot_patch):
    hyper_plotters.plot_hyper_noise_map(
        hyper_noise_map=hyper_noise_map_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "hyper_noise_map.png" in plot_patch.paths


def test__plot_chi_squared_map(lens_fit_x1_plane_7x7, hyper_plotter_path, plot_patch):
    hyper_plotters.plot_chi_squared_map(
        chi_squared_map=lens_fit_x1_plane_7x7.chi_squared_map(return_in_2d=True),
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "chi_squared_map.png" in plot_patch.paths


def test__plot_hyper_chi_squared_map(
    lens_fit_x1_plane_7x7, hyper_plotter_path, plot_patch
):
    hyper_plotters.plot_hyper_chi_squared_map(
        hyper_chi_squared_map=lens_fit_x1_plane_7x7.chi_squared_map(return_in_2d=True),
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "hyper_chi_squared_map.png" in plot_patch.paths


def test__plot_hyper_galaxy(
    hyper_galaxy_image_0_7x7,
    contribution_map_7x7,
    noise_map_7x7,
    hyper_noise_map_7x7,
    lens_fit_x1_plane_7x7,
    hyper_plotter_path,
    plot_patch,
):
    hyper_plotters.plot_hyper_galaxy_subplot(
        hyper_galaxy_image=hyper_galaxy_image_0_7x7,
        contribution_map=contribution_map_7x7,
        noise_map=noise_map_7x7,
        hyper_noise_map=hyper_noise_map_7x7,
        chi_squared_map=lens_fit_x1_plane_7x7.chi_squared_map(return_in_2d=True),
        hyper_chi_squared_map=lens_fit_x1_plane_7x7.chi_squared_map(return_in_2d=True),
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "hyper_galaxy.png" in plot_patch.paths


def test__plot_hyper_galaxy_images(
    hyper_galaxy_image_0_7x7,
    hyper_galaxy_image_1_7x7,
    mask_7x7,
    hyper_plotter_path,
    plot_patch,
):
    hyper_galaxy_image_path_dict = {}

    hyper_galaxy_image_path_dict[("g0",)] = hyper_galaxy_image_0_7x7
    hyper_galaxy_image_path_dict[("g1",)] = hyper_galaxy_image_1_7x7

    hyper_plotters.plot_hyper_galaxy_images_subplot(
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
        mask=mask_7x7,
        should_plot_mask=True,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "hyper_galaxy_images.png" in plot_patch.paths

    hyper_plotters.plot_hyper_galaxy_cluster_images_subplot(
        hyper_galaxy_cluster_image_path_dict=hyper_galaxy_image_path_dict,
        mask=mask_7x7,
        should_plot_mask=True,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "hyper_galaxy_cluster_images.png" in plot_patch.paths

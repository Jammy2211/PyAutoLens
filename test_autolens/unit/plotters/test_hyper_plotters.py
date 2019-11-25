import autolens as al
import os

import pytest


@pytest.fixture(name="hyper_plotter_path")
def make_hyper_plotter_setup():
    return "{}/../../test_files/plotting/hyper_galaxies/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__plot_individual_images(
    hyper_model_image_7x7,
    hyper_galaxy_image_0_7x7,
    contribution_map_7x7,
    hyper_noise_map_7x7,
    masked_imaging_fit_x1_plane_7x7,
    hyper_plotter_path,
    plot_patch,
):
    al.plot.hyper.hyper_model_image(
        hyper_model_image=hyper_model_image_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "hyper_model_image.png" in plot_patch.paths

    al.plot.hyper.hyper_galaxy_image(
        hyper_galaxy_image=hyper_galaxy_image_0_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "hyper_galaxy_image.png" in plot_patch.paths

    al.plot.hyper.contribution_map(
        contribution_map=contribution_map_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "contribution_map.png" in plot_patch.paths

    al.plot.hyper.hyper_noise_map(
        hyper_noise_map=hyper_noise_map_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "hyper_noise_map.png" in plot_patch.paths

    al.plot.hyper.chi_squared_map(
        chi_squared_map=masked_imaging_fit_x1_plane_7x7.chi_squared_map,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "chi_squared_map.png" in plot_patch.paths

    al.plot.hyper.hyper_chi_squared_map(
        hyper_chi_squared_map=masked_imaging_fit_x1_plane_7x7.chi_squared_map,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "hyper_chi_squared_map.png" in plot_patch.paths


def test__plot_subplot_of_hyper_galaxy(
    hyper_galaxy_image_0_7x7,
    contribution_map_7x7,
    noise_map_7x7,
    hyper_noise_map_7x7,
    masked_imaging_fit_x1_plane_7x7,
    hyper_plotter_path,
    plot_patch,
):
    al.plot.hyper.subplot_of_hyper_galaxy(
        hyper_galaxy_image_sub=hyper_galaxy_image_0_7x7,
        contribution_map_sub=contribution_map_7x7,
        noise_map_sub=noise_map_7x7,
        hyper_noise_map_sub=hyper_noise_map_7x7,
        chi_squared_map_sub=masked_imaging_fit_x1_plane_7x7.chi_squared_map,
        hyper_chi_squared_map_sub=masked_imaging_fit_x1_plane_7x7.chi_squared_map,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "hyper_galaxies.png" in plot_patch.paths


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

    al.plot.hyper.subplot_of_hyper_galaxy_images(
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
        mask=mask_7x7,
        include_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=hyper_plotter_path,
        output_format="png",
    )

    assert hyper_plotter_path + "hyper_galaxy_images.png" in plot_patch.paths

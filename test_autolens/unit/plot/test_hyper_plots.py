import autolens.plot as aplt
import os

import pytest


@pytest.fixture(name="plot_path")
def make_hyper_plotter_setup():
    return "{}/files/plots/hyper_galaxies/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__plot_individual_images(
    hyper_galaxy_image_0_7x7, contribution_map_7x7, include_all, plot_path, plot_patch
):

    aplt.hyper.hyper_galaxy_image(
        galaxy_image=hyper_galaxy_image_0_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "hyper_galaxy_image.png" in plot_patch.paths

    aplt.hyper.contribution_map(
        contribution_map_in=contribution_map_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "contribution_map.png" in plot_patch.paths


def test__plot_subplot_of_hyper_galaxy(
    hyper_galaxy_image_0_7x7,
    contribution_map_7x7,
    masked_imaging_fit_x2_plane_7x7,
    include_all,
    plot_path,
    plot_patch,
):
    aplt.hyper.subplot_fit_hyper_galaxy(
        fit=masked_imaging_fit_x2_plane_7x7,
        hyper_fit=masked_imaging_fit_x2_plane_7x7,
        galaxy_image=hyper_galaxy_image_0_7x7,
        contribution_map_in=contribution_map_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "subplot_fit_hyper_galaxy.png" in plot_patch.paths


def test__plot_hyper_galaxy_images(
    hyper_galaxy_image_path_dict_7x7, mask_7x7, include_all, plot_path, plot_patch
):

    aplt.hyper.subplot_hyper_galaxy_images(
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict_7x7,
        mask=mask_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "subplot_hyper_galaxy_images.png" in plot_patch.paths

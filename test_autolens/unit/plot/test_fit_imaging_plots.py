import pytest
import os

import autolens.plot as aplt

from os import path

from autofit import conf

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="fit_imaging_plotter_path")
def make_fit_imaging_plotter_setup():
    return "{}/../../test_files/plotting/fit/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plot"), path.join(directory, "output")
    )


def test__subtracted_image_of_plane_is_output(
    masked_imaging_fit_x1_plane_7x7,
    masked_imaging_fit_x2_plane_7x7,
        include_all,
    fit_imaging_plotter_path,
    plot_patch,
):

    aplt.fit_imaging.subtracted_image_of_plane(
        fit=masked_imaging_fit_x1_plane_7x7,
        plane_index=0,
        include=include_all,
        plotter=aplt.Plotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert (
        fit_imaging_plotter_path + "subtracted_image_of_plane_0.png" in plot_patch.paths
    )

    aplt.fit_imaging.subtracted_image_of_plane(
        fit=masked_imaging_fit_x2_plane_7x7,
        plane_index=0,
        include=include_all,
        plotter=aplt.Plotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert (
        fit_imaging_plotter_path + "subtracted_image_of_plane_0.png" in plot_patch.paths
    )

    aplt.fit_imaging.subtracted_image_of_plane(
        fit=masked_imaging_fit_x2_plane_7x7,
        plane_index=1,
        include=include_all,
        plotter=aplt.Plotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert (
        fit_imaging_plotter_path + "subtracted_image_of_plane_1.png" in plot_patch.paths
    )


def test__model_image_of_plane_is_output(
    masked_imaging_fit_x1_plane_7x7,
    masked_imaging_fit_x2_plane_7x7,
        include_all,
    fit_imaging_plotter_path,
    plot_patch,
):

    aplt.fit_imaging.model_image_of_plane(
        fit=masked_imaging_fit_x1_plane_7x7,
        plane_index=0,
        include=include_all,
        plotter=aplt.Plotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert fit_imaging_plotter_path + "model_image_of_plane_0.png" in plot_patch.paths

    aplt.fit_imaging.model_image_of_plane(
        fit=masked_imaging_fit_x2_plane_7x7,
        plane_index=0,
        include=include_all,
        plotter=aplt.Plotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert fit_imaging_plotter_path + "model_image_of_plane_0.png" in plot_patch.paths

    aplt.fit_imaging.model_image_of_plane(
        fit=masked_imaging_fit_x2_plane_7x7,
        plane_index=1,
        include=include_all,
        plotter=aplt.Plotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert fit_imaging_plotter_path + "model_image_of_plane_1.png" in plot_patch.paths


def test_subplot_fit_imaging_is_output(
    masked_imaging_fit_x2_plane_7x7, include_all, fit_imaging_plotter_path, plot_patch
):

    aplt.fit_imaging.subplot_fit_imaging(
        fit=masked_imaging_fit_x2_plane_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert fit_imaging_plotter_path + "subplot_fit_imaging.png" in plot_patch.paths


def test__subplot_of_plane(
    masked_imaging_fit_x1_plane_7x7,
    masked_imaging_fit_x2_plane_7x7,
    include_all,
    fit_imaging_plotter_path,
    plot_patch,
):

    aplt.fit_imaging.subplot_of_plane(
        fit=masked_imaging_fit_x1_plane_7x7,
        plane_index=0,
        include=include_all,
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert fit_imaging_plotter_path + "subplot_of_plane_0.png" in plot_patch.paths

    aplt.fit_imaging.subplot_of_plane(
        fit=masked_imaging_fit_x2_plane_7x7,
        plane_index=0,
        include=include_all,
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert fit_imaging_plotter_path + "subplot_of_plane_0.png" in plot_patch.paths

    aplt.fit_imaging.subplot_of_plane(
        fit=masked_imaging_fit_x2_plane_7x7,
        plane_index=1,
        include=include_all,
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert fit_imaging_plotter_path + "subplot_of_plane_1.png" in plot_patch.paths


    aplt.fit_imaging.subplot_of_planes(
        fit=masked_imaging_fit_x1_plane_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert fit_imaging_plotter_path + "subplot_of_plane_0.png" in plot_patch.paths

    aplt.fit_imaging.subplot_of_planes(
        fit=masked_imaging_fit_x2_plane_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert fit_imaging_plotter_path + "subplot_of_plane_0.png" in plot_patch.paths
    assert fit_imaging_plotter_path + "subplot_of_plane_1.png" in plot_patch.paths


def test__fit_individuals__source_and_lens__dependent_on_input(
    masked_imaging_fit_x1_plane_7x7,
    masked_imaging_fit_x2_plane_7x7,
        include_all,
    fit_imaging_plotter_path,
    plot_patch,
):

    aplt.fit_imaging.individuals(
        fit=masked_imaging_fit_x1_plane_7x7,
        plot_image=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_image=True,
        plot_chi_squared_map=True,
        plot_subtracted_images_of_planes=True,
        plot_model_images_of_planes=True,
        plot_plane_images_of_planes=True,
        include=include_all,
        plotter=aplt.Plotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert fit_imaging_plotter_path + "image.png" in plot_patch.paths

    assert fit_imaging_plotter_path + "noise_map.png" not in plot_patch.paths

    assert fit_imaging_plotter_path + "signal_to_noise_map.png" not in plot_patch.paths

    assert fit_imaging_plotter_path + "model_image.png" in plot_patch.paths

    assert fit_imaging_plotter_path + "residual_map.png" not in plot_patch.paths

    assert (
        fit_imaging_plotter_path + "normalized_residual_map.png" not in plot_patch.paths
    )

    assert fit_imaging_plotter_path + "chi_squared_map.png" in plot_patch.paths

    assert (
        fit_imaging_plotter_path + "subtracted_image_of_plane_0.png" in plot_patch.paths
    )

    assert fit_imaging_plotter_path + "model_image_of_plane_0.png" in plot_patch.paths

    assert fit_imaging_plotter_path + "plane_image_of_plane_0.png" in plot_patch.paths

    aplt.fit_imaging.individuals(
        fit=masked_imaging_fit_x2_plane_7x7,
        plot_image=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_image=True,
        plot_chi_squared_map=True,
        plot_subtracted_images_of_planes=True,
        plot_model_images_of_planes=True,
        plot_plane_images_of_planes=True,
        include=include_all,
        plotter=aplt.Plotter(
            output=aplt.Output(fit_imaging_plotter_path, format="png")
        ),
    )

    assert fit_imaging_plotter_path + "image.png" in plot_patch.paths

    assert fit_imaging_plotter_path + "noise_map.png" not in plot_patch.paths

    assert fit_imaging_plotter_path + "signal_to_noise_map.png" not in plot_patch.paths

    assert fit_imaging_plotter_path + "model_image.png" in plot_patch.paths

    assert fit_imaging_plotter_path + "residual_map.png" not in plot_patch.paths

    assert (
        fit_imaging_plotter_path + "normalized_residual_map.png" not in plot_patch.paths
    )

    assert fit_imaging_plotter_path + "chi_squared_map.png" in plot_patch.paths

    assert (
        fit_imaging_plotter_path + "subtracted_image_of_plane_0.png" in plot_patch.paths
    )
    assert (
        fit_imaging_plotter_path + "subtracted_image_of_plane_1.png" in plot_patch.paths
    )

    assert fit_imaging_plotter_path + "model_image_of_plane_0.png" in plot_patch.paths
    assert fit_imaging_plotter_path + "model_image_of_plane_1.png" in plot_patch.paths

    assert fit_imaging_plotter_path + "plane_image_of_plane_0.png" in plot_patch.paths
    assert fit_imaging_plotter_path + "plane_image_of_plane_1.png" in plot_patch.paths

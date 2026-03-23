from os import path

import pytest

from autolens.imaging.plot.fit_imaging_plots import (
    subplot_fit,
    subplot_fit_log10,
    subplot_of_planes,
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_fit_imaging_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test_subplot_fit_is_output(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_fit(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths


def test_subplot_fit_log10_is_output(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_fit_log10(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "subplot_fit_log10.png") in plot_patch.paths


def test__subplot_of_planes(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_of_planes(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "subplot_of_plane_0.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_of_plane_1.png") in plot_patch.paths

    plot_patch.paths = []

    subplot_of_planes(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
        plane_index=0,
    )

    assert path.join(plot_path, "subplot_of_plane_0.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_of_plane_1.png") not in plot_patch.paths

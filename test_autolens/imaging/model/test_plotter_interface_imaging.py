import os
import shutil
from os import path

import pytest
import autolens as al
from autolens.imaging.model.plotter_interface import PlotterInterfaceImaging

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_interface_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__fit_imaging(
    fit_imaging_x2_plane_inversion_7x7, include_2d_all, plot_path, plot_patch
):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    plotter_interface = PlotterInterfaceImaging(image_path=plot_path)

    plotter_interface.fit_imaging(
        fit=fit_imaging_x2_plane_inversion_7x7,
    )

    assert path.join(plot_path, "subplot_tracer.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_fit_log10.png") in plot_patch.paths

def test__fit_imaging_combined(
    fit_imaging_x2_plane_inversion_7x7, plot_path, plot_patch
):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceImaging(image_path=plot_path)

    visualizer.fit_imaging_combined(fit_list=2 * [fit_imaging_x2_plane_inversion_7x7])

    assert path.join(plot_path, "subplot_fit_combined.png") in plot_patch.paths
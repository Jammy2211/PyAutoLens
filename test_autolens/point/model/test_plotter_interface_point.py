import os
import shutil
from os import path

import pytest
from autolens.point.model.plotter_interface import PlotterInterfacePoint

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_interface_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__fit_point(fit_point_dataset_x2_plane, include_2d_all, plot_path, plot_patch):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    plotter_interface = PlotterInterfacePoint(image_path=plot_path)

    plotter_interface.fit_point(fit=fit_point_dataset_x2_plane, during_analysis=False)

    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths

    plot_path = path.join(plot_path, "fit_dataset")

    assert path.join(plot_path, "fit_point_positions.png") in plot_patch.paths

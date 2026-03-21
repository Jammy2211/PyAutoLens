from os import path

import pytest

from autolens.point.plot.fit_point_plots import subplot_fit

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_fit_point_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "fit_point",
    )


def test__subplot_fit(fit_point_dataset_x2_plane, plot_path, plot_patch):
    subplot_fit(
        fit=fit_point_dataset_x2_plane,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths

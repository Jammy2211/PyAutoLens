from os import path

import pytest

from autolens.point.plot.point_dataset_plots import subplot_dataset

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_point_dataset_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "point_dataset",
    )


def test__subplot_dataset(point_dataset, plot_path, plot_patch):
    subplot_dataset(
        dataset=point_dataset,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "subplot_dataset_point.png") in plot_patch.paths

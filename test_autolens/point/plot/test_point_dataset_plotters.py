from os import path

import pytest

import autolens.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_point_dataset_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "point_dataset",
    )


def test__point_dataset_quantities_are_output(point_dataset, plot_path, plot_patch):
    point_dataset_plotter = aplt.PointDatasetPlotter(
        dataset=point_dataset,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(path=plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    point_dataset_plotter.figures_2d(positions=True, fluxes=True)

    assert path.join(plot_path, "point_dataset_positions.png") in plot_patch.paths
    assert path.join(plot_path, "point_dataset_fluxes.png") in plot_patch.paths

    plot_patch.paths = []

    point_dataset_plotter.figures_2d(positions=True, fluxes=False)

    assert path.join(plot_path, "point_dataset_positions.png") in plot_patch.paths
    assert path.join(plot_path, "point_dataset_fluxes.png") not in plot_patch.paths

    plot_patch.paths = []

    point_dataset.fluxes = None

    point_dataset_plotter = aplt.PointDatasetPlotter(
        dataset=point_dataset,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    point_dataset_plotter.figures_2d(positions=True, fluxes=True)

    assert path.join(plot_path, "point_dataset_positions.png") in plot_patch.paths
    assert path.join(plot_path, "point_dataset_fluxes.png") not in plot_patch.paths


def test__subplot_dataset(point_dataset, plot_path, plot_patch):
    point_dataset_plotter = aplt.PointDatasetPlotter(
        dataset=point_dataset,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(path=plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    point_dataset_plotter.subplot_dataset()

    assert path.join(plot_path, "subplot_dataset_point.png") in plot_patch.paths

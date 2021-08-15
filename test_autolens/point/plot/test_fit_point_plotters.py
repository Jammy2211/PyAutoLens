from os import path

import pytest

import autolens.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_fit_point_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "fit_point",
    )


def test__fit_point_quantities_are_output(
    fit_point_dataset_x2_plane, include_2d_all, plot_path, plot_patch
):

    fit_point_plotter = aplt.FitPointDatasetPlotter(
        fit=fit_point_dataset_x2_plane,
        include_2d=include_2d_all,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(path=plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_point_plotter.figures_2d(positions=True, fluxes=True)

    assert path.join(plot_path, "fit_point_dataset_positions.png") in plot_patch.paths
    assert path.join(plot_path, "fit_point_dataset_fluxes.png") in plot_patch.paths

    plot_patch.paths = []

    fit_point_plotter.figures_2d(positions=True, fluxes=False)

    assert path.join(plot_path, "fit_point_dataset_positions.png") in plot_patch.paths
    assert path.join(plot_path, "fit_point_dataset_fluxes.png") not in plot_patch.paths

    plot_patch.paths = []

    fit_point_dataset_x2_plane.point_dataset.fluxes = None

    fit_point_plotter = aplt.FitPointDatasetPlotter(
        fit=fit_point_dataset_x2_plane,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_point_plotter.figures_2d(positions=True, fluxes=True)

    assert path.join(plot_path, "fit_point_dataset_positions.png") in plot_patch.paths
    assert path.join(plot_path, "fit_point_dataset_fluxes.png") not in plot_patch.paths


def test__subplot_fit_point(
    fit_point_dataset_x2_plane, include_2d_all, plot_path, plot_patch
):

    fit_point_plotter = aplt.FitPointDatasetPlotter(
        fit=fit_point_dataset_x2_plane,
        include_2d=include_2d_all,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(path=plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_point_plotter.subplot_fit_point()

    assert path.join(plot_path, "subplot_fit_point.png") in plot_patch.paths

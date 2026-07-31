import shutil
from pathlib import Path

import pytest
from autolens.point.model.plotter import PlotterPoint

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plotter_plotter_setup():
    return directory / "files"


def test__fit_point(fit_point_dataset_x2_plane, plot_path, plot_patch):
    if plot_path.exists():
        shutil.rmtree(plot_path)

    plotter = PlotterPoint(image_path=plot_path)

    plotter.fit_point(fit=fit_point_dataset_x2_plane)

    assert str(plot_path / "fit.png") in plot_patch.paths


def test__fit_point__quick_update__writes_normal_fit_subplot(
    fit_point_dataset_x2_plane, plot_path, plot_patch
):
    if plot_path.exists():
        shutil.rmtree(plot_path)

    plotter = PlotterPoint(image_path=plot_path)

    plotter.fit_point(fit=fit_point_dataset_x2_plane, quick_update=True)

    assert str(plot_path / "fit.png") in plot_patch.paths
    assert str(plot_path / "fit_quick.png") not in plot_patch.paths

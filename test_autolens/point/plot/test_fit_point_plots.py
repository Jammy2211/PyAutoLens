from pathlib import Path

import pytest

from autolens.point.plot.fit_point_plots import subplot_fit

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_fit_point_plotter_setup():
    return Path(__file__).resolve().parent / "files" / "plots" / "fit_point"


def test__subplot_fit(fit_point_dataset_x2_plane, plot_path, plot_patch):
    subplot_fit(
        fit=fit_point_dataset_x2_plane,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit.png") in plot_patch.paths


def test__plot_image_with_points(point_dataset, fit_point_dataset_x2_plane, plot_path, plot_patch):
    import autolens as al

    from autolens.point.plot.fit_point_plots import plot_image_with_points

    image = al.Array2D.ones(shape_native=(20, 20), pixel_scales=0.25)

    plot_image_with_points(
        image=image,
        dataset=point_dataset,
        fit=fit_point_dataset_x2_plane,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "image_with_points.png") in plot_patch.paths

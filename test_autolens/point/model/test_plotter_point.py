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


def _image():
    import autolens as al

    return al.Array2D.ones(shape_native=(20, 20), pixel_scales=0.25)


def _visualize(fit, image, plot_path, quick_update):
    from types import SimpleNamespace

    from autolens.point.model.visualizer import VisualizerPoint

    analysis = SimpleNamespace(
        dataset=fit.dataset,
        image=image,
        title_prefix=None,
        fit_for_visualization=lambda instance: fit,
    )
    paths = SimpleNamespace(image_path=plot_path, output_path=plot_path)

    VisualizerPoint.visualize_before_fit(analysis=analysis, paths=paths, model=None)
    VisualizerPoint.visualize(
        analysis=analysis,
        paths=paths,
        instance=None,
        during_analysis=False,
        quick_update=quick_update,
    )


def test__image_with_points(fit_point_dataset_x2_plane, plot_path, plot_patch):
    if plot_path.exists():
        shutil.rmtree(plot_path)

    plotter = PlotterPoint(image_path=plot_path)

    plotter.image_with_points(image=_image(), dataset=fit_point_dataset_x2_plane.dataset)

    assert str(plot_path / "image_with_points.png") in plot_patch.paths


@pytest.mark.parametrize("quick_update", [False, True])
def test__visualizer__image_with_points_before_and_after_fit(
    fit_point_dataset_x2_plane, plot_path, plot_patch, quick_update
):
    if plot_path.exists():
        shutil.rmtree(plot_path)

    _visualize(
        fit=fit_point_dataset_x2_plane,
        image=_image(),
        plot_path=plot_path,
        quick_update=quick_update,
    )

    image_with_points = [
        path for path in plot_patch.paths if path == str(plot_path / "image_with_points.png")
    ]

    # Once before the fit (observed positions) and once after (with model positions).
    assert len(image_with_points) == 2


def test__visualizer__no_image__no_image_with_points(
    fit_point_dataset_x2_plane, plot_path, plot_patch
):
    if plot_path.exists():
        shutil.rmtree(plot_path)

    _visualize(
        fit=fit_point_dataset_x2_plane, image=None, plot_path=plot_path, quick_update=True
    )

    assert str(plot_path / "fit.png") in plot_patch.paths
    assert str(plot_path / "image_with_points.png") not in plot_patch.paths

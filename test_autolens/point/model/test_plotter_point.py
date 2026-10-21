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
    assert str(plot_path / "fit_image_with_points.png") not in plot_patch.paths


def test__fit_image_with_points(fit_point_dataset_x2_plane, plot_path, plot_patch):
    if plot_path.exists():
        shutil.rmtree(plot_path)

    plotter = PlotterPoint(image_path=plot_path)

    plotter.image_with_points(
        image=_image(),
        dataset=fit_point_dataset_x2_plane.dataset,
        fit=fit_point_dataset_x2_plane,
    )

    assert str(plot_path / "fit_image_with_points.png") in plot_patch.paths
    assert str(plot_path / "image_with_points.png") not in plot_patch.paths


def test__visualizer__before_fit__only_image_with_points(
    fit_point_dataset_x2_plane, plot_path, plot_patch
):
    from types import SimpleNamespace

    from autolens.point.model.visualizer import VisualizerPoint

    if plot_path.exists():
        shutil.rmtree(plot_path)

    analysis = SimpleNamespace(
        dataset=fit_point_dataset_x2_plane.dataset, image=_image(), title_prefix=None
    )
    paths = SimpleNamespace(image_path=plot_path, output_path=plot_path)

    VisualizerPoint.visualize_before_fit(analysis=analysis, paths=paths, model=None)

    assert plot_patch.paths.count(str(plot_path / "image_with_points.png")) == 1
    assert str(plot_path / "fit_image_with_points.png") not in plot_patch.paths


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

    # The dataset-only figure before the fit and the fit figure after it are separate files, so both coexist.
    assert plot_patch.paths.count(str(plot_path / "image_with_points.png")) == 1
    assert plot_patch.paths.count(str(plot_path / "fit_image_with_points.png")) == 1


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
    assert str(plot_path / "fit_image_with_points.png") not in plot_patch.paths


def test__critical_curve_grid_from__contains_curve_beyond_multiple_images():
    import numpy as np

    import autolens as al
    from autolens.lens.plot.critical_curves import _compute_critical_curve_lines
    from autolens.lens.plot.critical_curves import critical_curve_grid_from

    # Multiple images inside a 1" box, an isothermal tangential critical curve of radius 1.6": a grid covering
    # only the images' bounding box clips the curve into arcs.
    dataset = al.PointDataset(
        name="point_0",
        positions=al.Grid2DIrregular([(1.0, 0.5), (0.5, 1.0), (-0.5, 0.0), (0.0, -0.5)]),
        positions_noise_map=al.ArrayIrregular([0.05, 0.05, 0.05, 0.05]),
    )

    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(redshift=0.5, mass=al.mp.Isothermal(einstein_radius=1.6)),
            al.Galaxy(redshift=1.0),
        ]
    )

    grid = critical_curve_grid_from(dataset=dataset)

    image_plane_lines, _, _, _ = _compute_critical_curve_lines(tracer, grid)

    assert len(image_plane_lines) == 1

    line = np.asarray(image_plane_lines[0])

    assert np.allclose(line[0], line[-1], atol=0.05)
    assert np.max(np.abs(line)) == pytest.approx(1.6, abs=0.05)


def test__critical_curve_grid_from__extent_is_x_first():
    import numpy as np

    import autolens as al
    from autolens.lens.plot.critical_curves import critical_curve_grid_from

    # Images spread far more in y than x: the grid must be centred on them in (y,x).
    dataset = al.PointDataset(
        name="point_0",
        positions=al.Grid2DIrregular([(3.0, 1.0), (5.0, 1.0)]),
        positions_noise_map=al.ArrayIrregular([0.05, 0.05]),
    )

    grid = np.asarray(critical_curve_grid_from(dataset=dataset).array)

    assert np.mean(grid[:, 0]) == pytest.approx(4.0, abs=0.05)
    assert np.mean(grid[:, 1]) == pytest.approx(1.0, abs=0.05)
    assert grid[:, 0].max() - grid[:, 0].min() == pytest.approx(
        grid[:, 1].max() - grid[:, 1].min(), abs=0.05
    )


def test__visualizer__before_fit__passes_analysis_solver_to_dataset_panel(
    fit_point_dataset_x2_plane, plot_path, plot_patch, monkeypatch
):
    from types import SimpleNamespace

    import autolens.point.model.plotter as plotter_module
    from autolens.point.model.visualizer import VisualizerPoint

    solvers = []

    def recording_subplot_dataset(dataset, **kwargs):
        solvers.append(kwargs.get("solver"))

    monkeypatch.setattr(plotter_module, "subplot_dataset", recording_subplot_dataset)

    analysis = SimpleNamespace(
        dataset=fit_point_dataset_x2_plane.dataset,
        solver=fit_point_dataset_x2_plane.solver,
        title_prefix=None,
    )
    paths = SimpleNamespace(image_path=plot_path, output_path=plot_path)

    VisualizerPoint.visualize_before_fit(analysis=analysis, paths=paths, model=None)

    assert solvers == [fit_point_dataset_x2_plane.solver]

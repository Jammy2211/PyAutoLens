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


@pytest.mark.parametrize(
    "fit_positions_cls, n_scatter", [("FitPositionsImagePair", 2), ("FitPositionsSource", 1)]
)
def test__plot_image_with_points__model_positions_only_for_image_plane_fits(
    point_dataset, plot_path, plot_patch, monkeypatch, fit_positions_cls, n_scatter
):
    import matplotlib.axes

    import autolens as al
    from autolens import fixtures

    from autolens.point.plot.fit_point_plots import plot_image_with_points

    fit = al.FitPointDataset(
        dataset=point_dataset,
        tracer=fixtures.make_tracer_x2_plane_point(),
        solver=fixtures.make_solver(),
        fit_positions_cls=getattr(al, fit_positions_cls),
    )

    scatter_calls = []
    scatter = matplotlib.axes.Axes.scatter

    def counting_scatter(self, *args, **kwargs):
        scatter_calls.append(kwargs.get("label"))
        return scatter(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", counting_scatter)

    plot_image_with_points(
        image=al.Array2D.ones(shape_native=(20, 20), pixel_scales=0.25),
        dataset=point_dataset,
        fit=fit,
        output_path=plot_path,
        output_format="png",
    )

    assert scatter_calls.count("Observed") == 1
    assert len([label for label in scatter_calls if label in ("Observed", "Model")]) == n_scatter

import numpy as np
import pytest

import autolens as al
from autolens import fixtures
from autolens.point.plot import fit_point_plots, multi_source_plots, point_dataset_plots
from autolens.point.plot.positions_panel import (
    fallback_extent_from,
    mass_centres_from,
    model_pairing_from,
    plot_fit_model_positions,
    positions_extent_from,
    sigma_label_from,
    unmatched_model_mask_from,
)


@pytest.fixture(name="figures")
def make_figures(monkeypatch):
    """
    Captures the figures the subplot functions would save, so their axes can be inspected.
    """
    figures = []

    def capture(fig, **kwargs):
        figures.append(fig)

    monkeypatch.setattr(point_dataset_plots, "save_figure", capture)
    monkeypatch.setattr(fit_point_plots, "save_figure", capture)
    monkeypatch.setattr(multi_source_plots, "save_figure", capture)
    return figures


def _legend_labels(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


def _solver(pixel_scales=0.5, shape_native=(20, 20)):
    return al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=shape_native, pixel_scales=pixel_scales),
        pixel_scale_precision=0.25,
        magnification_threshold=1e-8,
    )


def _point_dataset(noise=0.005):
    return al.PointDataset(
        name="point_0",
        positions=al.Grid2DIrregular([(1.0, 1.0), (-1.5, 0.5), (0.2, -2.2)]),
        positions_noise_map=noise,
    )


def test__positions_extent_from__solver_extent_when_solver_given():
    solver = _solver()
    extent = positions_extent_from(positions=_point_dataset().positions, solver=solver)

    assert extent == pytest.approx(
        [solver.x_min, solver.x_max, solver.y_min, solver.y_max]
    )


def test__positions_extent_from__enlarged_when_a_position_is_outside_solver_grid():
    solver = _solver(pixel_scales=0.1, shape_native=(10, 10))
    positions = _point_dataset().positions

    x_min, x_max, y_min, y_max = positions_extent_from(
        positions=positions, solver=solver
    )

    grid = np.asarray(positions.array)
    assert np.all(grid[:, 1] > x_min) and np.all(grid[:, 1] < x_max)
    assert np.all(grid[:, 0] > y_min) and np.all(grid[:, 0] < y_max)


def test__positions_extent_from__fallback_contains_all_positions():
    positions = _point_dataset().positions

    x_min, x_max, y_min, y_max = positions_extent_from(positions=positions, solver=None)

    assert [x_min, x_max, y_min, y_max] == fallback_extent_from(positions)
    grid = np.asarray(positions.array)
    assert np.all(grid[:, 1] > x_min) and np.all(grid[:, 1] < x_max)
    assert np.all(grid[:, 0] > y_min) and np.all(grid[:, 0] < y_max)


def test__sigma_label_from():
    assert sigma_label_from(al.ArrayIrregular([0.005, 0.005])) == "σ = 0.005″"
    assert sigma_label_from([0.005, 0.01]) == "σ = 0.005–0.01″"


def test__subplot_dataset__title_extent_sigma_legend_and_no_axis_labels(figures):
    solver = _solver()

    point_dataset_plots.subplot_dataset(dataset=_point_dataset(), solver=solver)

    ax = figures[0].axes[0]
    assert ax.get_title() == "Dataset Positions (point_0)"
    assert ax.get_xlabel() == "" and ax.get_ylabel() == ""
    assert ax.get_xlim() == pytest.approx((solver.x_min, solver.x_max))
    assert ax.get_ylim() == pytest.approx((solver.y_min, solver.y_max))
    assert _legend_labels(ax) == ["Observed (σ = 0.005″)"]

    # One noise circle per position, at the true noise-map radius (never inflated).
    radii = [patch.get_radius() for patch in ax.patches]
    assert radii == pytest.approx([0.005, 0.005, 0.005])


def test__subplot_dataset__title_prefix(figures):
    point_dataset_plots.subplot_dataset(dataset=_point_dataset(), title_prefix="Lens ")

    assert figures[0].axes[0].get_title() == "Lens Dataset Positions (point_0)"


def _fit(fit_positions_cls=al.FitPositionsImagePair):
    return al.FitPointDataset(
        dataset=fixtures.make_point_dataset(),
        tracer=fixtures.make_tracer_x2_plane_point(),
        solver=fixtures.make_solver(),
        fit_positions_cls=fit_positions_cls,
    )


def test__subplot_fit__title_extent_legend_mass_centres_and_curves(figures):
    fit = _fit()
    curve = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])

    fit_point_plots.subplot_fit(
        fit=fit,
        image_plane_lines=[curve],
        image_plane_line_colors=["white"],
    )

    ax = figures[0].axes[0]
    assert ax.get_title() == "Fit Positions (point_0)"
    # The source-plane panel moved to fit_per_image.png: fit.png is Fit Positions (+ Fit Fluxes).
    assert figures[0].axes[1].get_title() == "Fit Fluxes (point_0)"
    assert ax.get_xlabel() == "" and ax.get_ylabel() == ""
    assert ax.get_xlim() == pytest.approx((fit.solver.x_min, fit.solver.x_max))

    labels = _legend_labels(ax)
    assert "Observed (σ = 1″)" in labels
    assert "Critical curve" in labels
    assert "Mass centre" in labels

    # The mass centre of the tracer's lens galaxy is plotted.
    centres = mass_centres_from(fit.tracer)
    assert len(centres) >= 1
    mass_scatter = [c for c in ax.collections if c.get_label() == "Mass centre"][0]
    assert np.asarray(mass_scatter.get_offsets()) == pytest.approx(centres[:, ::-1])


def test__subplot_fit__computes_critical_curves_when_none_given(figures, monkeypatch):
    from autolens.point.plot import point_sources

    calls = []

    def fake_curves(tracer, plane_indexes, extent, shape=400):
        calls.append(list(plane_indexes))
        line = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
        return {
            j: point_sources.PlaneCurves(j, 1.0, [line], [], [line], [])
            for j in set(plane_indexes)
        }

    monkeypatch.setattr(point_sources, "curves_for_planes", fake_curves)

    fit_point_plots.subplot_fit(fit=_fit())

    assert len(calls) == 1
    assert "Critical curve" in _legend_labels(figures[0].axes[0])

    figures.clear()
    fit_point_plots.subplot_fit(fit=_fit(), image_plane_lines=[], source_plane_lines=[])

    assert len(calls) == 1
    assert "Critical curve" not in _legend_labels(figures[0].axes[0])


def test__subplot_fit__source_plane_fit_draws_no_model_on_image_plane(figures):
    fit_point_plots.subplot_fit(
        fit=_fit(fit_positions_cls=al.FitPositionsSource), image_plane_lines=[]
    )

    labels = _legend_labels(figures[0].axes[0])
    assert "Model (source plane)" not in labels
    assert "Model" not in labels
    assert "Residual" not in labels
    assert "Unmatched model image" not in labels


def _fixed_pair_fit(cls, data, model):
    """
    An image-plane positions fit of class ``cls`` with fixed observed and model positions.
    """

    class Fixed(cls):
        def __init__(self):
            pass

        @property
        def data(self):
            return al.Grid2DIrregular(data)

        @property
        def model_data(self):
            return al.Grid2DIrregular(model)

    return Fixed()


def test__model_pairing_from__each_fit_class_uses_its_own_matching():
    data = [(0.0, 0.0), (0.0, 0.4)]
    model = [(0.0, 0.3), (0.0, 1.0)]

    # Nearest neighbour (repeats allowed): both observed positions pair with model 0.
    _, _, indexes = model_pairing_from(
        _fixed_pair_fit(al.FitPositionsImagePairRepeat, data, model)
    )
    assert list(indexes) == [0, 0]

    # Hungarian one-to-one assignment: observed 0 -> model 0, observed 1 -> model 1.
    _, _, indexes = model_pairing_from(
        _fixed_pair_fit(al.FitPositionsImagePair, data, model)
    )
    assert sorted(indexes) == [0, 1]

    # Source-plane fits have no image-plane pairing.
    assert model_pairing_from(_fit(al.FitPositionsSource).positions) is None


def test__plot_fit_model_positions__residual_lines_and_unmatched_model_images():
    import matplotlib.pyplot as plt

    data = [(0.0, 0.0), (1.0, 1.0)]
    model = [(0.0, 0.1), (1.0, 1.1), (-2.0, 2.0), (np.inf, np.inf)]
    fit_positions = _fixed_pair_fit(al.FitPositionsImagePairRepeat, data, model)

    _, finite_model, indexes = model_pairing_from(fit_positions)
    assert len(finite_model) == 3
    assert list(unmatched_model_mask_from(finite_model, indexes)) == [
        False,
        False,
        True,
    ]

    fig, ax = plt.subplots()
    plot_fit_model_positions(ax, fit_positions)

    residual_lines = [line for line in ax.lines]
    assert len(residual_lines) == 2
    assert residual_lines[0].get_label() == "Residual"
    assert np.asarray(residual_lines[0].get_xydata()) == pytest.approx(
        np.array([[0.0, 0.0], [0.1, 0.0]])
    )

    scatters = {c.get_label(): c for c in ax.collections}
    assert np.asarray(scatters["Model"].get_offsets()) == pytest.approx(
        np.array([[0.1, 0.0], [1.1, 1.0]])
    )
    assert np.asarray(scatters["Unmatched model image"].get_offsets()) == pytest.approx(
        np.array([[2.0, -2.0]])
    )
    plt.close(fig)


def test__critical_curve_grid_from__covers_solver_grid():
    from autolens.lens.plot.critical_curves import critical_curve_grid_from

    dataset = _point_dataset()
    solver = _solver(pixel_scales=1.0, shape_native=(20, 20))

    grid = np.asarray(critical_curve_grid_from(dataset=dataset, solver=solver).array)

    assert grid[:, 0].min() <= solver.y_min + 1e-8
    assert grid[:, 0].max() >= solver.y_max - 1e-8
    assert grid[:, 1].min() <= solver.x_min + 1e-8
    assert grid[:, 1].max() >= solver.x_max - 1e-8

    # A solver grid inside the default square leaves it unchanged.
    small = _solver(pixel_scales=0.1, shape_native=(10, 10))
    assert np.asarray(
        critical_curve_grid_from(dataset=dataset, solver=small).array
    ) == pytest.approx(np.asarray(critical_curve_grid_from(dataset=dataset).array))

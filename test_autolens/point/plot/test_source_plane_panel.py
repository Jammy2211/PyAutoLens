import importlib
import inspect
import pkgutil

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Circle, Ellipse

import autolens as al
from autolens.point.fit.positions.abstract import AbstractFitPositions
from autolens.point.plot import fit_point_plots
from autolens.point.plot import source_plane_panel as spp
from autolens.point.plot.positions_panel import image_numbers_from, image_order_from

SOURCE_CENTRE = (0.07, 0.07)


class FitPositionsSourceJacobian(al.FitPositionsSource):
    weighting = "jacobian"


# Every positions fit class and the presentation its source-plane panel must use.
PRESENTATIONS = [
    (al.FitPositionsImagePair, "image", False, None),
    (al.FitPositionsImagePairRepeat, "image", False, None),
    (al.FitPositionsImagePairRepeatSolved, "image", True, None),
    (al.FitPositionsImagePairAll, "image", False, None),
    (al.FitPositionsImagePairAllSolved, "image", True, None),
    (al.FitPositionsSource, "source", False, "magnification"),
    (FitPositionsSourceJacobian, "source", False, "jacobian"),
    (al.FitPositionsSourceSolved, "source", True, "jacobian"),
]


def _tracer(solved: bool, einstein_radius: float = 1.6):
    point = al.ps.PointSolved() if solved else al.ps.Point(centre=SOURCE_CENTRE)
    return al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.0, 0.0),
                    einstein_radius=einstein_radius,
                    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
                ),
            ),
            al.Galaxy(redshift=1.0, point_0=point),
        ]
    )


@pytest.fixture(name="solver", scope="module")
def make_solver():
    return al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.5),
        pixel_scale_precision=0.01,
        magnification_threshold=0.1,
    )


@pytest.fixture(name="dataset", scope="module")
def make_dataset(solver):
    """The images of the true model, so every fit class (incl. the solved ones) has a well-posed fit."""
    positions = solver.solve(
        tracer=_tracer(solved=False), source_plane_coordinate=SOURCE_CENTRE
    )
    positions = np.asarray(positions.array)
    positions = positions[np.isfinite(positions).all(axis=1)]
    return al.PointDataset(
        name="point_0",
        positions=al.Grid2DIrregular(positions + 0.002),
        positions_noise_map=0.005,
    )


def _fit(cls, dataset, solver, einstein_radius=1.62):
    return al.FitPointDataset(
        dataset=dataset,
        tracer=_tracer(
            solved="Solved" in cls.__name__, einstein_radius=einstein_radius
        ),
        solver=solver,
        fit_positions_cls=cls,
    )


def _positions_fit_classes():
    import autolens.point.fit.positions as package

    classes = set()
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, AbstractFitPositions)
                and cls.__module__ == module.__name__
                and not cls.__name__.startswith("Abstract")
            ):
                classes.add(cls)
    return classes


def test__presentation_table__covers_every_positions_fit_class():
    classes = _positions_fit_classes()

    assert len(classes) == 7
    assert classes == set(spp.presentation_table())


@pytest.mark.parametrize("cls, plane, solved, weighting", PRESENTATIONS)
def test__presentation_from__paired_to_fit_class(
    cls, plane, solved, weighting, dataset, solver
):
    presentation = spp.presentation_from(_fit(cls, dataset, solver).positions)

    assert presentation.chi_squared_plane == plane
    assert presentation.solved_centre is solved
    assert presentation.weighting == weighting

    assert presentation.centre_marker == ("D" if solved else "*")
    assert presentation.centre_label == (
        "Solved source centre (β*)" if solved else "Model source centre"
    )


def _draw(fit_positions, caustics=True):
    data = spp.source_plane_data_from(fit_positions)
    tangential = [
        np.array([[0.0, 0.2], [0.2, 0.0], [0.0, -0.2], [-0.2, 0.0], [0.0, 0.2]])
    ]
    fig, ax = plt.subplots(figsize=(6, 6))
    zoom = spp.plot_source_plane_panel(
        ax,
        data,
        title="Fit Source Plane (point_0)",
        tangential_caustics=tangential if caustics else [],
    )
    return fig, ax, data, zoom


def _texts(ax):
    return [text.get_text() for text in ax.texts]


def _legend_labels(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


@pytest.mark.parametrize("cls, plane, solved, weighting", PRESENTATIONS)
def test__plot_source_plane_panel__draws_what_the_fit_class_uses(
    cls, plane, solved, weighting, dataset, solver
):
    fit_positions = _fit(cls, dataset, solver).positions
    fig, ax, data, zoom = _draw(fit_positions)

    circles = [p for p in ax.patches if isinstance(p, Circle)]
    ellipses = [
        p for p in ax.patches if isinstance(p, Ellipse) and not isinstance(p, Circle)
    ]
    solid_ellipses = [e for e in ellipses if e.get_linestyle() in ("-", "solid")]
    dashed_ellipses = [e for e in ellipses if e.get_linestyle() in ("--", "dashed")]
    labels = _legend_labels(ax)
    chi_squared_labels = [label for label in labels if label.startswith("χ² =")]
    n = len(data.traced)

    assert ax.get_title() == "Fit Source Plane (point_0)"
    # No subtitle or other free-floating fit-type text: only the image numbers are annotations.
    assert all(text.get_text().isdigit() for text in ax.texts)
    assert not any(label.startswith("χ²") for label in _texts(ax))
    assert data.presentation.centre_label in labels
    assert "Caustic" in labels
    assert len(ax.child_axes) == 1  # the caustic inset

    if plane == "image":
        assert circles == [] and ellipses == []
        assert chi_squared_labels == []
        assert not any("noise" in label for label in labels)
    elif weighting == "magnification":
        assert len(circles) == n
        assert len(dashed_ellipses) == n and solid_ellipses == []
        assert any(label.startswith("χ² noise σ/|μ|") for label in labels)
        assert "Propagated noise A·σ (reference)" in labels
        assert chi_squared_labels == [
            f"χ² = {spp._format_chi_squared(data.chi_squared)}"
        ]
    else:
        assert circles == []
        assert len(solid_ellipses) == n and dashed_ellipses == []
        assert "χ² noise A·σ (jacobian)" in labels
        assert "Propagated noise A·σ (reference)" not in labels
        assert chi_squared_labels == [
            f"χ² = {spp._format_chi_squared(data.chi_squared)}"
        ]

    # The zoom contains every traced position and the centre.
    x_min, x_max, y_min, y_max = zoom
    points = np.vstack([data.traced, data.centre[None, :]])
    assert np.all((points[:, 1] > x_min) & (points[:, 1] < x_max))
    assert np.all((points[:, 0] > y_min) & (points[:, 0] < y_max))

    plt.close(fig)


def test__plot_source_plane_panel__no_inset_without_caustics(dataset, solver):
    fig, ax, _, _ = _draw(
        _fit(al.FitPositionsImagePair, dataset, solver).positions, caustics=False
    )

    assert ax.child_axes == []
    plt.close(fig)


def test__source_plane_data__matches_the_fits_own_chi_squared(dataset, solver):
    magnification = _fit(al.FitPositionsSource, dataset, solver).positions
    data = spp.source_plane_data_from(magnification)

    # Traced exactly as the fit traces them.
    assert data.traced == pytest.approx(np.asarray(magnification.model_data.array))

    # The σ/|μ| circle is the noise of the magnification chi-squared.
    residuals = np.linalg.norm(data.traced - data.centre, axis=1)
    assert data.chi_squared == pytest.approx(
        np.sum(residuals**2 / data.magnification_radii**2)
    )

    # The A·σ ellipse is the covariance of the jacobian chi-squared.
    jacobian = _fit(FitPositionsSourceJacobian, dataset, solver).positions
    jacobian_data = spp.source_plane_data_from(jacobian)
    assert spp.chi_squared_from(
        jacobian_data.traced - jacobian_data.centre, jacobian_data.covariances
    ) == pytest.approx(jacobian_data.chi_squared, rel=1e-6)

    # The solved centre is the fit's own β*.
    solved = _fit(al.FitPositionsSourceSolved, dataset, solver).positions
    assert spp.source_plane_data_from(solved).centre == pytest.approx(
        np.asarray(solved.source_plane_coordinate)
    )


def test__image_plane_fit__no_chi_squared_computed(dataset, solver):
    data = spp.source_plane_data_from(
        _fit(al.FitPositionsImagePair, dataset, solver).positions
    )

    assert data.chi_squared is None
    assert spp.chi_squared_label_from(data) is None


def test__ellipse_from__axes_and_angle():
    semi_major, semi_minor, angle = spp.ellipse_from(np.array([[1.0, 0.0], [0.0, 4.0]]))

    # (y,x) covariance: variance 4 along x.
    assert semi_major == pytest.approx(2.0)
    assert semi_minor == pytest.approx(1.0)
    assert abs(np.cos(np.radians(angle))) == pytest.approx(1.0)

    assert spp.ellipse_from(np.full((2, 2), np.nan)) is None


def test__image_numbers_from__point_solver_order():
    positions = np.array([[-1.0, -1.0], [1.0, 0.1], [0.1, 1.0]])

    order = image_order_from(positions)
    numbers = image_numbers_from(positions)

    assert list(order) == [1, 2, 0]
    assert list(numbers) == [3, 1, 2]


@pytest.fixture(name="figures")
def make_figures(monkeypatch):
    figures = []
    monkeypatch.setattr(
        fit_point_plots, "save_figure", lambda fig, **kwargs: figures.append(fig)
    )
    return figures


def _number_texts(ax):
    return sorted(
        (t.get_text(), tuple(np.round(t.xy, 8)))
        for t in ax.texts
        if t.get_text().isdigit()
    )


@pytest.mark.parametrize("fluxes", [None, [1.0, 2.0, 3.0, 4.0]])
def test__subplot_fit__layout_and_consistent_image_numbers(
    figures, dataset, solver, fluxes
):
    dataset = al.PointDataset(
        name="point_0",
        positions=dataset.positions,
        positions_noise_map=dataset.positions_noise_map,
        fluxes=None if fluxes is None else fluxes[: len(dataset.positions)],
        fluxes_noise_map=None if fluxes is None else [0.1] * len(dataset.positions),
    )
    fit = _fit(al.FitPositionsSource, dataset, solver)

    fit_point_plots.subplot_fit(fit=fit, image_plane_lines=[], source_plane_lines=[])

    axes = figures[0].axes
    titles = [ax.get_title() for ax in axes]
    expected = ["Fit Positions (point_0)", "Fit Source Plane (point_0)"]
    if fluxes is not None:
        expected.append("Fit Fluxes (point_0)")
    assert titles == expected

    # The source-plane fit's traced positions are not drawn on the image plane.
    image_labels = _legend_labels(axes[0])
    assert "Model (source plane)" not in image_labels

    # Image k is the same observed position on both panels.
    numbers = image_numbers_from(dataset.positions)
    traced = spp.traced_positions_from(fit.positions)
    observed = np.asarray(dataset.positions.array)
    expected_image = sorted(
        (str(n), tuple(np.round(p[::-1], 8))) for n, p in zip(numbers, observed)
    )
    expected_source = sorted(
        (str(n), tuple(np.round(p[::-1], 8))) for n, p in zip(numbers, traced)
    )
    assert _number_texts(axes[0]) == expected_image
    assert _number_texts(axes[1]) == expected_source


def _three_plane_fit(solver, dataset):
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.4),
            ),
            al.Galaxy(
                redshift=1.0,
                mass=al.mp.Isothermal(centre=(0.3, 0.3), einstein_radius=0.2),
                point_0=al.ps.Point(centre=SOURCE_CENTRE),
            ),
            al.Galaxy(redshift=2.0, mass=al.mp.Isothermal(einstein_radius=0.1)),
        ]
    )
    return al.FitPointDataset(
        dataset=dataset,
        tracer=tracer,
        solver=solver,
        fit_positions_cls=al.FitPositionsSource,
    )


def test__source_plane_caustics_from__point_plane_in_multi_plane_tracer(
    monkeypatch, solver, dataset
):
    fit = _three_plane_fit(solver, dataset)
    assert fit.positions.plane_index == 1

    calls = []

    def fake_caustics(tracer, plane_index, grid):
        calls.append(plane_index)
        return [np.array([[0.0, 0.0], [0.1, 0.1]])], []

    monkeypatch.setattr(spp, "point_plane_caustics_from", fake_caustics)

    # The pre-computed lines are the last plane's: not reused for a point in plane 1.
    tangential, _ = fit_point_plots.source_plane_caustics_from(
        fit=fit,
        lines_given=True,
        source_plane_lines=[np.array([[5.0, 5.0], [6.0, 6.0]])],
        source_plane_line_colors=["white"],
    )
    assert calls == [1]
    assert tangential[0] == pytest.approx(np.array([[0.0, 0.0], [0.1, 0.1]]))

    # A two-plane point is in the last plane: the pre-computed lines are reused.
    two_plane = _fit(al.FitPositionsSource, dataset, solver)
    tangential, radial = fit_point_plots.source_plane_caustics_from(
        fit=two_plane,
        lines_given=True,
        source_plane_lines=[
            np.array([[5.0, 5.0], [6.0, 6.0]]),
            np.array([[1.0, 1.0], [2.0, 2.0]]),
        ],
        source_plane_line_colors=["white", "yellow"],
    )
    assert calls == [1]
    assert len(tangential) == 1 and len(radial) == 1


def test__point_plane_caustics_from__own_plane_differs_from_last_plane(solver, dataset):
    fit = _three_plane_fit(solver, dataset)
    grid = al.Grid2D.uniform(shape_native=(80, 80), pixel_scales=0.05)

    own, _ = spp.point_plane_caustics_from(fit.tracer, plane_index=1, grid=grid)
    last, _ = spp.point_plane_caustics_from(fit.tracer, plane_index=2, grid=grid)

    assert len(own) >= 1 and len(last) >= 1
    assert np.vstack(own).mean(axis=0) != pytest.approx(np.vstack(last).mean(axis=0))


@pytest.mark.parametrize(
    "cls, label",
    [
        (al.FitPositionsImagePair, "Model"),
        (al.FitPositionsImagePairAllSolved, "Model (from solved β*)"),
        (al.FitPositionsImagePairRepeatSolved, "Model (from solved β*)"),
    ],
)
def test__fit_positions_panel__model_label_names_a_solved_centre(
    cls, label, dataset, solver
):
    from autolens.point.plot.positions_panel import model_label_from

    assert model_label_from(_fit(cls, dataset, solver).positions) == label


@pytest.mark.parametrize(
    "cls, linestyle, label",
    [
        (al.FitPositionsImagePair, "-", "Residual"),
        (al.FitPositionsImagePairRepeat, "-", "Residual"),
        (al.FitPositionsImagePairRepeatSolved, "-", "Residual"),
        (al.FitPositionsImagePairAll, "--", "Residual (nearest, soft pairing)"),
        (al.FitPositionsImagePairAllSolved, "--", "Residual (nearest, soft pairing)"),
    ],
)
def test__fit_positions_panel__residual_lines_styled_by_pairing(
    cls, linestyle, label, dataset, solver
):
    from autolens.point.plot.positions_panel import (
        plot_fit_model_positions,
        residual_style_from,
    )

    fit_positions = _fit(cls, dataset, solver).positions
    assert residual_style_from(fit_positions) == (linestyle, label)

    fig, ax = plt.subplots()
    plot_fit_model_positions(ax, fit_positions)
    residuals = [line for line in ax.lines if line.get_color() == "r"]
    assert residuals
    expected = {"-": ("-", "solid"), "--": ("--", "dashed")}[linestyle]
    assert all(line.get_linestyle() in expected for line in residuals)
    assert [line.get_label() for line in residuals][0] == label
    plt.close(fig)

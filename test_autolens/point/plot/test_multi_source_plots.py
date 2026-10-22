import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import autolens as al
from autolens.point.plot import multi_source_plots as msp
from autolens.point.plot import point_sources as ps

directory = Path(__file__).resolve().parent


def _tracer(n_sources=2):
    redshifts = [1.0, 2.0, 1.5, 2.5, 3.0, 1.2, 1.8, 2.2, 2.8, 3.5][:n_sources]
    centres = [(0.05, 0.03), (-0.04, 0.06), (0.02, -0.05), (0.0, 0.08), (-0.06, -0.02)]
    sources = [
        al.Galaxy(
            redshift=z,
            **{f"point_{k}": al.ps.Point(centre=centres[k % len(centres)])},
        )
        for k, z in enumerate(redshifts)
    ]
    return al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.0, 0.0), ell_comps=(0.05, 0.0), einstein_radius=1.0
                ),
            )
        ]
        + sources
    )


def _solver():
    return al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(60, 60), pixel_scales=0.05),
        pixel_scale_precision=0.001,
        magnification_threshold=0.1,
    )


def _datasets(tracer, solver, n_sources=2, offset=0.003):
    datasets = []
    for k in range(n_sources):
        plane_index = tracer.extract_plane_index_of_profile(profile_name=f"point_{k}")
        centre = tracer.planes[plane_index][0].cls_list_from(cls=al.ps.Point)[0].centre
        positions = solver.solve(
            tracer=tracer, source_plane_coordinate=centre, plane_redshift=tracer.planes[plane_index].redshift
        )
        positions = np.asarray(positions.array)
        positions = positions[np.isfinite(positions).all(axis=1)] + offset
        datasets.append(
            al.PointDataset(
                name=f"point_{k}",
                positions=al.Grid2DIrregular(values=positions),
                positions_noise_map=0.005,
                redshift=float(tracer.planes[plane_index].redshift),
            )
        )
    return datasets


@pytest.fixture(name="cluster", scope="module")
def make_cluster():
    tracer = _tracer()
    solver = _solver()
    datasets = _datasets(tracer, solver)
    fits = {
        cls.__name__: [
            al.FitPointDataset(
                dataset=d, tracer=tracer, solver=solver, fit_positions_cls=cls
            )
            for d in datasets
        ]
        for cls in (
            al.FitPositionsImagePair,
            al.FitPositionsImagePairAll,
            al.FitPositionsSource,
        )
    }
    return SimpleNamespace(tracer=tracer, solver=solver, datasets=datasets, fits=fits)


@pytest.fixture(name="figures")
def make_figures(monkeypatch):
    figures = []

    def capture(fig, path=None, filename=None, **kwargs):
        figures.append(SimpleNamespace(fig=fig, path=path, filename=filename))

    monkeypatch.setattr(msp, "save_figure", capture)
    return figures


def _legend_labels(ax):
    legend = ax.get_legend()
    return [] if legend is None else [t.get_text() for t in legend.get_texts()]


def test__curves_for_planes__each_source_plane_matches_lens_calc(cluster):
    from autogalaxy.operate.lens_calc import LensCalc

    tracer = cluster.tracer
    extent = [-2.0, 2.0, -2.0, 2.0]
    curves = ps.curves_for_planes(tracer, [1, 2], extent, shape=300)

    grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.02)
    radii = {}
    for j in (1, 2):
        lens_calc = LensCalc.from_tracer(
            tracer=tracer, use_multi_plane=True, plane_i=0, plane_j=j
        )
        reference = np.asarray(
            lens_calc.tangential_critical_curve_list_from(grid=grid)[0].array
        )
        fast = curves[j].tangential[0]
        assert np.median(np.hypot(*fast.T)) == pytest.approx(
            np.median(np.hypot(*reference.T)), abs=0.03
        )
        radii[j] = np.median(np.hypot(*fast.T))
        # Caustics are the critical curves traced to the plane: far smaller than the curves.
        assert np.max(np.abs(curves[j].tangential_caustics[0])) < 0.5

    # The z=2 source is lensed more strongly than the z=1 source: a larger critical curve.
    assert radii[2] > radii[1] + 0.05


def test__curves_for_planes__fast_plots_returns_none(cluster, monkeypatch):
    monkeypatch.setenv("PYAUTO_FAST_PLOTS", "1")
    assert ps.curves_for_planes(cluster.tracer, [1], [-2, 2, -2, 2]) == {}


def test__assign_styles__black_first_then_colour_and_shape_then_redshift():
    records = [
        ps.SourceRecord(
            k=k, name=f"point_{k}", redshift=1.0 + 0.1 * k,
            observed=np.zeros((1, 2)), sigma=np.ones(1), numbers=np.ones(1, dtype=int),
        )
        for k in range(10)
    ]
    assert ps.assign_styles(records[:3]) is None
    assert [r.color for r in records[:3]] == ["k", "#E69F00", "#56B4E9"]
    assert [r.marker for r in records[:3]] == ["o", "s", "^"]

    colormap = ps.assign_styles(records)
    assert colormap is not None
    assert len({tuple(r.color) for r in records}) == 10


def test__plot_dataset_positions__all_sources_labelled_no_curves(cluster, figures):
    msp.plot_dataset_positions(cluster.datasets, solver=cluster.solver)

    ax = figures[0].fig.axes[0]
    assert figures[0].filename == "dataset_point"
    assert ax.get_title() == "Dataset Positions"
    labels = _legend_labels(ax)
    assert labels == ["point_0 (z = 1, σ = 0.005″)", "point_1 (z = 2, σ = 0.005″)"]
    texts = {t.get_text() for t in ax.texts}
    assert "0·1" in texts and "1·1" in texts
    assert len(ax.lines) == 0


def test__plot_fit_positions__global_figure(cluster, figures):
    msp.plot_fit_positions(cluster.fits["FitPositionsImagePair"], solver=cluster.solver)

    ax = figures[0].fig.axes[0]
    assert figures[0].filename == "fit"
    assert ax.get_title() == "Fit Positions"
    labels = _legend_labels(ax)
    assert "point_0 (z = 1, σ = 0.005″)" in labels
    assert "Critical curve (each source's z)" in labels
    assert any(label.startswith("χ² = ") for label in labels)
    # Residual lines are drawn at true length (observed -> model), never exaggerated.
    record = ps.source_record_from_fit(0, cluster.fits["FitPositionsImagePair"][0])
    residuals = [
        line for line in ax.lines if line.get_color() == "r" and len(line.get_xydata()) == 2
    ]
    lengths = sorted(np.hypot(*np.diff(line.get_xydata(), axis=0)[0]) for line in residuals)
    assert max(lengths) < 10 * np.nanmax(record.delta) + 1e-6


def test__plot_fit_positions__mixture_shows_minus_two_ln_l(cluster, figures):
    msp.plot_fit_positions(
        cluster.fits["FitPositionsImagePairAll"], solver=cluster.solver
    )

    labels = _legend_labels(figures[0].fig.axes[0])
    assert any(label.startswith("−2 ln L = ") for label in labels)
    assert not any(label.startswith("χ²") for label in labels)
    assert "Residual (nearest, soft pairing)" in labels


def test__subplot_fit_per_image__rows_no_legends_sigma_values(cluster, figures):
    msp.subplot_fit_per_image(
        cluster.fits["FitPositionsImagePair"], solver=cluster.solver
    )

    fig = figures[0].fig
    assert figures[0].filename == "fit_per_image"
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == [
        "Residuals (point_0, z = 1)",
        "IP (point_0, z = 1)",
        "SP (point_0, z = 1)",
        "Residuals (point_1, z = 2)",
        "IP (point_1, z = 2)",
        "SP (point_1, z = 2)",
    ]
    assert all(ax.get_legend() is None for ax in fig.axes)
    ranking = fig.axes[0]
    assert not ranking.xaxis.get_visible()
    assert any(t.get_text().endswith("σ") for t in ranking.texts)


def test__subplot_fit_per_image__source_plane_fit_ranks_by_own_chi(cluster):
    fit = cluster.fits["FitPositionsSource"][0]
    record = ps.source_record_from_fit(0, fit)
    chi_squared_map = np.asarray(fit.positions.chi_squared_map.array).ravel()
    assert record.residual_sigmas == pytest.approx(np.sqrt(chi_squared_map))
    assert not record.has_model


def test__ranking_scales__shared_where_all_sources_visible_lone_source_own_range():
    def record(values, k):
        n = len(values)
        r = ps.SourceRecord(
            k=k, name=f"point_{k}", redshift=1.0, observed=np.zeros((n, 2)),
            sigma=np.ones(n), numbers=np.arange(1, n + 1),
        )
        r.delta = np.asarray(values, dtype=float)
        return r

    small = record([0.5, 1.0], 0)
    large = record([10.0, 20.0, 30.0], 1)

    shared = ps.ranking_scales_from([small, large])
    lone = ps.ranking_scales_from([small])

    assert shared.xmax > 20.0
    assert lone.xmax == pytest.approx(4.0)
    assert shared.n_slots == 3


def test__subplot_fit_per_image__many_sources_one_file_each(figures):
    tracer = _tracer(n_sources=10)
    solver = _solver()
    fits = [
        al.FitPointDataset(
            dataset=d, tracer=tracer, solver=solver,
            fit_positions_cls=al.FitPositionsImagePair,
        )
        for d in _datasets(tracer, solver, n_sources=10)
    ]

    msp.subplot_fit_per_image(fits, curves={}, output_path="out")

    assert len(figures) == 10
    assert {f.filename for f in figures} == {f"point_{k}" for k in range(10)}
    assert all(str(f.path).endswith("fit_per_image") for f in figures)


@pytest.mark.parametrize("with_fit", [False, True])
def test__plot_image_with_points__all_sources_no_legend(cluster, figures, with_fit):
    image = al.Array2D.ones(shape_native=(30, 30), pixel_scales=0.1)

    msp.plot_image_with_points(
        image,
        datasets=cluster.datasets,
        fits=cluster.fits["FitPositionsImagePair"] if with_fit else None,
        solver=cluster.solver,
    )

    ax = figures[0].fig.axes[0]
    assert ax.get_legend() is None
    assert figures[0].filename == ("fit_image_with_points" if with_fit else "image_with_points")
    texts = {t.get_text() for t in ax.texts}
    assert "0·1" in texts and "1·1" in texts


def test__visualizer__combined_figures_written_for_several_point_analyses(
    cluster, plot_patch
):
    from autolens.point.model.visualizer import VisualizerPoint

    plot_path = directory / "files" / "plots" / "combined"
    if plot_path.exists():
        shutil.rmtree(plot_path)

    image = al.Array2D.ones(shape_native=(30, 30), pixel_scales=0.1)
    analyses = [
        al.AnalysisPoint(
            dataset=d,
            solver=cluster.solver,
            image=image,
            fit_positions_cls=al.FitPositionsImagePair,
            use_jax=False,
        )
        for d in cluster.datasets
    ]
    instance = SimpleNamespace(galaxies=cluster.tracer.galaxies)
    paths = SimpleNamespace(image_path=plot_path, output_path=plot_path)

    VisualizerPoint.visualize_combined(
        analyses=analyses,
        paths=paths,
        instance=[instance, instance],
        during_analysis=False,
    )

    for name in (
        "fit.png",
        "fit_per_image.png",
        "fit_image_with_points.png",
        "dataset_point.png",
        "image_with_points.png",
    ):
        assert str(plot_path / name) in plot_patch.paths

    # A single point analysis has no combined figures (its own figures are output per analysis).
    plot_patch.paths.clear()
    VisualizerPoint.visualize_combined(
        analyses=analyses[:1], paths=paths, instance=[instance], during_analysis=False
    )
    assert plot_patch.paths == []


def test__plotter__fit_per_image_toggle(cluster, plot_patch, monkeypatch):
    import autolens.point.model.plotter as plotter_module
    from autolens.point.model.plotter import PlotterPoint

    plot_path = directory / "files" / "plots" / "toggle"
    plotter = PlotterPoint(image_path=plot_path)
    fits = cluster.fits["FitPositionsImagePair"][:1]

    plotter.fit_per_image(fit_list=fits, curves={})
    assert str(plot_path / "fit_per_image.png") in plot_patch.paths

    plot_patch.paths.clear()
    monkeypatch.setattr(plotter_module, "plot_setting", lambda section, name: False)
    plotter.fit_per_image(fit_list=fits, curves={})
    assert plot_patch.paths == []

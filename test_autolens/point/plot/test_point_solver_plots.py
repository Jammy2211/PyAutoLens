from pathlib import Path

import numpy as np
import pytest

import autolens as al
import autolens.plot as aplt
from autolens.point.plot import point_solver_plots as psp


@pytest.fixture(name="tracer")
def make_tracer():
    return al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.0, 0.0),
                    einstein_radius=1.6,
                    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
                ),
            ),
            al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.07, 0.07))),
        ]
    )


@pytest.fixture(name="solver")
def make_solver():
    """A coarse teaching grid: 6 refinement steps, quick to solve."""
    return al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.5),
        pixel_scale_precision=0.01,
        magnification_threshold=0.1,
    )


@pytest.fixture(name="data")
def make_data(solver, tracer):
    return psp.point_solver_data_from(
        solver=solver,
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
        image_plane_lines=[],
    )


class _Step:
    def __init__(self, number):
        self.number = number


def test__select_steps__first_three_and_final():
    steps = [_Step(i) for i in range(6)]
    assert [s.number for s in psp.select_steps(steps)] == [0, 1, 2, 5]

    steps = [_Step(i) for i in range(3)]
    assert [s.number for s in psp.select_steps(steps)] == [0, 1, 2]

    steps = [_Step(0)]
    assert [s.number for s in psp.select_steps(steps)] == [0]


def test__triangles_xy_from__swaps_yx_and_drops_nan_rows():
    triangles = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
        ]
    )
    xy = psp.triangles_xy_from(triangles)
    assert xy.shape == (1, 3, 2)
    assert xy[0, 0].tolist() == [2.0, 1.0]


def test__point_solver_data_from(data, solver):
    assert data.n_steps == solver.n_steps
    assert [s.number for s in data.steps] == [0, 1, 2, solver.n_steps - 1]

    # Four images, plus the lens-centre candidate the magnification threshold rejects.
    assert len(data.solved) == 4
    assert len(data.rejected) == 1
    assert np.linalg.norm(data.rejected[0]) < 0.1
    assert data.rejected_at_lens_centre

    # Every panel shares one extent covering the solver grid.
    assert data.extent[0] < solver.x_min and data.extent[1] > solver.x_max
    assert data.extent[2] < solver.y_min and data.extent[3] > solver.y_max


def test__split_kept__rejected_triangles_converge_on_rejected_candidate(data):
    kept, rejected = psp.split_kept_from(
        data.steps[0], data.final_centres, data.rejected
    )
    assert len(kept) > 0
    assert len(rejected) > 0
    centre = rejected.mean(axis=(0, 1))
    assert np.linalg.norm(centre) < 0.5


def test__too_small_to_draw__final_step_only(data):
    assert not psp.is_too_small_to_draw(data.steps[0], data, panel_inches=5.0)
    assert psp.is_too_small_to_draw(data.steps[-1], data, panel_inches=5.0)


def test__draw_panel__final_step_draws_centroid_markers(data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    marked = psp.draw_point_solver_panel(ax, data.steps[-1], data, panel_inches=5.0)
    assert marked
    assert any(
        getattr(c, "get_paths", None) and c.get_offsets().shape[0] > 0
        for c in ax.collections
    )
    assert ax.get_xlabel() == "" and ax.get_ylabel() == ""
    assert all('"' in label.get_text() for label in ax.get_xticklabels())
    plt.close(fig)


def test__legend__too_small_entry_only_when_marked(data):
    labels = [h.get_label() for h in psp.legend_handles_from(data, marked=False)]
    assert "Kept (too small to draw)" not in labels
    assert "Kept, then rejected (μ < 0.1)" in labels

    labels = [h.get_label() for h in psp.legend_handles_from(data, marked=True)]
    assert "Kept (too small to draw)" in labels


def test__plot_point_solver__still(solver, tracer, plot_patch, tmp_path):
    aplt.plot_point_solver(
        solver=solver,
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
        positions=al.Grid2DIrregular([(1.0, 1.0)]),
        image_plane_lines=[],
        output_path=tmp_path,
        output_format="png",
    )
    assert str(tmp_path / "point_solver.png") in plot_patch.paths


def test__plot_point_solver__animate_writes_looping_gif(solver, tracer, tmp_path):
    from PIL import Image

    aplt.plot_point_solver(
        solver=solver,
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
        image_plane_lines=[],
        output_path=tmp_path,
        animate=True,
    )
    gif = tmp_path / "point_solver.gif"
    assert gif.is_file()

    with Image.open(gif) as image:
        durations = []
        for i in range(image.n_frames):
            image.seek(i)
            durations.append(image.info["duration"])
        assert image.info.get("loop") == 0

    # One frame per step (0, 1, 2, final), the identical held final frames merged into one longer frame.
    assert len(durations) == 4
    assert durations[-1] == pytest.approx(
        psp.FINAL_FRAME_REPEATS * durations[0], rel=0.05
    )


def test__plot_point_solver__animate_without_path_returns_animation(solver, tracer):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    animation = aplt.plot_point_solver(
        solver=solver,
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
        image_plane_lines=[],
        animate=True,
    )
    assert isinstance(animation, FuncAnimation)
    animation._draw_was_started = (
        True  # never rendered here, which matplotlib would otherwise warn about
    )
    plt.close("all")


def test__warn_if_near_jax_cap(data, caplog):
    import logging

    data.max_kept = 3
    with caplog.at_level(logging.WARNING):
        psp.warn_if_near_jax_cap(data)
    assert "MAX_CONTAINING_SIZE" not in caplog.text

    data.max_kept = 19
    with caplog.at_level(logging.WARNING):
        psp.warn_if_near_jax_cap(data)
    assert "MAX_CONTAINING_SIZE" in caplog.text


def _number_texts(ax):
    return [t for t in ax.texts if t.get_text().isdigit()]


def _gap_in_points(fig, ax, text, x, y):
    """Distance (points) from an image's display position to the nearest point of its number's box."""
    bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
    px, py = ax.transData.transform((x, y))
    gap_x = max(bbox.x0 - px, 0.0, px - bbox.x1)
    gap_y = max(bbox.y0 - py, 0.0, py - bbox.y1)
    return np.hypot(gap_x, gap_y) * 72.0 / fig.dpi


@pytest.mark.parametrize("crowded", [False, True])
def test__image_numbers_sit_next_to_their_own_images(data, crowded):
    """
    Every step-0 image number is drawn a few points from its own image, never moved away to find space, even when
    images crowd together (the crowded case packs the images into a small part of the panel).
    """
    import matplotlib.pyplot as plt

    if crowded:
        data.solved = np.array(
            [[0.2, 0.0], [0.25, 0.12], [0.1, 0.22], [-0.05, 0.15], [0.3, -0.1]]
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    psp.draw_point_solver_panel(ax, data.steps[0], data, panel_inches=5.0)
    fig.canvas.draw()

    texts = _number_texts(ax)
    assert [t.get_text() for t in texts] == [
        str(i) for i in range(1, len(data.solved) + 1)
    ]
    for text, (y, x) in zip(texts, data.solved):
        assert text.xy == pytest.approx((x, y))
        assert _gap_in_points(fig, ax, text, x, y) < 10.0
    plt.close(fig)


def test__several_rejected_candidates__corner_label_with_one_arrow_each(data):
    import matplotlib.pyplot as plt

    data.rejected = np.array([[0.0, 0.0], [1.0, -1.0], [-1.0, 0.5]])
    fig, ax = plt.subplots(figsize=(6, 6))
    psp.draw_point_solver_panel(ax, data.steps[0], data, panel_inches=5.0)

    labels = [t for t in ax.texts if "magnification_threshold" in t.get_text()]
    assert len(labels) == 1
    assert labels[0].get_transform() == ax.transAxes
    arrows = [
        t for t in ax.texts if t.get_text() == "" and getattr(t, "arrow_patch", None)
    ]
    assert len(arrows) == 3
    plt.close(fig)

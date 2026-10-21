"""
The ``point_solver.png`` figure: how the ``PointSolver`` refines its triangle tiling until it finds the multiple
images of a point source.

The solver tiles the image plane with triangles, ray-traces every triangle to the source plane and keeps the
triangles whose traced image contains the source-plane coordinate. The kept triangles, plus a rim of neighbours, are
up-sampled (each triangle split into four) and the next step repeats the test on them, until the triangles are
smaller than ``pixel_scale_precision``. The centroids of the final kept triangles are the images, after those with an
absolute magnification below ``magnification_threshold`` (e.g. the triangle at the centre of the lens) are rejected.

The figure shows steps 0, 1, 2 and the final step, every panel on the same fixed extent so the triangles visibly
shrink onto the images:

- **Triangles tested** (grey outlines): every triangle ray-traced at that step.
- **Kept** (blue): the triangles whose traced image contains the source.
- **Kept, then rejected** (hatched, muted blue): kept triangles that converge onto a position the magnification
  threshold rejects.
- **Neighbours** (orange): the rim added around the kept triangles before up-sampling.
- **Up-sampled** (green outlines): the triangles tested at the next step.
- the critical curve, the observed positions (hollow circles) and the solved images (red crosses).

Panel 0 numbers the images and labels a rejected candidate. Kept triangles too small to draw at the panel's
size (below ~3 screen pixels) are drawn as a triangle marker at their centroid instead.

One per-panel draw function, :func:`draw_point_solver_panel`, is shared by the 2x2 still (``point_solver.png``) and
the teaching animation (``point_solver.gif``, one panel per frame), so the two cannot drift apart.

The figure is NumPy-only: the steps are re-solved with ``xp=numpy`` even for a solver built with ``use_jax=True``,
because the JAX path pads every step to ``MAX_CONTAINING_SIZE`` triangles and, inside ``jax.jit``, hides the steps
behind the implicit-differentiation rule.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TESTED_COLOR = "#8a8f95"
NEIGHBOUR_COLOR = "#f2a33a"
KEPT_COLOR = "#1f6fd1"
REJECTED_COLOR = "#9db7d9"
UP_SAMPLED_COLOR = "#2e9e5b"
CRITICAL_CURVE_COLOR = "#7b3fa0"
OBSERVED_COLOR = "k"
SOLVED_COLOR = "r"

# A kept triangle whose edge spans fewer typographic points (1/72 inch) than this on the panel is drawn as a
# centroid marker: about 3 screen pixels, independent of the dpi the figure is saved at.
MIN_TRIANGLE_POINTS = 2.5

# Above this many solved images (e.g. a cluster) step 0 is not numbered, so the panel stays readable.
MAX_NUMBERED_IMAGES = 10

# A step-0 tiling with more triangles than this is drawn with hairline outlines.
DENSE_TILING = 5000

# Tested triangles whose edge spans fewer points than this (e.g. a cluster's step-0 tiling of hundreds of thousands
# of triangles) merge into a solid slab when outlined, so they are drawn as a light grey shade instead.
MIN_OUTLINE_POINTS = 1.5

# At most this many rejected candidates get an arrow from the step-0 label.
MAX_REJECTED_ARROWS = 5

# Seconds each animation frame is shown; the final step is held for ``FINAL_FRAME_REPEATS`` frames.
FRAME_SECONDS = 1.25
FINAL_FRAME_REPEATS = 3


@dataclass
class PointSolverFigureData:
    """
    Everything the point-solver figure draws, collected once with NumPy and shared by every panel and frame.

    Attributes
    ----------
    steps
        The selected refinement steps (steps 0, 1, 2 and the final step, or every step when there are fewer).
    n_steps
        The total number of refinement steps the solver took.
    solved
        The (y,x) solved image positions, ordered by angle about the origin so the numbering is stable.
    rejected
        The (y,x) centroids of the final kept triangles the magnification threshold rejects.
    final_centres
        The (y,x) centroids of every final kept triangle (accepted and rejected).
    magnification_threshold
        The solver's ``magnification_threshold``, quoted in the legend.
    extent
        The ``[x_min, x_max, y_min, y_max]`` extent shared by every panel.
    lines
        The critical-curve lines, each a (N, 2) array of (y,x) coordinates.
    positions
        The (y,x) observed positions, or ``None``.
    max_kept
        The largest number of triangles kept at any step.
    rejected_at_lens_centre
        Whether the first rejected candidate lies at the centre of a mass profile, which sets its step-0 label.
    """

    steps: list
    n_steps: int
    solved: np.ndarray
    rejected: np.ndarray
    final_centres: np.ndarray
    magnification_threshold: float
    extent: List[float]
    lines: List[np.ndarray]
    positions: Optional[np.ndarray]
    max_kept: int
    rejected_at_lens_centre: bool = False


def _yx(grid) -> np.ndarray:
    grid = grid.array if hasattr(grid, "array") else grid
    grid = np.asarray(grid, dtype=float).reshape(-1, 2)
    return grid[np.isfinite(grid).all(axis=1)]


def triangles_xy_from(triangles) -> np.ndarray:
    """
    Returns the finite triangles of a triangles container as a (N, 3, 2) array of (x,y) vertices, ready for a
    matplotlib ``PolyCollection``.

    The solver stores vertices as (y,x), so they are swapped here (the retired ``point/visualise.py`` plotted them
    unswapped, transposing every figure). NaN-padded rows (the JAX containers' fixed-size padding) are dropped.
    """
    array = np.asarray(
        triangles.triangles if hasattr(triangles, "triangles") else triangles,
        dtype=float,
    ).reshape(-1, 3, 2)
    array = array[np.isfinite(array).all(axis=(1, 2))]
    return array[:, :, ::-1]


def _centroid_keys(triangles_xy: np.ndarray) -> set:
    return {tuple(c) for c in np.round(triangles_xy.mean(axis=1), 7)}


def neighbour_rim_from(step) -> np.ndarray:
    """
    Returns the (x,y) neighbourhood triangles of a step which are not themselves kept: the rim of neighbours added
    around the kept triangles before up-sampling.
    """
    kept = _centroid_keys(triangles_xy_from(step.filtered_triangles))
    neighbourhood = triangles_xy_from(step.neighbourhood)
    if len(neighbourhood) == 0:
        return neighbourhood
    keys = np.round(neighbourhood.mean(axis=1), 7)
    return neighbourhood[[tuple(k) not in kept for k in keys]]


def split_kept_from(
    step, final_centres: np.ndarray, rejected: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a step's kept triangles into those converging onto an accepted image and those converging onto a
    position the magnification threshold rejects, returned as two (N, 3, 2) arrays of (x,y) vertices.

    Each kept triangle is assigned to its nearest final kept centroid; it is "kept, then rejected" when that
    centroid is one the threshold rejects.
    """
    kept = triangles_xy_from(step.filtered_triangles)
    if len(kept) == 0 or len(rejected) == 0:
        return kept, kept[:0]

    centres = kept.mean(axis=1)[:, ::-1]  # (y,x)
    to_final = np.linalg.norm(centres[:, None, :] - final_centres[None], axis=2).min(
        axis=1
    )
    to_rejected = np.linalg.norm(centres[:, None, :] - rejected[None], axis=2).min(
        axis=1
    )
    is_rejected = np.isclose(to_rejected, to_final)
    return kept[~is_rejected], kept[is_rejected]


def select_steps(steps: list) -> list:
    """
    Returns the steps drawn by the figure: steps 0, 1, 2 and the final step, or every step when there are four or
    fewer. On a fine grid steps 1, 2 and final can look alike, as the kept triangles are already sub-pixel.
    """
    indexes = sorted({i for i in (0, 1, 2, len(steps) - 1) if 0 <= i < len(steps)})
    return [steps[i] for i in indexes]


def _extent_from(solver, positions) -> List[float]:
    from autolens.point.plot.positions_panel import (
        positions_extent_from,
        solver_extent_from,
    )

    if positions is not None and len(positions):
        extent = positions_extent_from(positions=positions, solver=solver)
    else:
        extent = solver_extent_from(solver)
    if extent is None:
        raise ValueError(
            "The point-solver figure needs a solver with finite rectangular bounds (y_min, y_max, x_min, x_max)."
        )
    pad = 0.02 * max(extent[1] - extent[0], extent[3] - extent[2])
    return [extent[0] - pad, extent[1] + pad, extent[2] - pad, extent[3] + pad]


def _critical_curve_lines_from(tracer, extent) -> List[np.ndarray]:
    import autoarray as aa
    from autolens.lens.plot.critical_curves import _compute_critical_curve_lines

    x_min, x_max, y_min, y_max = extent
    grid = aa.Grid2D.from_extent(
        extent=(x_min, x_max, y_min, y_max), shape_native=(100, 100)
    )
    lines, _, _, _ = _compute_critical_curve_lines(tracer, grid)
    return [np.asarray(line) for line in (lines or [])]


def point_solver_data_from(
    solver,
    tracer,
    source_plane_coordinate: Tuple[float, float],
    positions=None,
    plane_redshift: Optional[float] = None,
    image_plane_lines=None,
) -> PointSolverFigureData:
    """
    Re-solves the point source with NumPy and collects everything the point-solver figure draws.

    Parameters
    ----------
    solver
        The ``PointSolver`` whose refinement is shown. A solver built with ``use_jax=True`` is re-run with NumPy.
    tracer
        The tracer that ray-traces the triangles to the source plane.
    source_plane_coordinate
        The (y,x) source-plane coordinate of the point source.
    positions
        The observed (y,x) positions, drawn as hollow circles; ``None`` draws none.
    plane_redshift
        The redshift of the plane the source is in, for multi-plane lenses (e.g. clusters). ``None`` is the last
        plane.
    image_plane_lines
        Pre-computed critical-curve lines. ``None`` computes them on a grid covering the panel; pass ``[]`` to draw
        none.
    """
    from autoarray.structures.triangles.shape import Point

    beta = tuple(float(v) for v in np.asarray(source_plane_coordinate).ravel()[:2])

    steps = list(
        solver.steps(
            tracer=tracer, shape=Point(*beta), xp=np, plane_redshift=plane_redshift
        )
    )
    if len(steps) == 0:
        raise ValueError(
            "The solver took no refinement steps, so there is nothing to plot."
        )

    final_centres = triangles_xy_from(steps[-1].filtered_triangles).mean(axis=1)[
        :, ::-1
    ]
    if len(final_centres):
        filtered = np.asarray(
            solver._filter_low_magnification(
                tracer=tracer,
                points=final_centres,
                xp=np,
                plane_redshift=plane_redshift,
            )
        ).reshape(-1, 2)
        rejected = final_centres[~np.isfinite(filtered).all(axis=1)]
    else:
        final_centres = np.zeros((0, 2))
        rejected = np.zeros((0, 2))

    solved = _yx(
        solver.solve(
            tracer=tracer,
            source_plane_coordinate=beta,
            xp=np,
            plane_redshift=plane_redshift,
        )
    )
    solved = solved[np.argsort(-np.arctan2(solved[:, 0], solved[:, 1]))]

    positions = None if positions is None else _yx(positions)
    extent = _extent_from(solver=solver, positions=positions)

    if image_plane_lines is None:
        lines = _critical_curve_lines_from(tracer, extent)
    else:
        lines = [np.asarray(line) for line in image_plane_lines]

    rejected_at_lens_centre = False
    if len(rejected):
        from autolens.point.plot.positions_panel import mass_centres_from

        centres = mass_centres_from(tracer)
        if len(centres):
            rejected_at_lens_centre = bool(
                np.linalg.norm(centres - rejected[0], axis=1).min()
                <= 2.0 * float(solver.scale)
            )

    return PointSolverFigureData(
        steps=select_steps(steps),
        n_steps=len(steps),
        solved=solved,
        rejected=rejected,
        final_centres=final_centres,
        magnification_threshold=float(solver.magnification_threshold),
        extent=extent,
        lines=lines,
        positions=positions,
        max_kept=max(len(triangles_xy_from(s.filtered_triangles)) for s in steps),
        rejected_at_lens_centre=rejected_at_lens_centre,
    )


def _edge_of(step) -> float:
    tested = triangles_xy_from(step.initial_triangles)
    if len(tested) == 0:
        return 0.0
    return float(max(np.ptp(tested[0][:, 0]), np.ptp(tested[0][:, 1])))


def is_too_small_to_draw(
    step, data: PointSolverFigureData, panel_inches: float
) -> bool:
    """
    Returns whether a step's triangles span fewer than ``MIN_TRIANGLE_POINTS`` points on a panel
    ``panel_inches`` wide, in which case its kept triangles are drawn as centroid markers.
    """
    width = max(data.extent[1] - data.extent[0], data.extent[3] - data.extent[2])
    return _edge_of(step) / width * panel_inches * 72.0 < MIN_TRIANGLE_POINTS


def panel_title_from(step, data: PointSolverFigureData, short: bool = False) -> str:
    """
    Returns a panel title, e.g. ``Step 0 (273 tested, 8 kept + 1 rejected)`` or ``Step 5 — final (...)``; the
    ``short`` form (used by the animation) omits the counts.
    """
    final = step.number == data.n_steps - 1
    title = f"Step {step.number}" + (" — final" if final else "")
    if short:
        return title
    kept, rejected = split_kept_from(step, data.final_centres, data.rejected)
    counts = (
        f"{len(triangles_xy_from(step.initial_triangles)):,} tested, {len(kept)} kept"
    )
    if len(rejected):
        counts += f" + {len(rejected)} rejected"
    return f"{title} ({counts})"


# Candidate placements of an image number relative to its image, as (dx, dy) offsets in points plus the text's
# horizontal and vertical alignment: the number's nearest corner sits a few points off the image marker, upper-left
# first (the teaching figure's look), so a number is always immediately adjacent to its own image.
_NUMBER_PLACEMENTS = (
    (-6, 5, "right", "bottom"),
    (6, 5, "left", "bottom"),
    (-6, -5, "right", "top"),
    (6, -5, "left", "top"),
)
NUMBER_FONTSIZE = 14

# Radius (points) treated as occupied around an image marker when placing labels.
_MARKER_RADIUS_POINTS = 6.0

# Candidate placements of the rejected-candidate label next to a single candidate, as (x, y) offsets in points and
# horizontal alignment.
_REJECTED_LABEL_PLACEMENTS = (
    (16, -52, "left"),
    (-16, -52, "right"),
    (0, -58, "center"),
    (16, 40, "left"),
    (-16, 40, "right"),
    (0, 46, "center"),
)

# Corners (axes fraction, ha, va) the rejected-candidate label falls back to, with one arrow per candidate.
_REJECTED_LABEL_CORNERS = (
    (0.02, 0.98, "left", "top"),
    (0.98, 0.98, "right", "top"),
    (0.02, 0.02, "left", "bottom"),
    (0.98, 0.02, "right", "bottom"),
)


def _points_per_arcsec(data: PointSolverFigureData, panel_inches: float) -> float:
    width = max(data.extent[1] - data.extent[0], data.extent[3] - data.extent[2])
    return panel_inches * 72.0 / width


def _to_points(data: PointSolverFigureData, xy: np.ndarray, panel_inches: float):
    """(x,y) arc-seconds -> (x,y) points from the panel's lower-left corner."""
    scale = _points_per_arcsec(data, panel_inches)
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return (xy - np.array([data.extent[0], data.extent[2]])) * scale


def _box_overlaps_circles(box, centres, radius) -> np.ndarray:
    x0, y0, x1, y1 = box
    if len(centres) == 0:
        return np.zeros(0, dtype=bool)
    gap_x = np.maximum(np.maximum(x0 - centres[:, 0], centres[:, 0] - x1), 0.0)
    gap_y = np.maximum(np.maximum(y0 - centres[:, 1], centres[:, 1] - y1), 0.0)
    return np.hypot(gap_x, gap_y) < radius


def _boxes_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _panel_points(data: PointSolverFigureData, panel_inches: float):
    scale = _points_per_arcsec(data, panel_inches)
    return (
        (data.extent[1] - data.extent[0]) * scale,
        (data.extent[3] - data.extent[2]) * scale,
    )


def number_placements_from(
    data: PointSolverFigureData, panel_inches: float
) -> List[Tuple[float, float, str, str]]:
    """
    Returns, for every solved image, the ``(dx, dy, ha, va)`` offset (points) of its step-0 number.

    Each number sits a few points off its own image (one of ``_NUMBER_PLACEMENTS``): the first placement which
    stays inside the panel and overlaps neither another image marker nor an already-placed number, else the one
    overlapping least. A number is therefore never moved away from its image to find space.
    """
    markers = _to_points(
        data,
        np.vstack([data.solved[:, ::-1], data.rejected[:, ::-1]]),
        panel_inches,
    )
    width, height = _panel_points(data, panel_inches)
    placed_boxes, placements = [], []
    for index, (x, y) in enumerate(markers[: len(data.solved)]):
        text_width = 0.65 * NUMBER_FONTSIZE * len(str(index + 1)) + 2.0
        text_height = NUMBER_FONTSIZE + 2.0
        others = np.delete(markers, index, axis=0)
        best, best_score = _NUMBER_PLACEMENTS[0], np.inf
        for dx, dy, ha, va in _NUMBER_PLACEMENTS:
            x0 = x + dx - (text_width if ha == "right" else 0.0)
            y0 = y + dy - (text_height if va == "top" else 0.0)
            box = (x0, y0, x0 + text_width, y0 + text_height)
            inside = (
                box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height
            )
            score = (
                int(_box_overlaps_circles(box, others, _MARKER_RADIUS_POINTS).sum())
                + sum(_boxes_overlap(box, other) for other in placed_boxes)
                + (0 if inside else 100)
            )
            if score < best_score:
                best, best_score, best_box = (dx, dy, ha, va), score, box
            if score == 0:
                break
        placements.append(best)
        placed_boxes.append(best_box)
    return placements


def _rejected_label_placement(
    data: PointSolverFigureData, x, y, panel_inches: float, fontsize: float
) -> Optional[Tuple[float, float, str]]:
    """
    Returns the offset and alignment of the label of a single rejected candidate placed next to it: the first
    candidate placement which lies inside the panel and clear of every image and image number, or ``None`` if none
    is clear (the label then goes in a corner).
    """
    x_min, x_max, y_min, y_max = data.extent
    arcsec_per_point = max(x_max - x_min, y_max - y_min) / (panel_inches * 72.0)
    half_width = (
        0.5 * 0.55 * fontsize * 24 * arcsec_per_point
    )  # ~24 characters per line
    half_height = 2.1 * fontsize * arcsec_per_point  # three lines

    obstacles = data.solved[:, ::-1]  # (x,y)
    if 0 < len(data.solved) <= MAX_NUMBERED_IMAGES:
        numbers = obstacles + np.array([-15.0, 12.0]) * arcsec_per_point
        obstacles = np.vstack([obstacles, numbers])

    clear = 10.0 * arcsec_per_point
    for dx, dy, ha in _REJECTED_LABEL_PLACEMENTS:
        centre_x = (
            x
            + dx * arcsec_per_point
            + {"left": 1, "right": -1, "center": 0}[ha] * half_width
        )
        centre_y = y + dy * arcsec_per_point
        inside = (
            centre_x - half_width >= x_min
            and centre_x + half_width <= x_max
            and centre_y - half_height >= y_min
            and centre_y + half_height <= y_max
        )
        if len(obstacles):
            gap_x = np.maximum(np.abs(obstacles[:, 0] - centre_x) - half_width, 0.0)
            gap_y = np.maximum(np.abs(obstacles[:, 1] - centre_y) - half_height, 0.0)
            clearance = float(np.hypot(gap_x, gap_y).min())
        else:
            clearance = np.inf
        if inside and clearance >= clear:
            return dx, dy, ha
    return None


def _segment_hits(start, end, centres, radius) -> int:
    """Number of circles of ``radius`` around ``centres`` that the segment start -> end passes through."""
    if len(centres) == 0:
        return 0
    start, end = np.asarray(start, float), np.asarray(end, float)
    direction = end - start
    length2 = max(float(direction @ direction), 1e-12)
    t = np.clip(((centres - start) @ direction) / length2, 0.0, 1.0)
    closest = start + t[:, None] * direction
    return int((np.linalg.norm(centres - closest, axis=1) < radius).sum())


def rejected_corner_from(
    data: PointSolverFigureData, panel_inches: float, fontsize: float
) -> Tuple[float, float, str, str]:
    """
    Returns the corner ``(x, y, ha, va)`` (axes fraction) of the rejected-candidates label: the corner whose text
    box covers no image and whose arrows to the candidates cross the fewest image markers, then are shortest.
    """
    width, height = _panel_points(data, panel_inches)
    images = _to_points(data, data.solved[:, ::-1], panel_inches)
    targets = _to_points(data, data.rejected[:MAX_REJECTED_ARROWS, ::-1], panel_inches)
    box_width, box_height = 0.55 * fontsize * 24, 3.6 * fontsize

    best, best_score = _REJECTED_LABEL_CORNERS[0], np.inf
    for fx, fy, ha, va in _REJECTED_LABEL_CORNERS:
        x0 = fx * width - (box_width if ha == "right" else 0.0)
        y0 = fy * height - (box_height if va == "top" else 0.0)
        box = (x0, y0, x0 + box_width, y0 + box_height)
        centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        covered = int(
            _box_overlaps_circles(box, images, _MARKER_RADIUS_POINTS + 4.0).sum()
        )
        crossings, length = 0, 0.0
        for target in targets:
            others = images[np.linalg.norm(images - target, axis=1) > 1e-6]
            crossings += _segment_hits(centre, target, others, _MARKER_RADIUS_POINTS)
            length += float(np.hypot(*(np.asarray(target) - centre)))
        score = 1e6 * covered + 1e4 * crossings + length
        if score < best_score:
            best, best_score = (fx, fy, ha, va), score
    return best


def _annotate_images(
    ax,
    data: PointSolverFigureData,
    panel_inches: float,
    annotation_fontsize: float = 10,
):
    if len(data.solved) <= MAX_NUMBERED_IMAGES:
        placements = number_placements_from(data, panel_inches)
        for number, ((y, x), (dx, dy, ha, va)) in enumerate(
            zip(data.solved, placements), start=1
        ):
            ax.annotate(
                f"{number}",
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=NUMBER_FONTSIZE,
                weight="bold",
                zorder=10,
                bbox=dict(fc="w", ec="none", alpha=0.8, pad=0.5),
            )
    if len(data.rejected) == 0:
        return

    y, x = data.rejected[0]
    where = "at the lens centre" if data.rejected_at_lens_centre else "here"
    near = (
        _rejected_label_placement(
            data, x, y, panel_inches=panel_inches, fontsize=annotation_fontsize
        )
        if len(data.rejected) == 1
        else None
    )
    if near is not None:
        dx, dy, ha = near
        ax.annotate(
            f"kept {where},\nrejected by\nmagnification_threshold",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=annotation_fontsize,
            arrowprops=dict(arrowstyle="->", lw=0.8),
            zorder=10,
            bbox=dict(fc="w", ec="none", alpha=0.85, pad=1),
        )
        return

    # No clear spot next to the candidate (or several candidates): the label goes in the emptiest corner, with
    # one arrow per candidate drawn from it.
    text = (
        f"kept {where},\nrejected by\nmagnification_threshold"
        if len(data.rejected) == 1
        else f"{len(data.rejected)} candidates kept,\nthen rejected by\nmagnification_threshold"
    )
    fx, fy, ha, va = rejected_corner_from(data, panel_inches, annotation_fontsize)
    label = ax.text(
        fx,
        fy,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=annotation_fontsize,
        zorder=10,
        bbox=dict(fc="w", ec="0.6", lw=0.6, alpha=0.95, pad=3),
    )
    for other_y, other_x in data.rejected[:MAX_REJECTED_ARROWS]:
        ax.annotate(
            "",
            (other_x, other_y),
            xytext=(0.5, 0.5),
            textcoords=label,
            arrowprops=dict(arrowstyle="->", lw=0.8, shrinkB=3),
            zorder=9.5,
        )


def draw_point_solver_panel(
    ax,
    step,
    data: PointSolverFigureData,
    panel_inches: float,
    title: Optional[str] = None,
    tick_fontsize: float = 10,
    title_fontsize: float = 12,
    annotation_fontsize: float = 10,
) -> bool:
    """
    Draws one refinement step on ``ax``: the panel shared by the 2x2 still and every animation frame.

    Step 0 is drawn as an overview (no up-sampled triangles, hairline outlines on a dense tiling) with the images
    numbered and a rejected candidate labelled. Every panel uses the same fixed extent, with standard arc-second
    tick labels and no axis-label text.

    Returns ``True`` if the kept triangles were drawn as centroid markers (too small to draw), so the caller can
    add the matching legend entry.
    """
    from matplotlib.collections import PolyCollection

    from autoarray.plot.utils import apply_extent

    overview = step.number == 0
    tested = triangles_xy_from(step.initial_triangles)

    width = max(data.extent[1] - data.extent[0], data.extent[3] - data.extent[2])
    if _edge_of(step) / width * panel_inches * 72.0 < MIN_OUTLINE_POINTS:
        ax.add_collection(
            PolyCollection(
                tested,
                facecolor=TESTED_COLOR,
                alpha=0.18,
                edgecolor="none",
                zorder=1,
            )
        )
    else:
        ax.add_collection(
            PolyCollection(
                tested,
                facecolor="none",
                edgecolor=TESTED_COLOR,
                lw=0.2 if overview and len(tested) > DENSE_TILING else 0.56,
                zorder=1,
            )
        )
    ax.add_collection(
        PolyCollection(
            neighbour_rim_from(step),
            facecolor=NEIGHBOUR_COLOR,
            alpha=0.5,
            edgecolor=NEIGHBOUR_COLOR,
            lw=0.48,
            zorder=2,
        )
    )

    kept, rejected = split_kept_from(step, data.final_centres, data.rejected)
    as_markers = is_too_small_to_draw(step, data, panel_inches)
    if as_markers:
        for triangles, color in ((kept, KEPT_COLOR), (rejected, REJECTED_COLOR)):
            if len(triangles):
                centres = triangles.mean(axis=1)
                ax.scatter(
                    centres[:, 0],
                    centres[:, 1],
                    s=34,
                    marker="^",
                    c=color,
                    edgecolors=KEPT_COLOR,
                    linewidths=0.6,
                    zorder=9,
                )
    else:
        ax.add_collection(
            PolyCollection(
                kept,
                facecolor=KEPT_COLOR,
                alpha=0.85,
                edgecolor=KEPT_COLOR,
                lw=0.64,
                zorder=3,
            )
        )
        ax.add_collection(
            PolyCollection(
                rejected,
                facecolor=REJECTED_COLOR,
                alpha=0.85,
                edgecolor=KEPT_COLOR,
                hatch="////",
                lw=0.64,
                zorder=3,
            )
        )

    if not overview:
        ax.add_collection(
            PolyCollection(
                triangles_xy_from(step.up_sampled),
                facecolor="none",
                edgecolor=UP_SAMPLED_COLOR,
                lw=0.4,
                zorder=4,
            )
        )

    for line in data.lines:
        line = np.asarray(line)
        if line.ndim == 2 and len(line):
            ax.plot(
                line[:, 1], line[:, 0], color=CRITICAL_CURVE_COLOR, lw=1.0, zorder=5
            )

    if data.positions is not None and len(data.positions):
        ax.scatter(
            data.positions[:, 1],
            data.positions[:, 0],
            s=60,
            facecolors="none",
            edgecolors=OBSERVED_COLOR,
            linewidths=1.3,
            zorder=7,
        )
    if len(data.solved):
        ax.scatter(
            data.solved[:, 1],
            data.solved[:, 0],
            s=45,
            c=SOLVED_COLOR,
            marker="x",
            linewidths=1.4,
            zorder=8,
        )

    if overview:
        _annotate_images(
            ax, data, panel_inches=panel_inches, annotation_fontsize=annotation_fontsize
        )

    apply_extent(ax, data.extent)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelrotation=90)
    ax.set_title(
        panel_title_from(step, data) if title is None else title,
        fontsize=title_fontsize,
    )
    return as_markers


def legend_handles_from(data: PointSolverFigureData, marked: bool) -> list:
    """
    Returns the legend handles shared by the still and the animation; the ``Kept (too small to draw)`` entry is
    included only when a panel drew its kept triangles as centroid markers.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(fc="none", ec=TESTED_COLOR, label="Triangles tested"),
        Patch(fc=KEPT_COLOR, alpha=0.85, label="Kept (contains source)"),
        Patch(
            fc=REJECTED_COLOR,
            ec=KEPT_COLOR,
            hatch="////",
            label=f"Kept, then rejected (μ < {data.magnification_threshold:g})",
        ),
        Patch(fc=NEIGHBOUR_COLOR, alpha=0.5, label="Neighbours"),
        Patch(fc="none", ec=UP_SAMPLED_COLOR, label="Up-sampled (next step)"),
    ]
    if marked:
        handles.append(
            Line2D(
                [],
                [],
                marker="^",
                ls="",
                color=KEPT_COLOR,
                label="Kept (too small to draw)",
            )
        )
    handles.append(Line2D([], [], color=CRITICAL_CURVE_COLOR, label="Critical curve"))
    if data.positions is not None and len(data.positions):
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                mfc="none",
                mec=OBSERVED_COLOR,
                label="Observed",
            )
        )
    handles.append(
        Line2D([], [], marker="x", ls="", color=SOLVED_COLOR, label="Solved")
    )
    return handles


def warn_if_near_jax_cap(data: PointSolverFigureData, fraction: float = 0.8):
    """
    Logs a warning when a step keeps nearly as many triangles as the JAX solver's static ``MAX_CONTAINING_SIZE``.

    The JAX path truncates every step to that many kept triangles, so a model which keeps more (many images, or a
    source near a caustic) can lose images under JAX that this NumPy re-solve still shows.
    """
    from autoarray.structures.triangles.array import MAX_CONTAINING_SIZE

    if data.max_kept >= fraction * MAX_CONTAINING_SIZE:
        logger.warning(
            f"PointSolver figure: a refinement step kept {data.max_kept} triangles, close to (or above) the JAX "
            f"solver's MAX_CONTAINING_SIZE ({MAX_CONTAINING_SIZE}). The JAX path truncates each step to "
            f"{MAX_CONTAINING_SIZE} kept triangles, so a JAX model-fit may lose images this NumPy figure shows."
        )


def _resolve_format(output_format) -> str:
    from autoarray.plot.utils import _conf_output_format

    if output_format is None:
        output_format = _conf_output_format()
    if isinstance(output_format, (list, tuple)):
        output_format = output_format[0] if output_format else "show"
    return output_format


def _save_animation(data, output_path, filename, title_prefix, dpi=100):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig_width, fig_height = 6.4, 7.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.2)
    panel_inches = 0.89 * fig_width

    marked = any(is_too_small_to_draw(s, data, panel_inches) for s in data.steps)
    fig.legend(
        handles=legend_handles_from(data, marked),
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        frameon=False,
    )

    prefix = f"{title_prefix.rstrip()} " if title_prefix else ""
    last = len(data.steps) - 1
    frames = list(range(last)) + [last] * FINAL_FRAME_REPEATS

    def update(frame):
        ax.clear()
        step = data.steps[frame]
        draw_point_solver_panel(
            ax,
            step,
            data,
            panel_inches=panel_inches,
            title=prefix + panel_title_from(step, data, short=True),
            tick_fontsize=9,
            annotation_fontsize=9,
        )

    animation = FuncAnimation(fig, update, frames=frames)

    if output_path is None:
        return animation

    from autoarray.plot.utils import _FAST_PLOTS

    if not _FAST_PLOTS:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        animation.save(
            str(Path(output_path) / f"{filename}.gif"),
            writer=PillowWriter(fps=1.0 / FRAME_SECONDS),
            dpi=dpi,
        )
    plt.close(fig)
    return None


def plot_point_solver(
    solver,
    tracer,
    source_plane_coordinate: Tuple[float, float],
    positions=None,
    plane_redshift: Optional[float] = None,
    image_plane_lines=None,
    output_path: Optional[str] = None,
    output_format: str = None,
    title_prefix: str = None,
    animate: bool = False,
    filename: str = "point_solver",
):
    """
    Plot how the ``PointSolver`` refines its triangle tiling to find the multiple images of a point source.

    The figure (``point_solver.png``) is a 2x2 subplot of steps 0, 1, 2 and the final step on one fixed extent: the
    triangles tested, kept (blue), kept then rejected by ``magnification_threshold`` (hatched), the neighbour rim
    (orange) and the triangles up-sampled for the next step (green), with the critical curve, the observed
    positions and the solved images. See the module docstring for the full legend.

    With ``animate=True`` the same panels are written as a looping GIF (``point_solver.gif``), one step per frame
    (``FRAME_SECONDS`` each, the final step held longer), via ``matplotlib.animation`` and ``PillowWriter`` (no
    ffmpeg needed). Without an ``output_path`` the ``FuncAnimation`` is returned instead, e.g. for display in a
    notebook.

    The solve is repeated with NumPy whatever the solver's backend, as the JAX path pads and hides the steps.

    Parameters
    ----------
    solver
        The ``PointSolver`` whose refinement is shown.
    tracer
        The tracer that ray-traces the triangles to the source plane.
    source_plane_coordinate
        The (y,x) source-plane coordinate of the point source.
    positions
        The observed (y,x) positions, drawn as hollow circles.
    plane_redshift
        The redshift of the source's plane, for multi-plane lenses (e.g. clusters). ``None`` is the last plane.
    image_plane_lines
        Pre-computed critical-curve lines; ``None`` computes them, ``[]`` draws none.
    output_path
        Directory in which to save the figure. If ``None`` the still is shown (and the animation returned).
    output_format
        Image format of the still, passed to :func:`~autoarray.plot.utils.save_figure`. The animation is always
        a GIF.
    title_prefix
        Text prepended to every panel title.
    animate
        If ``True`` write the GIF animation instead of the still.
    filename
        The file name (without extension) of the output.
    """
    data = point_solver_data_from(
        solver=solver,
        tracer=tracer,
        source_plane_coordinate=source_plane_coordinate,
        positions=positions,
        plane_redshift=plane_redshift,
        image_plane_lines=image_plane_lines,
    )
    return plot_point_solver_from_data(
        data=data,
        output_path=output_path,
        output_format=output_format,
        title_prefix=title_prefix,
        animate=animate,
        filename=filename,
    )


def plot_point_solver_from_data(
    data: PointSolverFigureData,
    output_path: Optional[str] = None,
    output_format: str = None,
    title_prefix: str = None,
    animate: bool = False,
    filename: str = "point_solver",
):
    """
    Draws the point-solver still or animation from data already collected by :func:`point_solver_data_from`, so
    the still and the GIF can be written from one solve.
    """
    from autoarray.plot.utils import hide_unused_axes, save_figure, subplots

    if animate:
        return _save_animation(
            data=data,
            output_path=None if output_path is None else str(output_path),
            filename=filename,
            title_prefix=title_prefix,
        )

    n = len(data.steps)
    rows, cols = (2, 2) if n > 2 else (1, n)
    fig_width, fig_height = 6.0 * cols, 6.0 * rows + 0.6
    fig, axes = subplots(rows, cols, figsize=(fig_width, fig_height))
    axes_flat = list(np.array(axes).flatten())

    output_format = _resolve_format(output_format)
    panel_inches = 0.85 * fig_width / cols

    prefix = f"{title_prefix.rstrip()} " if title_prefix else ""
    marked = False
    for ax, step in zip(axes_flat, data.steps):
        marked |= draw_point_solver_panel(
            ax,
            step,
            data,
            panel_inches=panel_inches,
            title=prefix + panel_title_from(step, data),
        )
    hide_unused_axes(axes_flat[n:])

    fig.legend(
        handles=legend_handles_from(data, marked),
        loc="lower center",
        ncol=5,
        fontsize=10.5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.6 / fig_height + 0.03, 1, 1), h_pad=2.0)
    save_figure(
        fig,
        path=None if output_path is None else str(output_path),
        filename=filename,
        format=output_format,
    )

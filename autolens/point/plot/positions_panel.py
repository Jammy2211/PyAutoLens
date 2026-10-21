"""
Shared drawing of the point-source "Positions" panels of ``dataset_point.png`` and ``fit.png``.

Both panels are drawn on the same frame:

- **Extent.** The extent of the ``PointSolver`` grid when a solver is available, so the panel shows the region in
  which the model can find images. Without a solver it falls back to the critical-curve region of
  :func:`~autolens.lens.plot.critical_curves.critical_curve_grid_from` (a square of twice the half-span of the
  positions, plus a buffer), which contains every position.
- **Markers.** Large filled markers for the observed positions, with a circle of radius equal to each position's
  noise-map value (one isotropic sigma per position) and the sigma always stated in the legend. The circle is drawn
  at its true radius and is never inflated, so at high-resolution noise and a large solver extent it is sub-pixel.
- **No axis labels**, so the arc-second tick labels are not overlapped.

The fit panel additionally draws the model positions, the residual lines of the fit's own pairing, unmatched model
images, critical curves and the centres of the tracer's mass profiles.
"""

import numpy as np
from typing import List, Optional, Tuple

OBSERVED_COLOR = "k"
NOISE_COLOR = "tab:orange"
MODEL_COLOR = "r"
UNMATCHED_COLOR = "0.45"
MASS_CENTRE_COLOR = "tab:green"

# The model cross is thinner and smaller than the observed marker (``s=110``), so under an exact fit the observed
# dot stays visible around it rather than being hidden.
MODEL_MARKER_SIZE = 60
MODEL_MARKER_LINEWIDTH = 1.2

# Critical-curve colours: `_compute_critical_curve_lines` returns white (tangential) and yellow (radial) lines for
# dark image backgrounds, which are invisible on the white positions panel, so they are mapped to light colours
# here that stay readable without swamping the residual lines.
CRITICAL_CURVE_COLOR_MAP = {"white": "0.6", "yellow": "tab:olive"}


def _np_grid(grid) -> np.ndarray:
    """
    Returns a (N, 2) numpy array of (y,x) coordinates from a grid-like input.
    """
    grid = grid.array if hasattr(grid, "array") else grid
    return np.asarray(grid, dtype=float).reshape(-1, 2)


def solver_extent_from(solver) -> Optional[List[float]]:
    """
    Returns the ``[x_min, x_max, y_min, y_max]`` extent of a ``PointSolver``'s grid, or ``None`` if the solver is
    ``None`` or does not have finite, rectangular bounds (e.g. a custom solver).
    """
    if solver is None:
        return None
    try:
        y_min, y_max = float(solver.y_min), float(solver.y_max)
        x_min, x_max = float(solver.x_min), float(solver.x_max)
    except (AttributeError, TypeError, ValueError):
        return None
    bounds = np.array([x_min, x_max, y_min, y_max])
    if not np.all(np.isfinite(bounds)) or x_max <= x_min or y_max <= y_min:
        return None
    return [x_min, x_max, y_min, y_max]


def fallback_extent_from(
    positions, scale: float = 2.0, buffer: float = 0.1
) -> List[float]:
    """
    Returns the ``[x_min, x_max, y_min, y_max]`` extent used when no solver is available.

    This is the square of :func:`~autolens.lens.plot.critical_curves.critical_curve_grid_from` (centred on the
    bounding box of the positions, with half-width ``scale`` times their half-span plus ``buffer``), so the panel
    covers the region the critical curves are computed on and contains every position.
    """
    positions = _np_grid(positions)
    y_min, y_max = positions[:, 0].min(), positions[:, 0].max()
    x_min, x_max = positions[:, 1].min(), positions[:, 1].max()
    y_centre, x_centre = (y_min + y_max) / 2.0, (x_min + x_max) / 2.0
    half_width = scale * (max(y_max - y_min, x_max - x_min) / 2.0 + buffer)
    return [
        x_centre - half_width,
        x_centre + half_width,
        y_centre - half_width,
        y_centre + half_width,
    ]


def positions_extent_from(positions, solver=None) -> List[float]:
    """
    Returns the ``[x_min, x_max, y_min, y_max]`` extent of a positions panel.

    With a solver this is the solver grid's extent. If an observed position lies outside it (a solver grid too
    small for the data, which ``AnalysisPoint`` already warns about) the extent is enlarged just enough to contain
    that position with a margin, so no observed position is ever clipped. Without a solver it is
    :func:`fallback_extent_from`.
    """
    extent = solver_extent_from(solver)
    if extent is None:
        return fallback_extent_from(positions)

    positions = _np_grid(positions)
    margin = 0.05 * max(extent[1] - extent[0], extent[3] - extent[2])
    x_min, x_max, y_min, y_max = extent
    if np.any(positions[:, 1] < x_min) or np.any(positions[:, 1] > x_max):
        x_min = min(x_min, positions[:, 1].min() - margin)
        x_max = max(x_max, positions[:, 1].max() + margin)
    if np.any(positions[:, 0] < y_min) or np.any(positions[:, 0] > y_max):
        y_min = min(y_min, positions[:, 0].min() - margin)
        y_max = max(y_max, positions[:, 0].max() + margin)
    return [x_min, x_max, y_min, y_max]


def _format_arcsec(value: float) -> str:
    return f"{value:.3g}″"


def sigma_label_from(noise_map) -> str:
    """
    Returns the sigma text of the observed-positions legend entry, e.g. ``σ = 0.005″``, or ``σ = 0.005–0.01″``
    when the positions have different noise-map values.
    """
    noise = np.asarray(
        noise_map.array if hasattr(noise_map, "array") else noise_map, dtype=float
    ).ravel()
    noise = noise[np.isfinite(noise)]
    if noise.size == 0:
        return "σ = n/a"
    if np.allclose(noise, noise[0]):
        return f"σ = {_format_arcsec(noise[0])}"
    return f"σ = {noise.min():.3g}–{_format_arcsec(noise.max())}"


def plot_observed_positions(ax, positions, noise_map):
    """
    Draws the observed positions as large filled markers, each with a circle of radius equal to its noise-map
    value (never inflated), labelled ``Observed (σ = ...)`` in the legend.
    """
    from matplotlib.patches import Circle

    positions = _np_grid(positions)
    noise = np.broadcast_to(
        np.asarray(
            noise_map.array if hasattr(noise_map, "array") else noise_map, dtype=float
        ).ravel(),
        (len(positions),),
    )

    ax.scatter(
        positions[:, 1],
        positions[:, 0],
        s=110,
        c=OBSERVED_COLOR,
        marker="o",
        zorder=4,
        label=f"Observed ({sigma_label_from(noise_map)})",
    )
    for (y, x), sigma in zip(positions, noise):
        if np.isfinite(sigma) and sigma > 0:
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=sigma,
                    fill=False,
                    edgecolor=NOISE_COLOR,
                    linewidth=1.0,
                    zorder=5,
                )
            )


def plot_critical_curves(ax, lines, line_colors=None):
    """
    Draws critical curves as thin, light lines labelled ``Critical curve`` in the legend.

    The curves are computed on :func:`~autolens.lens.plot.critical_curves.critical_curve_grid_from` passed the
    solver, whose grid covers the whole panel, so a curve is never broken inside the panel: a curve which runs past
    the panel edge is cut cleanly by the axes frame, which tells the reader it extends beyond the solver grid.
    """
    if not lines:
        return
    labelled = False
    for i, line in enumerate(lines):
        line = np.asarray(line)
        if line.ndim != 2 or len(line) == 0:
            continue
        color = line_colors[i] if line_colors and i < len(line_colors) else "white"
        ax.plot(
            line[:, 1],
            line[:, 0],
            color=CRITICAL_CURVE_COLOR_MAP.get(color, color),
            linewidth=0.9,
            alpha=0.8,
            zorder=2,
            label=None if labelled else "Critical curve",
        )
        labelled = True


def mass_centres_from(tracer) -> np.ndarray:
    """
    Returns the (y,x) centres of every mass profile of every galaxy in the tracer, as a (N, 2) array with
    duplicates removed.

    External shear is skipped, as its ``centre`` has no physical meaning.
    """
    import autogalaxy as ag

    centres = []
    for galaxy in getattr(tracer, "galaxies", []):
        try:
            mass_profiles = galaxy.cls_list_from(cls=ag.mp.MassProfile)
        except AttributeError:
            continue
        for mass_profile in mass_profiles:
            if isinstance(mass_profile, ag.mp.ExternalShear):
                continue
            centre = getattr(mass_profile, "centre", None)
            if centre is None:
                continue
            try:
                centre = (float(centre[0]), float(centre[1]))
            except (TypeError, ValueError, IndexError):
                continue
            if np.all(np.isfinite(centre)) and centre not in centres:
                centres.append(centre)
    return np.asarray(centres, dtype=float).reshape(-1, 2)


def plot_mass_centres(ax, tracer):
    """
    Draws the centre of every mass profile in the tracer as a ``+`` labelled ``Mass centre`` in the legend.
    """
    centres = mass_centres_from(tracer)
    if len(centres) == 0:
        return
    ax.scatter(
        centres[:, 1],
        centres[:, 0],
        s=140,
        c=MASS_CENTRE_COLOR,
        marker="+",
        linewidths=2,
        zorder=4.5,
        label="Mass centre",
    )


def model_pairing_from(
    fit_positions,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns the image-plane pairing of observed to model positions which the fit's chi-squared uses, as a tuple
    ``(observed, model, model_index_per_observed)``, where ``model`` holds only the finite model positions and
    ``model_index_per_observed[i]`` is the model position observed position ``i`` is paired with (``-1`` if the
    model has no images).

    - ``FitPositionsImagePair``: the Hungarian (one-to-one) assignment; observed positions the assignment cannot
      pair (fewer model than observed images) are paired with their nearest model position, as in its residuals.
    - ``FitPositionsImagePairRepeat``: each observed position's nearest model position (repeats allowed).
    - ``FitPositionsImagePairAll``: each observed position's nearest model position, the dominant term of its
      all-pairs mixture likelihood (the mixture has no hard pairing).

    Source-plane fits (e.g. ``FitPositionsSource``) have no image-plane pairing, so ``None`` is returned.
    """
    from autolens.point.fit.positions.image.abstract import (
        AbstractFitPositionsImagePair,
    )
    from autolens.point.fit.positions.image.pair import FitPositionsImagePair

    if not isinstance(fit_positions, AbstractFitPositionsImagePair):
        return None

    observed = _np_grid(fit_positions.data)
    model = _np_grid(fit_positions.model_data)
    model = model[np.isfinite(model).all(axis=1)]

    if len(model) == 0:
        return observed, model, np.full(len(observed), -1, dtype=int)

    distances = np.linalg.norm(observed[:, None, :] - model[None, :, :], axis=2)
    indexes = np.argmin(distances, axis=1)

    if isinstance(fit_positions, FitPositionsImagePair):
        from scipy.optimize import linear_sum_assignment

        data_indexes, model_indexes = linear_sum_assignment(distances)
        indexes[data_indexes] = model_indexes

    return observed, model, indexes


def unmatched_model_mask_from(model: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask over the (finite) model positions, true for every model image no observed position is
    paired with: an image the model predicts but which is not observed.
    """
    mask = np.ones(len(model), dtype=bool)
    mask[indexes[indexes >= 0]] = False
    return mask


def plot_fit_model_positions(ax, fit_positions):
    """
    Draws the model positions of a positions fit on top of the observed positions.

    For an image-plane fit the matched model positions are drawn as red crosses, a thin ``Residual`` line joins
    each observed position to the model position the fit pairs it with, and model images no observed position is
    paired with are drawn as hollow grey crosses (``Unmatched model image``).

    For a source-plane fit (e.g. ``FitPositionsSource``) the model data are the observed positions ray-traced to
    the source plane: they are drawn as red crosses labelled ``Model (source plane)`` and no residual lines are
    drawn, as there is no image-plane pairing.
    """
    pairing = model_pairing_from(fit_positions)

    if pairing is None:
        model = _np_grid(fit_positions.model_data)
        model = model[np.isfinite(model).all(axis=1)]
        if len(model):
            ax.scatter(
                model[:, 1],
                model[:, 0],
                s=MODEL_MARKER_SIZE,
                c=MODEL_COLOR,
                marker="x",
                linewidths=MODEL_MARKER_LINEWIDTH,
                zorder=7,
                label="Model (source plane)",
            )
        return

    observed, model, indexes = pairing

    labelled = False
    for obs, index in zip(observed, indexes):
        if index < 0:
            continue
        ax.plot(
            [obs[1], model[index, 1]],
            [obs[0], model[index, 0]],
            color=MODEL_COLOR,
            linewidth=1.0,
            alpha=0.7,
            zorder=6,
            label=None if labelled else "Residual",
        )
        labelled = True

    unmatched = unmatched_model_mask_from(model, indexes)

    if np.any(~unmatched):
        ax.scatter(
            model[~unmatched, 1],
            model[~unmatched, 0],
            s=MODEL_MARKER_SIZE,
            c=MODEL_COLOR,
            marker="x",
            linewidths=MODEL_MARKER_LINEWIDTH,
            zorder=7,
            label="Model",
        )
    if np.any(unmatched):
        ax.scatter(
            model[unmatched, 1],
            model[unmatched, 0],
            s=110,
            facecolors="none",
            edgecolors=UNMATCHED_COLOR,
            marker="X",
            linewidths=1.5,
            zorder=7,
            label="Unmatched model image",
        )


def finish_positions_panel(ax, title: str, extent: List[float]):
    """
    Applies the title, extent and arc-second ticks of a positions panel, with no axis labels, an equal aspect
    ratio (so noise circles are circles) and a legend.
    """
    from autoarray.plot.utils import apply_labels, apply_extent

    apply_labels(ax, title=title, xlabel="", ylabel="")
    apply_extent(ax, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=9, framealpha=0.85, markerscale=0.7)

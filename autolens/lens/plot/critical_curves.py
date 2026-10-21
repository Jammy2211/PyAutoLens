"""Shared helper for computing critical-curve / caustic overlay lines.

This lives under ``autolens.lens.plot`` (whose package ``__init__`` files are
empty) rather than ``autolens.plot`` so that the fit-plot modules can import it
at module load without triggering the ``autolens.plot`` aggregator ``__init__``,
which imports those same fit-plot modules and would create a circular import.
"""

import logging

import autogalaxy as ag
from autogalaxy.util.plot_utils import _critical_curves_from, _caustics_from
from autoarray.plot.utils import numpy_lines as _to_lines

logger = logging.getLogger(__name__)


def _compute_critical_curve_lines(tracer, grid):
    """Compute critical-curve and caustic lines for a tracer on a given grid.

    Returns a 4-tuple ``(image_plane_lines, image_plane_line_colors,
    source_plane_lines, source_plane_line_colors)`` suitable for passing
    directly to :func:`~autoarray.plot.array.plot_array`.  On failure
    (e.g. the mass model has no critical curves) returns
    ``(None, None, None, None)``.

    Parameters
    ----------
    tracer
        The tracer whose mass distribution is used to trace critical curves
        and caustics.
    grid
        Image-plane grid on which the curves are evaluated.
    """
    try:
        tan_cc, rad_cc = _critical_curves_from(tracer, grid)
        tan_ca, rad_ca = _caustics_from(tracer, grid)
        _tan_cc_lines = _to_lines(list(tan_cc) if tan_cc is not None else []) or []
        _rad_cc_lines = _to_lines(list(rad_cc) if rad_cc is not None else []) or []
        _tan_ca_lines = _to_lines(list(tan_ca) if tan_ca is not None else []) or []
        _rad_ca_lines = _to_lines(list(rad_ca) if rad_ca is not None else []) or []
        image_plane_lines = (_tan_cc_lines + _rad_cc_lines) or None
        image_plane_line_colors = (
            ["white"] * len(_tan_cc_lines) + ["yellow"] * len(_rad_cc_lines)
        )
        source_plane_lines = (_tan_ca_lines + _rad_ca_lines) or None
        source_plane_line_colors = (
            ["white"] * len(_tan_ca_lines) + ["yellow"] * len(_rad_ca_lines)
        )
        return image_plane_lines, image_plane_line_colors, source_plane_lines, source_plane_line_colors
    except (ModuleNotFoundError, ValueError):
        # ModuleNotFoundError: jax_zero_contour missing — already warned upstream in
        # plot_utils._critical_curves_method().
        # ValueError: no zero crossings in the eigenvalue grid (e.g. slope >= 2
        # isothermal where lambda_r > 0 everywhere). Curves don't exist for this
        # model, so rendering without overlays is correct.
        return None, None, None, None
    except Exception:
        # Anything else — log loudly with traceback so the next regression of the
        # "ZeroSolver raised inside model-fit, viz fell back to all-zero" failure
        # mode (PyAutoGalaxy abd7b717, PyAutoFit #1280) does not stay silent.
        logger.warning(
            "Critical-curve computation failed unexpectedly; rendering without "
            "overlays. Investigate — this used to be a silent fallback.",
            exc_info=True,
        )
        return None, None, None, None


def critical_curve_grid_from(
    dataset, shape_native=(100, 100), scale: float = 2.0, buffer: float = 0.1
) -> "ag.Grid2D":
    """
    Returns the grid on which the critical curves and caustics of a dataset's lens model are computed.

    The grid is a square centred on the bounding box of the dataset's positions (``dataset.extent_from``), whose
    half-width is ``scale`` times the half-span of that box (plus ``buffer``). For a point dataset critical curves
    often extend beyond the multiple images, so a grid covering only their bounding box clips them into
    disconnected arcs and ``scale=2.0`` is used. For a weak lensing dataset the source galaxies already span the
    field, so ``scale=1.0`` covers it.

    ``dataset.extent_from`` returns ``[y_min, y_max, x_min, x_max]`` whereas ``Grid2D.from_extent`` takes the
    extent x first, which this function handles.

    Parameters
    ----------
    dataset
        A dataset with an ``extent_from`` method (e.g. a ``PointDataset`` or ``WeakDataset``) whose positions set
        the centre and size of the grid.
    shape_native
        The 2D shape of the grid.
    scale
        The factor by which the grid's half-width exceeds the half-span of the positions.
    buffer
        A buffer (arc-seconds) added to the half-span of the positions.
    """
    y_min, y_max, x_min, x_max = dataset.extent_from(buffer=0.0)

    y_centre, x_centre = (y_min + y_max) / 2.0, (x_min + x_max) / 2.0
    half_width = scale * (max(y_max - y_min, x_max - x_min) / 2.0 + buffer)

    # `Grid2D.from_extent` takes the extent x first: (x0, x1, y0, y1).
    return ag.Grid2D.from_extent(
        extent=(
            x_centre - half_width,
            x_centre + half_width,
            y_centre - half_width,
            y_centre + half_width,
        ),
        shape_native=shape_native,
    )

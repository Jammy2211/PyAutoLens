import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_array(
    array,
    ax=None,
    title="",
    lines=None,
    positions=None,
    colormap="jet",
    use_log10=False,
    vmin=None,
    vmax=None,
    output_path=None,
    output_filename="array",
    output_format="png",
):
    """Plot an Array2D (or numpy array) using autoarray's low-level plot_array.

    When *ax* is provided the figure is rendered into that axes object (subplot
    mode) and no file is written.  When *ax* is ``None`` the figure is saved to
    *output_path/output_filename.output_format* (standalone mode).
    """
    from autoarray.plot.plots.array import plot_array as _aa_plot_array
    from autoarray.structures.plot.structure_plotters import (
        _auto_mask_edge,
        _numpy_lines,
        _numpy_positions,
        _zoom_array,
    )

    array = _zoom_array(array)

    try:
        arr = array.native.array
        extent = array.geometry.extent
    except AttributeError:
        arr = np.asarray(array)
        extent = None

    mask = _auto_mask_edge(array) if hasattr(array, "mask") else None
    _lines = lines if isinstance(lines, list) else _numpy_lines(lines)
    _positions = (
        positions if isinstance(positions, list) else _numpy_positions(positions)
    )

    _aa_plot_array(
        array=arr,
        ax=ax,
        extent=extent,
        mask=mask,
        positions=_positions,
        lines=_lines,
        title=title,
        colormap=colormap,
        use_log10=use_log10,
        vmin=vmin,
        vmax=vmax,
        output_path=output_path if ax is None else None,
        output_filename=output_filename,
        output_format=output_format,
        structure=array,
    )


def plot_grid(
    grid,
    ax=None,
    title="",
    output_path=None,
    output_filename="grid",
    output_format="png",
):
    """Plot a Grid2D using autoarray's low-level plot_grid."""
    from autoarray.plot.plots.grid import plot_grid as _aa_plot_grid

    _aa_plot_grid(
        grid=np.array(grid.array),
        ax=ax,
        title=title,
        output_path=output_path if ax is None else None,
        output_filename=output_filename,
        output_format=output_format,
    )


def _to_lines(*items):
    """Convert multiple line sources into a flat list of (N, 2) numpy arrays."""
    result = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, list):
            for sub in item:
                try:
                    arr = np.array(sub.array if hasattr(sub, "array") else sub)
                    if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                        result.append(arr)
                except Exception:
                    pass
        else:
            try:
                arr = np.array(item.array if hasattr(item, "array") else item)
                if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                    result.append(arr)
            except Exception:
                pass
    return result or None


def _to_positions(*items):
    """Convert multiple position sources into a flat list of (N, 2) numpy arrays."""
    return _to_lines(*items)


def _save_subplot(fig, output_path, filename, output_format="png"):
    """Save a subplot figure to disk (or show it when output_path is None)."""
    # Normalise: format may be a list (e.g. ['png']) or a plain string.
    if isinstance(output_format, (list, tuple)):
        fmts = output_format
    else:
        fmts = [output_format]

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        for fmt in fmts:
            fig.savefig(
                os.path.join(output_path, f"{filename}.{fmt}"),
                bbox_inches="tight",
                pad_inches=0.1,
            )
    else:
        plt.show()
    plt.close(fig)


def _critical_curves_from(tracer, grid):
    """Return (tangential_critical_curves, radial_critical_curves) as lists of arrays."""
    from autolens.lens import tracer_util

    try:
        tan_cc, rad_cc = tracer_util.critical_curves_from(tracer=tracer, grid=grid)
        return list(tan_cc), list(rad_cc)
    except Exception:
        return [], []


def _caustics_from(tracer, grid):
    """Return (tangential_caustics, radial_caustics) as lists of arrays."""
    from autolens.lens import tracer_util

    try:
        tan_ca, rad_ca = tracer_util.caustics_from(tracer=tracer, grid=grid)
        return list(tan_ca), list(rad_ca)
    except Exception:
        return [], []

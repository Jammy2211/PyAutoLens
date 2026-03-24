import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def _zoom_array(array):
    """Apply zoom_around_mask from config if requested."""
    try:
        from autoconf import conf
        zoom_around_mask = conf.instance["visualize"]["general"]["general"]["zoom_around_mask"]
    except Exception:
        zoom_around_mask = False

    if zoom_around_mask and hasattr(array, "mask") and not array.mask.is_all_false:
        try:
            from autoarray.mask.derive.zoom_2d import Zoom2D
            return Zoom2D(mask=array.mask).array_2d_from(array=array, buffer=1)
        except Exception:
            pass
    return array


def _auto_mask_edge(array) -> Optional[np.ndarray]:
    """Return edge-pixel (y, x) coords from array.mask, or None."""
    try:
        if not array.mask.is_all_false:
            return np.array(array.mask.derive_grid.edge.array)
    except AttributeError:
        pass
    return None


def _numpy_lines(lines) -> Optional[List[np.ndarray]]:
    """Convert lines (Grid2DIrregular or list) to list of (N,2) numpy arrays."""
    if lines is None:
        return None
    result = []
    try:
        for line in lines:
            try:
                arr = np.array(line.array if hasattr(line, "array") else line)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    result.append(arr)
            except Exception:
                pass
    except TypeError:
        pass
    return result or None


def _numpy_positions(positions) -> Optional[List[np.ndarray]]:
    """Convert positions to list of (N,2) numpy arrays."""
    if positions is None:
        return None
    try:
        arr = np.array(positions.array if hasattr(positions, "array") else positions)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return [arr]
    except Exception:
        pass
    if isinstance(positions, list):
        result = []
        for p in positions:
            try:
                result.append(np.array(p.array if hasattr(p, "array") else p))
            except Exception:
                pass
        return result or None
    return None


def _prepare_array(array):
    """Zoom and extract (arr_2d, extent, mask) from an Array2D-like object.

    Returns a plain (N, M) numpy array suitable for passing to
    ``autoarray.plot.plots.array.plot_array``, along with the spatial *extent*
    and edge-pixel *mask* overlays.
    """
    array = _zoom_array(array)
    try:
        arr = array.native.array
        extent = array.geometry.extent
    except AttributeError:
        arr = np.asarray(array)
        extent = None
    mask = _auto_mask_edge(array) if hasattr(array, "mask") else None
    return arr, extent, mask


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
    """Plot an Array2D (or numpy array) via autoarray's plot_array."""
    from autoarray.plot.plots.array import plot_array as _aa_plot_array

    arr, extent, mask = _prepare_array(array)
    _aa_plot_array(
        array=arr,
        ax=ax,
        extent=extent,
        mask=mask,
        positions=positions if isinstance(positions, list) else _numpy_positions(positions),
        lines=lines if isinstance(lines, list) else _numpy_lines(lines),
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
    """Plot a Grid2D via autoarray's plot_grid."""
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

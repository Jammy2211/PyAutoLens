from typing import List, Optional
import numpy as np

from autoarray.plot.wrap.base.abstract import set_backend

set_backend()

from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.grid import plot_grid
from autoarray.structures.plot.structure_plotters import (
    _mask_edge_from,
    _grid_from_visuals,
    _lines_from_visuals,
    _positions_from_visuals,
    _output_for_mat_plot,
    _zoom_array,
)
from autogalaxy.plot.abstract_plotters import Plotter as _AGPlotter
from autogalaxy.plot.plots.overlays import (
    _galaxy_lines_from_visuals,
    _galaxy_positions_from_visuals,
)

from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.plot.visuals.two_d import Visuals2D


def _to_lines(*items) -> Optional[List[np.ndarray]]:
    """Flatten one or more line sources into a single list of (N,2) numpy arrays.

    Each item can be:
      - None (skipped)
      - a list of array-like objects each of shape (N,2)
      - a single array-like of shape (N,2)
    """
    result = []
    for item in items:
        if item is None:
            continue
        try:
            for sub in item:
                try:
                    arr = np.array(sub.array if hasattr(sub, "array") else sub)
                    if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                        result.append(arr)
                except Exception:
                    pass
        except TypeError:
            try:
                arr = np.array(item.array if hasattr(item, "array") else item)
                if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                    result.append(arr)
            except Exception:
                pass
    return result or None


def _to_positions(*items) -> Optional[List[np.ndarray]]:
    """Flatten one or more position sources into a single list of (N,2) numpy arrays."""
    return _to_lines(*items)


class Plotter(_AGPlotter):

    def __init__(
        self,
        mat_plot_1d: MatPlot1D = None,
        visuals_1d: Visuals1D = None,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):

        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
        )

        self.visuals_1d = visuals_1d or Visuals1D()
        self.mat_plot_1d = mat_plot_1d or MatPlot1D()

        self.visuals_2d = visuals_2d or Visuals2D()
        self.mat_plot_2d = mat_plot_2d or MatPlot2D()

    def _plot_array(
        self,
        array,
        auto_labels,
        lines: Optional[List[np.ndarray]] = None,
        positions: Optional[List[np.ndarray]] = None,
        grid=None,
        visuals_2d=None,
    ):
        """Plot an Array2D using the standalone plot_array() function.

        Overlays are supplied directly as lists of numpy arrays via *lines* and
        *positions*, or extracted from an optional *visuals_2d* object (which may
        carry mask, grid, base lines, base positions, profile centres, critical
        curves, caustics, etc.).  The two sources are merged so that either – or
        both – may be provided at the same time.

        Parameters
        ----------
        array
            The 2-D array to plot.
        auto_labels
            Title / filename configuration from the calling plotter.
        lines
            Extra line overlays (critical curves, caustics …) as a list of
            (N, 2) numpy arrays.
        positions
            Extra scatter-point overlays as a list of (N, 2) numpy arrays.
        grid
            Optional extra scatter grid (passed through to plot_array).
        visuals_2d
            Legacy Visuals2D object; mask, base lines/positions and profile
            centres are extracted from it and merged with the explicit kwargs.
        """
        if array is None:
            return

        v2d = visuals_2d if visuals_2d is not None else self.visuals_2d

        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None

        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_2d,
            is_sub,
            auto_labels.filename if auto_labels else "array",
        )

        array = _zoom_array(array)

        try:
            import numpy as _np
            arr = array.native.array
            extent = array.geometry.extent
        except AttributeError:
            arr = np.asarray(array)
            extent = None

        # Merge overlays from visuals_2d with explicit kwargs
        vis_lines = _galaxy_lines_from_visuals(v2d) or []
        vis_positions = _galaxy_positions_from_visuals(v2d) or []

        all_lines = vis_lines + (lines or []) or None
        all_positions = vis_positions + (positions or []) or None

        plot_array(
            array=arr,
            ax=ax,
            extent=extent,
            mask=_mask_edge_from(array if hasattr(array, "mask") else None, v2d),
            grid=_grid_from_visuals(v2d) if grid is None else grid,
            positions=all_positions,
            lines=all_lines,
            title=auto_labels.title if auto_labels else "",
            colormap=self.mat_plot_2d.cmap.cmap,
            use_log10=self.mat_plot_2d.use_log10,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
            structure=array,
        )

    def _plot_grid(
        self,
        grid,
        auto_labels,
        lines: Optional[List[np.ndarray]] = None,
        visuals_2d=None,
    ):
        """Plot a Grid2D using the standalone plot_grid() function."""
        if grid is None:
            return

        v2d = visuals_2d if visuals_2d is not None else self.visuals_2d

        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None

        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_2d,
            is_sub,
            auto_labels.filename if auto_labels else "grid",
        )

        vis_lines = _galaxy_lines_from_visuals(v2d) or []
        all_lines = vis_lines + (lines or []) or None

        try:
            grid_arr = np.array(grid.array if hasattr(grid, "array") else grid)
        except Exception:
            grid_arr = np.asarray(grid)

        plot_grid(
            grid=grid_arr,
            ax=ax,
            lines=all_lines,
            title=auto_labels.title if auto_labels else "",
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
        )

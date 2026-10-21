import numpy as np
from typing import Optional

from autoarray.plot.utils import subplots, save_figure, conf_subplot_figsize, tight_layout


def subplot_fit(
    fit,
    output_path: Optional[str] = None,
    output_format: str = None,
    image_plane_lines=None,
    image_plane_line_colors=None,
    source_plane_lines=None,
    source_plane_line_colors=None,
    title_prefix: str = None,
):
    """
    Produce a subplot summarising a `FitPointDataset`.

    The subplot contains one or two panels depending on whether flux
    measurements are present in the dataset:

    * **Positions panel** (always shown): observed point-source positions
      plotted as a grid, with the model-predicted positions overlaid as
      red scatter points.
    * **Fluxes panel** (shown only when ``fit.dataset.fluxes`` is not
      ``None``): a bar/line plot of the observed flux values.

    Parameters
    ----------
    fit : FitPointDataset
        The point-source dataset fit to visualise.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    """
    from autogalaxy.util.plot_utils import plot_grid
    from autoarray.plot.yx import plot_yx

    has_fluxes = fit.dataset.fluxes is not None
    n = 2 if has_fluxes else 1

    fig, axes = subplots(1, n, figsize=conf_subplot_figsize(1, n))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    # Positions panel
    obs_grid = np.array(
        fit.dataset.positions.array
        if hasattr(fit.dataset.positions, "array")
        else fit.dataset.positions
    )
    model_grid = np.array(
        fit.positions.model_data.array
        if hasattr(fit.positions.model_data, "array")
        else fit.positions.model_data
    )

    _prefix = f"{title_prefix.rstrip()} " if title_prefix else ""
    plot_grid(
        grid=obs_grid,
        ax=axes_flat[0],
        title=f"{_prefix}{fit.dataset.name} Fit Positions",
        output_path=None,
        output_filename=None,
        output_format=output_format,
    )
    axes_flat[0].scatter(model_grid[:, 1], model_grid[:, 0], c="r", s=20, zorder=5)

    # Fluxes panel
    if has_fluxes and n > 1:
        y = np.array(fit.dataset.fluxes)
        x = np.arange(len(y))
        plot_yx(
            y=y,
            x=x,
            ax=axes_flat[1],
            title=f"{_prefix}{fit.dataset.name} Fit Fluxes",
            output_path=None,
            output_filename="fit_point_fluxes",
            output_format=output_format,
        )

    tight_layout()
    save_figure(fig, path=output_path, filename="fit", format=output_format)


def plot_image_with_points(
    image,
    dataset,
    fit=None,
    output_path: Optional[str] = None,
    output_format: str = None,
    image_plane_lines=None,
    image_plane_line_colors=None,
    title_prefix: str = None,
):
    """
    Plot an observed image of a lensed point source (e.g. a quasar) with its point-source positions overlaid.

    The observed positions of the `PointDataset` are drawn as high-contrast markers (white fill, black edge)
    so they stand out against any colormap. When a `FitPointDataset` is passed, the model-predicted positions
    are drawn as red crosses, and critical curves can be overlaid via ``image_plane_lines``.

    The image is used only for visualization: it does not enter the point-source likelihood.

    Parameters
    ----------
    image : Array2D
        The observed image the positions are plotted over.
    dataset : PointDataset
        The point-source dataset whose observed positions are overlaid.
    fit : FitPointDataset, optional
        If given, its model positions are overlaid as red crosses.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    image_plane_lines : list, optional
        Critical-curve lines overlaid on the image.
    image_plane_line_colors : list, optional
        Colours for each image-plane line.
    """
    from autogalaxy.util.plot_utils import plot_array
    from autoarray.plot.utils import conf_figsize, _overlay_yx_for_origin

    fig, ax = subplots(1, 1, figsize=conf_figsize("figures"))

    _prefix = f"{title_prefix.rstrip()} " if title_prefix else ""
    plot_array(
        array=image,
        ax=ax,
        lines=image_plane_lines,
        line_colors=image_plane_line_colors,
        title=f"{_prefix}{dataset.name} Image With Points",
    )

    # Positions are (y,x) arc-seconds: reflect them exactly as `plot_array` does its own overlays, using the
    # extent and origin of the raster it drew, so they stay registered to the image (PyAutoArray#565).
    extent, origin = (ax.images[0].get_extent(), ax.images[0].origin) if ax.images else (None, None)

    def _yx(grid):
        grid = np.array(grid.array if hasattr(grid, "array") else grid).reshape(-1, 2)
        return _overlay_yx_for_origin(grid, extent, origin)

    obs_grid = _yx(dataset.positions)
    ax.scatter(
        obs_grid[:, 1], obs_grid[:, 0],
        s=60, c="w", edgecolors="k", linewidths=1.5, zorder=6, label="Observed",
    )

    if fit is not None:
        model_grid = _yx(fit.positions.model_data)
        ax.scatter(
            model_grid[:, 1], model_grid[:, 0],
            s=60, c="r", marker="x", linewidths=2, zorder=7, label="Model",
        )

    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    save_figure(fig, path=output_path, filename="image_with_points", format=output_format)

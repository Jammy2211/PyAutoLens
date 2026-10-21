import numpy as np
from typing import Optional

from autoarray.plot.utils import subplots, save_figure, conf_subplot_figsize, tight_layout


def subplot_dataset(
    dataset,
    output_path: Optional[str] = None,
    output_format: str = None,
    title_prefix: str = None,
    solver=None,
):
    """
    Produce a subplot visualising a `PointDataset`.

    The subplot contains one or two panels depending on whether flux
    measurements are present:

    * **Positions panel** (always shown), titled ``Dataset Positions (<name>)``:
      the observed point-source positions as large filled markers, each with a
      circle at its noise-map radius (never inflated, so it may be sub-pixel)
      and the sigma stated in the legend, e.g. ``Observed (σ = 0.005″)``. The
      panel extent is the ``PointSolver`` grid extent when ``solver`` is given
      (where the model can find images), otherwise the critical-curve region of
      ``critical_curve_grid_from``, so no position is clipped. There are no axis
      labels, only arc-second tick labels.
    * **Fluxes panel** (shown only when ``dataset.fluxes`` is not
      ``None``): a line/bar plot of the observed flux values indexed by
      image position.

    Parameters
    ----------
    dataset : PointDataset
        The point-source dataset to visualise.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    title_prefix : str, optional
        Text prepended to every panel title.
    solver : PointSolver, optional
        The solver of the model-fit, whose grid sets the positions panel extent.
    """
    from autoarray.plot.yx import plot_yx
    from autolens.point.plot.positions_panel import (
        finish_positions_panel,
        plot_observed_positions,
        positions_extent_from,
    )

    has_fluxes = dataset.fluxes is not None
    n = 2 if has_fluxes else 1

    fig, axes = subplots(1, n, figsize=conf_subplot_figsize(1, n))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    _prefix = f"{title_prefix.rstrip()} " if title_prefix else ""

    plot_observed_positions(
        ax=axes_flat[0],
        positions=dataset.positions,
        noise_map=dataset.positions_noise_map,
    )
    finish_positions_panel(
        ax=axes_flat[0],
        title=f"{_prefix}Dataset Positions ({dataset.name})",
        extent=positions_extent_from(positions=dataset.positions, solver=solver),
    )

    if has_fluxes and n > 1:
        y = np.array(dataset.fluxes)
        x = np.arange(len(y))
        plot_yx(
            y=y,
            x=x,
            ax=axes_flat[1],
            title=f"{_prefix}Dataset Fluxes ({dataset.name})",
            output_path=None,
            output_filename="point_dataset_fluxes",
            output_format=output_format,
        )

    tight_layout()
    save_figure(fig, path=output_path, filename="dataset_point", format=output_format)

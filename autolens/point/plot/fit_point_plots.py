import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from autoarray.plot.plots.utils import save_figure


def subplot_fit(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
):
    """Subplot of a FitPointDataset: positions panel and (optionally) fluxes panel."""
    from autoarray.plot.plots.grid import plot_grid
    from autoarray.plot.plots.yx import plot_yx

    has_fluxes = fit.dataset.fluxes is not None
    n = 2 if has_fluxes else 1

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
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

    plot_grid(
        grid=obs_grid,
        ax=axes_flat[0],
        title=f"{fit.dataset.name} Fit Positions",
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
            title=f"{fit.dataset.name} Fit Fluxes",
            output_path=None,
            output_filename="fit_point_fluxes",
            output_format=output_format,
        )

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="subplot_fit", format=output_format)

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from autoarray.plot.plots.utils import save_figure


def subplot_dataset(
    dataset,
    output_path: Optional[str] = None,
    output_format: str = "png",
):
    """Subplot of a PointDataset: positions panel and (optionally) fluxes panel."""
    from autoarray.plot.plots.grid import plot_grid
    from autoarray.plot.plots.yx import plot_yx

    has_fluxes = dataset.fluxes is not None
    n = 2 if has_fluxes else 1

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    grid = np.array(
        dataset.positions.array
        if hasattr(dataset.positions, "array")
        else dataset.positions
    )

    plot_grid(
        grid=grid,
        ax=axes_flat[0],
        title=f"{dataset.name} Positions",
        output_path=None,
        output_filename=None,
        output_format=output_format,
    )

    if has_fluxes and n > 1:
        y = np.array(dataset.fluxes)
        x = np.arange(len(y))
        plot_yx(
            y=y,
            x=x,
            ax=axes_flat[1],
            title=f"{dataset.name} Fluxes",
            output_path=None,
            output_filename="point_dataset_fluxes",
            output_format=output_format,
        )

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="subplot_dataset_point", format=output_format)

import matplotlib.pyplot as plt
import numpy as np

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap
from autoarray.plot.plots.grid import plot_grid
from autoarray.plot.plots.yx import plot_yx
from autoarray.structures.plot.structure_plotters import _output_for_plotter
from autogalaxy.plot.abstract_plotters import _save_subplot

from autolens.point.dataset import PointDataset
from autolens.plot.abstract_plotters import Plotter


class PointDatasetPlotter(Plotter):
    def __init__(
        self,
        dataset: PointDataset,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

        self.dataset = dataset

    def figures_2d(self, positions: bool = False, fluxes: bool = False, ax=None):
        standalone = ax is None

        if positions:
            if standalone:
                output_path, filename, fmt = _output_for_plotter(
                    self.output, "point_dataset_positions"
                )
            else:
                output_path, filename, fmt = None, "point_dataset_positions", "png"

            grid = np.array(
                self.dataset.positions.array
                if hasattr(self.dataset.positions, "array")
                else self.dataset.positions
            )

            plot_grid(
                grid=grid,
                ax=ax,
                title=f"{self.dataset.name} Positions",
                output_path=output_path,
                output_filename=filename,
                output_format=fmt,
            )

        if fluxes:
            if self.dataset.fluxes is not None:
                if standalone:
                    output_path, filename, fmt = _output_for_plotter(
                        self.output, "point_dataset_fluxes"
                    )
                else:
                    output_path, filename, fmt = None, "point_dataset_fluxes", "png"

                y = np.array(self.dataset.fluxes)
                x = np.arange(len(y))

                plot_yx(
                    y=y,
                    x=x,
                    ax=ax,
                    title=f"{self.dataset.name} Fluxes",
                    output_path=output_path,
                    output_filename=filename,
                    output_format=fmt,
                )

    def subplot_dataset(self):
        has_fluxes = self.dataset.fluxes is not None
        n = 2 if has_fluxes else 1

        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
        axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

        self.figures_2d(positions=True, ax=axes_flat[0])
        if has_fluxes and n > 1:
            self.figures_2d(fluxes=True, ax=axes_flat[1])

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_dataset_point")

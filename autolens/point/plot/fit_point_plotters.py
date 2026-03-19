import matplotlib.pyplot as plt
import numpy as np

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap
from autoarray.plot.plots.grid import plot_grid
from autoarray.plot.plots.yx import plot_yx
from autoarray.plot.plots.utils import save_figure
from autoarray.structures.plot.structure_plotters import _output_for_plotter
from autogalaxy.plot.abstract_plotters import _save_subplot

from autolens.plot.abstract_plotters import Plotter
from autolens.point.fit.dataset import FitPointDataset


class FitPointDatasetPlotter(Plotter):
    def __init__(
        self,
        fit: FitPointDataset,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

        self.fit = fit

    def figures_2d(self, positions: bool = False, fluxes: bool = False, ax=None):
        standalone = ax is None

        if positions:
            if standalone:
                output_path, filename, fmt = _output_for_plotter(
                    self.output, "fit_point_positions"
                )
            else:
                output_path, filename, fmt = None, "fit_point_positions", "png"

            obs_grid = np.array(
                self.fit.dataset.positions.array
                if hasattr(self.fit.dataset.positions, "array")
                else self.fit.dataset.positions
            )
            model_grid = np.array(
                self.fit.positions.model_data.array
                if hasattr(self.fit.positions.model_data, "array")
                else self.fit.positions.model_data
            )

            pos_ax = ax
            if standalone:
                fig, pos_ax = plt.subplots(1, 1)

            plot_grid(
                grid=obs_grid,
                ax=pos_ax,
                title=f"{self.fit.dataset.name} Fit Positions",
                output_path=None,
                output_filename=None,
                output_format=fmt,
            )

            pos_ax.scatter(model_grid[:, 1], model_grid[:, 0], c="r", s=20, zorder=5)

            if standalone:
                save_figure(
                    pos_ax.get_figure(),
                    path=output_path or "",
                    filename=filename,
                    format=fmt,
                )

        if fluxes:
            if self.fit.dataset.fluxes is not None:
                if standalone:
                    output_path, filename, fmt = _output_for_plotter(
                        self.output, "fit_point_fluxes"
                    )
                else:
                    output_path, filename, fmt = None, "fit_point_fluxes", "png"

                y = np.array(self.fit.dataset.fluxes)
                x = np.arange(len(y))

                plot_yx(
                    y=y,
                    x=x,
                    ax=ax,
                    title=f"{self.fit.dataset.name} Fit Fluxes",
                    output_path=output_path,
                    output_filename=filename,
                    output_format=fmt,
                )

    def subplot_fit(self):
        has_fluxes = self.fit.dataset.fluxes is not None
        n = 2 if has_fluxes else 1

        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
        axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

        self.figures_2d(positions=True, ax=axes_flat[0])
        if has_fluxes and n > 1:
            self.figures_2d(fluxes=True, ax=axes_flat[1])

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_fit")

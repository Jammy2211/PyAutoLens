import numpy as np

import autogalaxy.plot as aplt

from autoarray.plot.plots.grid import plot_grid
from autoarray.plot.plots.yx import plot_yx
from autoarray.structures.plot.structure_plotters import _output_for_mat_plot

from autolens.plot.abstract_plotters import Plotter
from autolens.point.fit.dataset import FitPointDataset


class FitPointDatasetPlotter(Plotter):
    def __init__(
        self,
        fit: FitPointDataset,
        mat_plot_1d: aplt.MatPlot1D = None,
        mat_plot_2d: aplt.MatPlot2D = None,
    ):
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
        )

        self.fit = fit

    def figures_2d(self, positions: bool = False, fluxes: bool = False):
        if positions:
            if self.mat_plot_2d.axis.kwargs.get("extent") is None:
                buffer = 0.1

                y_max = (
                    max(
                        max(self.fit.dataset.positions[:, 0]),
                        max(self.fit.positions.model_data[:, 0]),
                    )
                    + buffer
                )
                y_min = (
                    min(
                        min(self.fit.dataset.positions[:, 0]),
                        min(self.fit.positions.model_data[:, 0]),
                    )
                    - buffer
                )
                x_max = (
                    max(
                        max(self.fit.dataset.positions[:, 1]),
                        max(self.fit.positions.model_data[:, 1]),
                    )
                    + buffer
                )
                x_min = (
                    min(
                        min(self.fit.dataset.positions[:, 1]),
                        min(self.fit.positions.model_data[:, 1]),
                    )
                    - buffer
                )

                self.mat_plot_2d.axis.kwargs["extent"] = [y_min, y_max, x_min, x_max]

            is_sub = self.mat_plot_2d.is_for_subplot
            ax = self.mat_plot_2d.setup_subplot() if is_sub else None
            output_path, filename, fmt = _output_for_mat_plot(
                self.mat_plot_2d, is_sub, "fit_point_positions"
            )

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

            import matplotlib.pyplot as plt
            from autoarray.plot.plots.utils import save_figure

            owns_figure = ax is None
            if owns_figure:
                fig, ax = plt.subplots(1, 1)

            plot_grid(
                grid=obs_grid,
                ax=ax,
                title=f"{self.fit.dataset.name} Fit Positions",
                output_path=None,
                output_filename=None,
                output_format=fmt,
            )

            ax.scatter(model_grid[:, 1], model_grid[:, 0], c="r", s=20, zorder=5)

            if owns_figure:
                save_figure(
                    ax.get_figure(),
                    path=output_path or "",
                    filename=filename,
                    format=fmt,
                )

        # nasty hack to ensure subplot index between 2d and 1d plots are synced.
        if (
            self.mat_plot_1d.subplot_index is not None
            and self.mat_plot_2d.subplot_index is not None
        ):
            self.mat_plot_1d.subplot_index = max(
                self.mat_plot_1d.subplot_index, self.mat_plot_2d.subplot_index
            )

        if fluxes:
            if self.fit.dataset.fluxes is not None:
                is_sub = self.mat_plot_1d.is_for_subplot
                ax = self.mat_plot_1d.setup_subplot() if is_sub else None
                output_path, filename, fmt = _output_for_mat_plot(
                    self.mat_plot_1d, is_sub, "fit_point_fluxes"
                )

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

    def subplot(
        self,
        positions: bool = False,
        fluxes: bool = False,
        auto_filename: str = "subplot_fit",
    ):
        self._subplot_custom_plot(
            positions=positions,
            fluxes=fluxes,
            auto_labels=aplt.AutoLabels(filename=auto_filename),
        )

    def subplot_fit(self):
        self.subplot(positions=True, fluxes=True)

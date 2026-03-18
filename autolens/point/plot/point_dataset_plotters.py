import numpy as np

import autogalaxy.plot as aplt

from autoarray.plot.plots.grid import plot_grid
from autoarray.plot.plots.yx import plot_yx
from autoarray.structures.plot.structure_plotters import _output_for_mat_plot

from autolens.point.dataset import PointDataset
from autolens.plot.abstract_plotters import Plotter


class PointDatasetPlotter(Plotter):
    def __init__(
        self,
        dataset: PointDataset,
        mat_plot_1d: aplt.MatPlot1D = None,
        mat_plot_2d: aplt.MatPlot2D = None,
    ):
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
        )

        self.dataset = dataset

    def figures_2d(self, positions: bool = False, fluxes: bool = False):
        if positions:
            is_sub = self.mat_plot_2d.is_for_subplot
            ax = self.mat_plot_2d.setup_subplot() if is_sub else None
            output_path, filename, fmt = _output_for_mat_plot(
                self.mat_plot_2d, is_sub, "point_dataset_positions"
            )

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

        # nasty hack to ensure subplot index between 2d and 1d plots are synced.
        if (
            self.mat_plot_1d.subplot_index is not None
            and self.mat_plot_2d.subplot_index is not None
        ):
            self.mat_plot_1d.subplot_index = max(
                self.mat_plot_1d.subplot_index, self.mat_plot_2d.subplot_index
            )

        if fluxes:
            if self.dataset.fluxes is not None:
                is_sub = self.mat_plot_1d.is_for_subplot
                ax = self.mat_plot_1d.setup_subplot() if is_sub else None
                output_path, filename, fmt = _output_for_mat_plot(
                    self.mat_plot_1d, is_sub, "point_dataset_fluxes"
                )

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

    def subplot(
        self,
        positions: bool = False,
        fluxes: bool = False,
        auto_filename="subplot_dataset_point",
    ):
        self._subplot_custom_plot(
            positions=positions,
            fluxes=fluxes,
            auto_labels=aplt.AutoLabels(filename=auto_filename),
        )

    def subplot_dataset(self):
        self.subplot(positions=True, fluxes=True)

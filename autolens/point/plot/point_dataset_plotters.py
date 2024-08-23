import autogalaxy.plot as aplt

from autolens.point.dataset import PointDataset
from autolens.plot.abstract_plotters import Plotter


class PointDatasetPlotter(Plotter):
    def __init__(
        self,
        dataset: PointDataset,
        mat_plot_1d: aplt.MatPlot1D = aplt.MatPlot1D(),
        visuals_1d: aplt.Visuals1D = aplt.Visuals1D(),
        include_1d: aplt.Include1D = aplt.Include1D(),
        mat_plot_2d: aplt.MatPlot2D = aplt.MatPlot2D(),
        visuals_2d: aplt.Visuals2D = aplt.Visuals2D(),
        include_2d: aplt.Include2D = aplt.Include2D(),
    ):
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.dataset = dataset

    def get_visuals_1d(self) -> aplt.Visuals1D:
        return self.visuals_1d

    def get_visuals_2d(self) -> aplt.Visuals2D:
        return self.visuals_2d

    def figures_2d(self, positions: bool = False, fluxes: bool = False):
        if positions:
            self.mat_plot_2d.plot_grid(
                grid=self.dataset.positions,
                y_errors=self.dataset.positions_noise_map,
                x_errors=self.dataset.positions_noise_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=f"{self.dataset.name} Positions",
                    filename="point_dataset_positions",
                ),
                buffer=0.1,
            )

        # nasty hack to ensure subplot index between 2d and 1d plots are syncs. Need a refactor that mvoes subplot
        # functionality out of mat_plot and into plotter.

        if (
            self.mat_plot_1d.subplot_index is not None
            and self.mat_plot_2d.subplot_index is not None
        ):
            self.mat_plot_1d.subplot_index = max(
                self.mat_plot_1d.subplot_index, self.mat_plot_2d.subplot_index
            )

        if fluxes:
            if self.dataset.fluxes is not None:
                self.mat_plot_1d.plot_yx(
                    y=self.dataset.fluxes,
                    y_errors=self.dataset.fluxes_noise_map,
                    visuals_1d=self.get_visuals_1d(),
                    auto_labels=aplt.AutoLabels(
                        title=f" {self.dataset.name} Fluxes",
                        filename="point_dataset_fluxes",
                        xlabel="Point Number",
                    ),
                    plot_axis_type_override="errorbar",
                )

    def subplot(
        self,
        positions: bool = False,
        fluxes: bool = False,
        auto_filename="subplot_dataset",
    ):
        self._subplot_custom_plot(
            positions=positions,
            fluxes=fluxes,
            auto_labels=aplt.AutoLabels(filename=auto_filename),
        )

    def subplot_dataset(self):
        self.subplot(positions=True, fluxes=True)

import autogalaxy.plot as aplt

from autolens.point.dataset import PointDataset
from autolens.plot.abstract_plotters import Plotter


class PointDatasetPlotter(Plotter):
    def __init__(
        self,
        dataset: PointDataset,
        mat_plot_1d: aplt.MatPlot1D = None,
        visuals_1d: aplt.Visuals1D = None,
        mat_plot_2d: aplt.MatPlot2D = None,
        visuals_2d: aplt.Visuals2D = None,
    ):
        """
        Plots the attributes of `PointDataset` objects using the matplotlib methods and functions functions which
        customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `Imaging` and plotted via the visuals object.

        Parameters
        ----------
        dataset
            The imaging dataset the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        """
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
        )

        self.dataset = dataset

    def figures_2d(self, positions: bool = False, fluxes: bool = False):
        """
        Plots the individual attributes of the plotter's `PointDataset` object in 2D.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        positions
            If `True`, the dataset's positions are plotted on the figure compared to the model positions.
        fluxes
            If `True`, the dataset's fluxes are plotted on the figure compared to the model fluxes.
        """
        if positions:
            self.mat_plot_2d.plot_grid(
                grid=self.dataset.positions,
                y_errors=self.dataset.positions_noise_map,
                x_errors=self.dataset.positions_noise_map,
                visuals_2d=self.visuals_2d,
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
                    visuals_1d=self.visuals_1d,
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
        auto_filename="subplot_dataset_point",
    ):
        self._subplot_custom_plot(
            positions=positions,
            fluxes=fluxes,
            auto_labels=aplt.AutoLabels(filename=auto_filename),
        )

    def subplot_dataset(self):
        self.subplot(positions=True, fluxes=True)

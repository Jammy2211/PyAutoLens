import autogalaxy.plot as aplt

from autolens.plot.abstract_plotters import Plotter
from autolens.point.fit.dataset import FitPointDataset


class FitPointDatasetPlotter(Plotter):
    def __init__(
        self,
        fit: FitPointDataset,
        mat_plot_1d: aplt.MatPlot1D = None,
        visuals_1d: aplt.Visuals1D = None,
        mat_plot_2d: aplt.MatPlot2D = None,
        visuals_2d: aplt.Visuals2D = None,
    ):
        """
        Plots the attributes of `FitPointDataset` objects using matplotlib methods and functions which customize the
        plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `FitImaging` and plotted via the visuals object.

        Parameters
        ----------
        fit
            The fit to a point source dataset, which includes the data, model positions and other quantities which can
            be plotted like the residual_map and chi-squared map.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make the plot.
        visuals_2d
            Contains visuals that can be overlaid on the plot.
        """
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
        )

        self.fit = fit

    def figures_2d(self, positions: bool = False, fluxes: bool = False):
        """
        Plots the individual attributes of the plotter's `FitPointDataset` object in 2D.

        The API is such that every plottable attribute of the `FitPointDataset` object is an input parameter of type
        bool of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        positions
            If `True`, the dataset's positions are plotted on the figure compared to the model positions.
        fluxes
            If `True`, the dataset's fluxes are plotted on the figure compared to the model fluxes.
        """
        if positions:
            visuals_2d = self.visuals_2d

            visuals_2d += visuals_2d.__class__(
                multiple_images=self.fit.positions.model_data
            )

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

                extent = [y_min, y_max, x_min, x_max]

                self.mat_plot_2d.axis.kwargs["extent"] = extent

            self.mat_plot_2d.plot_grid(
                grid=self.fit.dataset.positions,
                y_errors=self.fit.dataset.positions_noise_map,
                x_errors=self.fit.dataset.positions_noise_map,
                visuals_2d=visuals_2d,
                auto_labels=aplt.AutoLabels(
                    title=f"{self.fit.dataset.name} Fit Positions",
                    filename="fit_point_positions",
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
            if self.fit.dataset.fluxes is not None:
                visuals_1d = self.visuals_1d

                # Dataset may have flux but model may not

                try:
                    visuals_1d += visuals_1d.__class__(
                        model_fluxes=self.fit.flux.model_fluxes.array
                    )
                except AttributeError:
                    pass

                self.mat_plot_1d.plot_yx(
                    y=self.fit.dataset.fluxes,
                    y_errors=self.fit.dataset.fluxes_noise_map,
                    visuals_1d=visuals_1d,
                    auto_labels=aplt.AutoLabels(
                        title=f" {self.fit.dataset.name} Fit Fluxes",
                        filename="fit_point_fluxes",
                        xlabel="Point Number",
                    ),
                    plot_axis_type_override="errorbar",
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

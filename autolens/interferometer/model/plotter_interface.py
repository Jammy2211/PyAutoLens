from autogalaxy.interferometer.model.plotter_interface import (
    PlotterInterfaceInterferometer as AgPlotterInterfaceInterferometer,
)

from autogalaxy.interferometer.model.plotter_interface import fits_to_fits

from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.interferometer.plot.fit_interferometer_plotters import (
    FitInterferometerPlotter,
)
from autolens.analysis.plotter_interface import PlotterInterface

from autolens.analysis.plotter_interface import plot_setting


class PlotterInterfaceInterferometer(PlotterInterface):
    interferometer = AgPlotterInterfaceInterferometer.interferometer

    def fit_interferometer(
        self,
        fit: FitInterferometer,
        quick_update: bool = False,
    ):
        """
        Visualizes a `FitInterferometer` object, which fits an interferometer dataset.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitInterferometer` of the non-linear search which is used to plot the fit.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_interferometer"], name=name)

        output = self.output_from()

        fit_plotter = FitInterferometerPlotter(
            fit=fit,
            output=output,
        )

        if should_plot("subplot_fit"):
            fit_plotter.subplot_fit()

        if should_plot("subplot_fit_dirty_images"):
            fit_plotter.subplot_fit_dirty_images()

        if quick_update:
            return

        if should_plot("subplot_fit_real_space"):
            fit_plotter.subplot_fit_real_space()

        output = self.output_from()

        fit_plotter = FitInterferometerPlotter(
            fit=fit,
            output=output,
        )

        if plot_setting(section="inversion", name="subplot_mappings"):
            fit_plotter.subplot_mappings_of_plane(
                plane_index=len(fit.tracer.planes) - 1
            )

        fits_to_fits(
            should_plot=should_plot,
            image_path=self.image_path,
            fit=fit,
        )

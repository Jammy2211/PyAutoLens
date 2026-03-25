from autogalaxy.interferometer.model.plotter import (
    PlotterInterferometer as AgPlotterInterferometer,
)

from autogalaxy.interferometer.model.plotter import fits_to_fits
from autogalaxy.interferometer.plot import fit_interferometer_plots as ag_fit_interferometer_plots

from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.interferometer.plot.fit_interferometer_plots import (
    subplot_fit,
    subplot_fit_real_space,
)
from autolens.analysis.plotter import Plotter

from autolens.analysis.plotter import plot_setting


class PlotterInterferometer(Plotter):
    interferometer = AgPlotterInterferometer.interferometer

    def fit_interferometer(
        self,
        fit: FitInterferometer,
        quick_update: bool = False,
    ):
        """
        Visualizes a `FitInterferometer` object.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitInterferometer` of the non-linear search.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_interferometer"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        if should_plot("subplot_fit"):
            subplot_fit(fit, output_path=output_path, output_format=fmt)

        if should_plot("subplot_fit_dirty_images") or quick_update:
            ag_fit_interferometer_plots.subplot_fit_dirty_images(
                fit=fit,
                output_path=self.image_path,
                output_format=self.fmt,
            )

        if quick_update:
            return

        if should_plot("subplot_fit_real_space"):
            subplot_fit_real_space(fit, output_path=output_path, output_format=fmt)

        fits_to_fits(
            should_plot=should_plot,
            image_path=self.image_path,
            fit=fit,
        )

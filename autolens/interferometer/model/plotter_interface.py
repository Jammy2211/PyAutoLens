from autogalaxy.interferometer.model.plotter_interface import (
    PlotterInterfaceInterferometer as AgPlotterInterfaceInterferometer,
)

from autogalaxy.interferometer.model.plotter_interface import fits_to_fits

from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.interferometer.plot.fit_interferometer_plots import (
    subplot_fit,
    subplot_fit_real_space,
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

        if should_plot("subplot_fit_dirty_images"):
            # Use the autoarray FitInterferometerMeta plotter for dirty images subplot
            try:
                import autogalaxy.plot as aplt
                from autoarray.fit.plot.fit_interferometer_plotters import FitInterferometerPlotterMeta
                output = self.output_from()
                meta_plotter = FitInterferometerPlotterMeta(
                    fit=fit,
                    output=output,
                )
                meta_plotter.subplot_fit_dirty_images()
            except Exception:
                pass

        if quick_update:
            return

        if should_plot("subplot_fit_real_space"):
            subplot_fit_real_space(fit, output_path=output_path, output_format=fmt)

        if plot_setting(section="inversion", name="subplot_mappings"):
            try:
                import autogalaxy.plot as aplt
                inversion_plotter = aplt.InversionPlotter(
                    inversion=fit.inversion,
                    mat_plot_2d=aplt.MatPlot2D(
                        output=aplt.Output(path=self.image_path, format=fmt),
                    ),
                )
                inversion_plotter.subplot_of_mapper(
                    mapper_index=0,
                    auto_filename="subplot_mappings_0",
                )
            except (IndexError, AttributeError, TypeError, Exception):
                pass

        fits_to_fits(
            should_plot=should_plot,
            image_path=self.image_path,
            fit=fit,
        )

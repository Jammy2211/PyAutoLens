from typing import Optional

import autoarray.plot as aplt

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
        visuals_2d_of_planes_list: Optional[aplt.Visuals2D] = None,
        quick_update: bool = False,
    ):
        """
        Visualizes a `FitInterferometer` object, which fits an interferometer dataset.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        is the output folder of the non-linear search.

        Visualization includes a subplot of individual images of attributes of the `FitInterferometer` (e.g. the model
        data,residual map) and .fits files containing its attributes grouped together.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `fit` and `fit_interferometer` headers.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitInterferometer` of the non-linear search which is used to plot the fit.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_interferometer"], name=name)

        mat_plot_1d = self.mat_plot_1d_from()
        mat_plot_2d = self.mat_plot_2d_from()

        fit_plotter = FitInterferometerPlotter(
            fit=fit,
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_fit"):
            fit_plotter.subplot_fit()

        if should_plot("subplot_fit_dirty_images"):
            fit_plotter.subplot_fit_dirty_images()

        if quick_update:
            return

        if should_plot("subplot_fit_real_space"):
            fit_plotter.subplot_fit_real_space()

        mat_plot_1d = self.mat_plot_1d_from()
        mat_plot_2d = self.mat_plot_2d_from()

        fit_plotter = FitInterferometerPlotter(
            fit=fit,
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
            visuals_2d_of_planes_list=visuals_2d_of_planes_list,
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

import matplotlib.pyplot as plt
import numpy as np
from typing import List

from autogalaxy.imaging.model.plotter_interface import PlotterInterfaceImaging as AgPlotterInterfaceImaging
from autogalaxy.imaging.model.plotter_interface import fits_to_fits

from autolens.analysis.plotter_interface import PlotterInterface
from autolens.imaging.fit_imaging import FitImaging
from autolens.imaging.plot.fit_imaging_plots import (
    subplot_fit,
    subplot_fit_log10,
    subplot_of_planes,
    subplot_tracer_from_fit,
    subplot_fit_combined,
    subplot_fit_combined_log10,
)

from autolens.analysis.plotter_interface import plot_setting


class PlotterInterfaceImaging(PlotterInterface):

    imaging = AgPlotterInterfaceImaging.imaging
    imaging_combined = AgPlotterInterfaceImaging.imaging_combined

    def fit_imaging(
        self, fit: FitImaging, quick_update: bool = False
    ):
        """
        Visualizes a `FitImaging` object, which fits an imaging dataset.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitImaging` of the non-linear search.
        quick_update
            If True only the essential subplot_fit is output.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        plane_indexes_to_plot = [i for i in fit.tracer.plane_indexes_with_images if i != 0]

        if should_plot("subplot_fit") or quick_update:

            if len(fit.tracer.planes) > 2:
                for plane_index in plane_indexes_to_plot:
                    subplot_fit(fit, output_path=output_path, output_format=fmt,
                                plane_index=plane_index)
            else:
                subplot_fit(fit, output_path=output_path, output_format=fmt)

        if quick_update:
            return

        if plot_setting(section="tracer", name="subplot_tracer"):
            subplot_tracer_from_fit(fit, output_path=output_path, output_format=fmt)

        if should_plot("subplot_fit_log10"):
            try:
                if len(fit.tracer.planes) > 2:
                    for plane_index in plane_indexes_to_plot:
                        subplot_fit_log10(fit, output_path=output_path, output_format=fmt,
                                          plane_index=plane_index)
                else:
                    subplot_fit_log10(fit, output_path=output_path, output_format=fmt)
            except ValueError:
                pass

        if should_plot("subplot_of_planes"):
            subplot_of_planes(fit, output_path=output_path, output_format=fmt)

        if plot_setting(section="inversion", name="subplot_mappings"):
            try:
                import autogalaxy.plot as aplt
                output = self.output_from()
                inversion_plotter = aplt.InversionPlotter(
                    inversion=fit.inversion,
                    mat_plot_2d=aplt.MatPlot2D(
                        output=aplt.Output(path=self.image_path, format=fmt),
                    ),
                )
                pixelization_index = 0
                inversion_plotter.subplot_of_mapper(
                    mapper_index=pixelization_index,
                    auto_filename=f"subplot_mappings_{pixelization_index}",
                )
            except (IndexError, AttributeError, TypeError, Exception):
                pass

        fits_to_fits(should_plot=should_plot, image_path=self.image_path, fit=fit)

    def fit_imaging_combined(
            self,
            fit_list: List[FitImaging],
            quick_update: bool = False,
    ):
        """
        Output visualization of all `FitImaging` objects in a summed combined analysis.

        Parameters
        ----------
        fit_list
            The list of imaging fits which are visualized.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        if should_plot("subplot_fit") or quick_update:
            subplot_fit_combined(fit_list, output_path=output_path, output_format=fmt)

            if quick_update:
                return

            subplot_fit_combined_log10(fit_list, output_path=output_path, output_format=fmt)

import matplotlib.pyplot as plt
import numpy as np
from typing import List

import autogalaxy.plot as aplt
from autogalaxy.plot.abstract_plotters import _save_subplot

from autogalaxy.imaging.model.plotter_interface import PlotterInterfaceImaging as AgPlotterInterfaceImaging

from autogalaxy.imaging.model.plotter_interface import fits_to_fits

from autolens.analysis.plotter_interface import PlotterInterface
from autolens.imaging.fit_imaging import FitImaging
from autolens.imaging.plot.fit_imaging_plotters import FitImagingPlotter

from autolens.analysis.plotter_interface import plot_setting


class PlotterInterfaceImaging(PlotterInterface):

    imaging = AgPlotterInterfaceImaging.imaging
    imaging_combined = AgPlotterInterfaceImaging.imaging_combined

    def fit_imaging(
        self, fit: FitImaging, quick_update: bool = False
    ):
        """
        Visualizes a `FitImaging` object, which fits an imaging dataset.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        points to the search's results folder and this function visualizes the maximum log likelihood `FitImaging`
        inferred by the search so far.

        Visualization includes a subplot of individual images of attributes of the `FitImaging` (e.g. the model data,
        residual map) and .fits files containing its attributes grouped together.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `fit` and `fit_imaging` header.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitImaging` of the non-linear search which is used to plot the fit.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        output = self.output_from()

        fit_plotter = FitImagingPlotter(
            fit=fit, output=output,
        )

        plane_indexes_to_plot = [i for i in fit.tracer.plane_indexes_with_images if i != 0]

        if should_plot("subplot_fit") or quick_update:

            if len(fit.tracer.planes) > 2:
                for plane_index in plane_indexes_to_plot:
                    fit_plotter.subplot_fit(plane_index=plane_index)
            else:
                fit_plotter.subplot_fit()

        if quick_update:
            return

        if plot_setting(section="tracer", name="subplot_tracer"):

            output = self.output_from()

            fit_plotter = FitImagingPlotter(
                fit=fit, output=output,
            )

            fit_plotter.subplot_tracer()

        if should_plot("subplot_fit_log10"):

            try:
                if len(fit.tracer.planes) > 2:
                    for plane_index in plane_indexes_to_plot:
                        fit_plotter.subplot_fit_log10(plane_index=plane_index)
                else:
                    fit_plotter.subplot_fit_log10()
            except ValueError:
                pass

        if should_plot("subplot_of_planes"):
            fit_plotter.subplot_of_planes()

        if plot_setting(section="inversion", name="subplot_mappings"):
            try:
                fit_plotter.subplot_mappings_of_plane(plane_index=len(fit.tracer.planes) - 1)
            except IndexError:
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

        output = self.output_from()

        fit_plotter_list = [
            FitImagingPlotter(fit=fit, output=output)
            for fit in fit_list
        ]

        if should_plot("subplot_fit") or quick_update:

            def make_subplot_fit(filename_suffix, use_log10=False):
                n_fits = len(fit_plotter_list)
                n_cols = 6
                fig, axes = plt.subplots(n_fits, n_cols, figsize=(7 * n_cols, 7 * n_fits))
                if n_fits == 1:
                    axes = [axes]
                axes = np.array(axes)

                final_plane_index = len(fit_list[0].tracer.planes) - 1

                for row, (plotter, fit) in enumerate(zip(fit_plotter_list, fit_list)):
                    if use_log10:
                        plotter.use_log10 = True

                    row_axes = axes[row] if n_fits > 1 else axes[0]

                    plotter._fit_imaging_meta_plotter._plot_array(
                        fit.data, "data", "Data", ax=row_axes[0]
                    )

                    try:
                        subtracted = fit.subtracted_images_of_planes_list[1]
                        plotter._fit_imaging_meta_plotter._plot_array(
                            subtracted, "subtracted_image", "Subtracted Image", ax=row_axes[1]
                        )
                    except (IndexError, AttributeError):
                        row_axes[1].axis("off")

                    try:
                        lens_model = fit.model_images_of_planes_list[0]
                        plotter._fit_imaging_meta_plotter._plot_array(
                            lens_model, "lens_model_image", "Lens Model Image", ax=row_axes[2]
                        )
                    except (IndexError, AttributeError):
                        row_axes[2].axis("off")

                    try:
                        source_model = fit.model_images_of_planes_list[final_plane_index]
                        plotter._fit_imaging_meta_plotter._plot_array(
                            source_model, "source_model_image", "Source Model Image", ax=row_axes[3]
                        )
                    except (IndexError, AttributeError):
                        row_axes[3].axis("off")

                    try:
                        plotter.figures_2d_of_planes(
                            plane_index=final_plane_index, plane_image=True, ax=row_axes[4]
                        )
                    except Exception:
                        row_axes[4].axis("off")

                    plotter._fit_imaging_meta_plotter._plot_array(
                        fit.normalized_residual_map, "normalized_residual_map", "Normalized Residual Map", ax=row_axes[5]
                    )

                plt.tight_layout()
                _save_subplot(fig, output, filename_suffix)

            make_subplot_fit(filename_suffix="subplot_fit_combined")

            if quick_update:
                return

            make_subplot_fit(filename_suffix="fit_combined_log10", use_log10=True)

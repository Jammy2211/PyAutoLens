from typing import List, Optional

import autoarray.plot as aplt

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
        self, fit: FitImaging, visuals_2d_of_planes_list : Optional[aplt.Visuals2D] = None, quick_update: bool = False
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

        mat_plot_2d = self.mat_plot_2d_from(quick_update=quick_update)

        fit_plotter = FitImagingPlotter(
            fit=fit, mat_plot_2d=mat_plot_2d, visuals_2d_of_planes_list=visuals_2d_of_planes_list,
        )

        plane_indexes_to_plot = [i for i in fit.tracer.plane_indexes_with_images if i != 0]

        if should_plot("subplot_fit") or quick_update:

            # This loop means that multiple subplot_fit objects are output for a double source plane lens.

            if len(fit.tracer.planes) > 2:
                for plane_index in plane_indexes_to_plot:
                    fit_plotter.subplot_fit(plane_index=plane_index)
            else:
                fit_plotter.subplot_fit()

        if quick_update:
            return

        if plot_setting(section="tracer", name="subplot_tracer"):

            mat_plot_2d = self.mat_plot_2d_from()

            fit_plotter = FitImagingPlotter(
                fit=fit, mat_plot_2d=mat_plot_2d, visuals_2d_of_planes_list=visuals_2d_of_planes_list,
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
            visuals_2d_of_planes_list : Optional[aplt.Visuals2D] = None,
            quick_update: bool = False,
    ):
        """
        Output visualization of all `FitImaging` objects in a summed combined analysis, typically during or after a
        model-fit is performed.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        is the output folder of the non-linear search.

        Visualization includes a subplot of individual images of attributes of each fit (e.g. data, normalized
        residual-map) on a single subplot, such that the full suite of multiple datasets can be viewed on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `fit` and `fit_imaging` headers.

        Parameters
        ----------
        fit_list
            The list of imaging fits which are visualized.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        mat_plot_2d = self.mat_plot_2d_from(quick_update=quick_update)

        fit_plotter_list = [
            FitImagingPlotter(
                fit=fit, mat_plot_2d=mat_plot_2d, visuals_2d_of_planes_list=visuals_2d_of_planes_list,
            )
            for fit in fit_list
        ]

        subplot_columns = 6

        subplot_shape = (len(fit_list), subplot_columns)

        multi_plotter = aplt.MultiFigurePlotter(
            plotter_list=fit_plotter_list, subplot_shape=subplot_shape
        )

        if should_plot("subplot_fit") or quick_update:

            def make_subplot_fit(filename_suffix):

                multi_plotter.subplot_of_figures_multi(
                    func_name_list=["figures_2d"],
                    figure_name_list=[
                        "data",
                    ],
                    filename_suffix=filename_suffix,
                    number_subplots=len(fit_list) * subplot_columns,
                    close_subplot=False,
                )

                multi_plotter.subplot_of_figures_multi(
                    func_name_list=["figures_2d_of_planes"],
                    figure_name_list=[
                        "subtracted_image",
                    ],
                    filename_suffix=filename_suffix,
                    number_subplots=len(fit_list) * subplot_columns,
                    open_subplot=False,
                    close_subplot=False,
                    subplot_index_offset=1,
                    plane_index=1
                )

                multi_plotter.subplot_of_figures_multi(
                    func_name_list=["figures_2d_of_planes"],
                    figure_name_list=[
                        "model_image",
                    ],
                    filename_suffix=filename_suffix,
                    number_subplots=len(fit_list) * subplot_columns,
                    open_subplot=False,
                    close_subplot=False,
                    subplot_index_offset=2,
                    plane_index=0
                )

                multi_plotter.subplot_of_figures_multi(
                    func_name_list=["figures_2d_of_planes"],
                    figure_name_list=[
                        "model_image",
                    ],
                    filename_suffix=filename_suffix,
                    number_subplots=len(fit_list) * subplot_columns,
                    open_subplot=False,
                    close_subplot=False,
                    subplot_index_offset=3,
                    plane_index=len(fit_list[0].tracer.planes) - 1
                )

                multi_plotter.subplot_of_figures_multi(
                    func_name_list=["figures_2d_of_planes"],
                    figure_name_list=[
                        "plane_image",
                    ],
                    filename_suffix=filename_suffix,
                    number_subplots=len(fit_list) * subplot_columns,
                    open_subplot=False,
                    close_subplot=False,
                    subplot_index_offset=4,
                    plane_index=len(fit_list[0].tracer.planes) - 1
                )

                multi_plotter.subplot_of_figures_multi(
                    func_name_list=["figures_2d"],
                    figure_name_list=[
                        "normalized_residual_map",
                    ],
                    filename_suffix=filename_suffix,
                    number_subplots=len(fit_list) * subplot_columns,
                    subplot_index_offset=5,
                    open_subplot=False,
                )

            make_subplot_fit(filename_suffix="fit_combined")

            if quick_update:
                return

            for plotter in multi_plotter.plotter_list:
                plotter.mat_plot_2d.use_log10 = True

            make_subplot_fit(filename_suffix="fit_combined_log10")
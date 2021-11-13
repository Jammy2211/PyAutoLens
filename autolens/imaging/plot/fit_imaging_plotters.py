import numpy as np
from typing import Optional

import autogalaxy as ag
import autogalaxy.plot as aplt

from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta

from autolens.plot.abstract_plotters import Plotter
from autolens.imaging.fit_imaging import FitImaging
from autolens.lens.plot.ray_tracing_plotters import TracerPlotter


class FitImagingPlotter(Plotter):
    def __init__(
        self,
        fit: FitImaging,
        mat_plot_2d: aplt.MatPlot2D = aplt.MatPlot2D(),
        visuals_2d: aplt.Visuals2D = aplt.Visuals2D(),
        include_2d: aplt.Include2D = aplt.Include2D(),
    ):

        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.fit = fit

        self._fit_imaging_meta_plotter = FitImagingPlotterMeta(
            fit=self.fit,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

        self.figures_2d = self._fit_imaging_meta_plotter.figures_2d
        self.subplot = self._fit_imaging_meta_plotter.subplot
        self.subplot_fit_imaging = self._fit_imaging_meta_plotter.subplot_fit_imaging

    def get_visuals_2d(self) -> aplt.Visuals2D:
        return self.get_2d.via_fit_imaging_from(fit=self.fit)

    @property
    def tracer(self):
        return self.fit.tracer

    @property
    def tracer_plotter(self) -> TracerPlotter:
        return TracerPlotter(
            tracer=self.tracer,
            grid=self.fit.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
        )

    def inversion_plotter_of_plane(self, plane_index: int) -> aplt.InversionPlotter:

        inversion_plotter = aplt.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.tracer_plotter.get_visuals_2d_of_plane(
                plane_index=plane_index
            ),
            include_2d=self.include_2d,
        )
        inversion_plotter.visuals_2d.border = None
        return inversion_plotter

    def plane_indexes_from(self, plane_index: int):

        if plane_index is None:
            return range(len(self.fit.tracer.planes))
        return [plane_index]

    def figures_2d_of_planes(
        self,
        plane_index: Optional[int] = None,
        subtracted_image: bool = False,
        model_image: bool = False,
        plane_image: bool = False,
    ):
        """Plot the model data of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.

        Parameters
        -----------
        fit : autolens.lens.fitting.Fitter
            Class containing fit between the model data and observed lens data (including residual_map, chi_squared_map etc.)
        output_path : str
            The path where the data is output if the output_type is a file format (e.g. png, fits)
        output_format : str
            How the data is output. File formats (e.g. png, fits) output the data to harddisk. 'show' displays the data \
            in the python interpreter window.
        """

        plane_indexes = self.plane_indexes_from(plane_index=plane_index)

        for plane_index in plane_indexes:

            if subtracted_image:

                if "vmin" in self.mat_plot_2d.cmap.kwargs:
                    vmin_stored = True
                else:
                    self.mat_plot_2d.cmap.kwargs["vmin"] = 0.0
                    vmin_stored = False

                if "vmax" in self.mat_plot_2d.cmap.kwargs:
                    vmax_stored = True
                else:
                    self.mat_plot_2d.cmap.kwargs["vmax"] = np.max(
                        self.fit.model_images_of_planes[plane_index]
                    )
                    vmax_stored = False

                self.mat_plot_2d.plot_array(
                    array=self.fit.subtracted_images_of_planes[plane_index],
                    visuals_2d=self.get_visuals_2d(),
                    auto_labels=aplt.AutoLabels(
                        title=f"Subtracted Image of Plane {plane_index}",
                        filename=f"subtracted_image_of_plane_{plane_index}",
                    ),
                )

                if not vmin_stored:
                    self.mat_plot_2d.cmap.kwargs.pop("vmin")

                if not vmax_stored:
                    self.mat_plot_2d.cmap.kwargs.pop("vmax")

            if model_image:

                if self.fit.inversion is None or plane_index == 0:

                    self.mat_plot_2d.plot_array(
                        array=self.fit.model_images_of_planes[plane_index],
                        visuals_2d=self.get_visuals_2d(),
                        auto_labels=aplt.AutoLabels(
                            title=f"Model Image of Plane {plane_index}",
                            filename=f"model_image_of_plane_{plane_index}",
                        ),
                    )

                else:

                    inversion_plotter = self.inversion_plotter_of_plane(plane_index=0)
                    inversion_plotter.figures_2d(reconstructed_image=True)

            if plane_image:

                if not self.tracer.planes[plane_index].has_pixelization:

                    self.tracer_plotter.figures_2d_of_planes(
                        plane_image=True, plane_index=plane_index
                    )

                elif self.tracer.planes[plane_index].has_pixelization:

                    inversion_plotter = self.inversion_plotter_of_plane(plane_index=1)
                    inversion_plotter.figures_2d_of_mapper(
                        mapper_index=0, reconstruction=True
                    )

    def subplot_of_planes(self, plane_index: Optional[int] = None):
        """Plot the model data of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.

        Parameters
        -----------
        fit : autolens.lens.fitting.Fitter
            Class containing fit between the model data and observed lens data (including residual_map, chi_squared_map etc.)
        output_path : str
            The path where the data is output if the output_type is a file format (e.g. png, fits)
        output_filename : str
            The name of the file that is output, if the output_type is a file format (e.g. png, fits)
        output_format : str
            How the data is output. File formats (e.g. png, fits) output the data to harddisk. 'show' displays the data \
            in the python interpreter window.
        """

        plane_indexes = self.plane_indexes_from(plane_index=plane_index)

        for plane_index in plane_indexes:

            self.open_subplot_figure(number_subplots=4)

            self.figures_2d(image=True)
            self.figures_2d_of_planes(subtracted_image=True, plane_index=plane_index)
            self.figures_2d_of_planes(model_image=True, plane_index=plane_index)
            self.figures_2d_of_planes(plane_image=True, plane_index=plane_index)

            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename=f"subplot_of_plane_{plane_index}"
            )
            self.close_subplot_figure()

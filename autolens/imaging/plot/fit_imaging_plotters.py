import numpy as np
from typing import Optional

import autogalaxy as ag
import autogalaxy.plot as aplt

from autoarray.fit.plot.fit_imaging_plotters import AbstractFitImagingPlotter

from autolens.imaging.fit_imaging import FitImaging
from autolens.lens.plot.ray_tracing_plotters import TracerPlotter


class FitImagingPlotter(AbstractFitImagingPlotter):
    def __init__(
        self,
        fit: FitImaging,
        mat_plot_2d: aplt.MatPlot2D = aplt.MatPlot2D(),
        visuals_2d: aplt.Visuals2D = aplt.Visuals2D(),
        include_2d: aplt.Include2D = aplt.Include2D(),
    ):

        super().__init__(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

    @property
    def visuals_with_include_2d(self) -> aplt.Visuals2D:
        """
        Extracts from a `Structure` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From an `AbstractStructure` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.

        Parameters
        ----------
        structure : abstract_structure.AbstractStructure
            The structure whose attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """

        visuals_2d = super().visuals_with_include_2d

        #      visuals_2d.mask = None

        return visuals_2d + visuals_2d.__class__(
            light_profile_centres=self.extract_2d(
                "light_profile_centres",
                self.tracer.planes[0].extract_attribute(
                    cls=ag.lp.LightProfile, attr_name="centre"
                ),
            ),
            mass_profile_centres=self.extract_2d(
                "mass_profile_centres",
                self.tracer.planes[0].extract_attribute(
                    cls=ag.mp.MassProfile, attr_name="centre"
                ),
            ),
            critical_curves=self.extract_2d(
                "critical_curves",
                self.tracer.critical_curves_from(grid=self.fit.grid),
                "critical_curves",
            ),
        )

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
            visuals_2d=self.tracer_plotter.visuals_with_include_2d_of_plane(
                plane_index=plane_index
            ),
            include_2d=self.include_2d,
        )
        inversion_plotter.visuals_2d.border = None
        return inversion_plotter

    def figures_2d(
        self,
        image: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
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

        super().figures_2d(
            image=image,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_image=model_image,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
        )

    def plane_indexes_from(self, plane_index: int):

        if plane_index is None:
            return range(len(self.fit.tracer.planes))
        else:
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
                    visuals_2d=self.visuals_with_include_2d,
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
                        visuals_2d=self.visuals_with_include_2d,
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

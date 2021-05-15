from autoarray.plot.mat_wrap import mat_plot as mp
from autoarray.plot import inversion_plotters, fit_imaging_plotters
from autolens.plot import ray_tracing_plotters
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autogalaxy.profiles import light_profiles, mass_profiles
from autolens.fit import fit_imaging

import numpy as np


class FitImagingPlotter(fit_imaging_plotters.AbstractFitImagingPlotter):
    def __init__(
        self,
        fit: fit_imaging.FitImaging,
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):

        super().__init__(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

    @property
    def visuals_with_include_2d(self) -> lensing_visuals.Visuals2D:
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
                    cls=light_profiles.LightProfile, attr_name="centre"
                ),
            ),
            mass_profile_centres=self.extract_2d(
                "mass_profile_centres",
                self.tracer.planes[0].extract_attribute(
                    cls=mass_profiles.MassProfile, attr_name="centre"
                ),
            ),
            critical_curves=self.extract_2d(
                "critical_curves",
                self.tracer.critical_curves_from_grid(grid=self.fit.grid),
                "critical_curves",
            ),
        )

    @property
    def tracer(self):
        return self.fit.tracer

    @property
    def tracer_plotter(self):
        return ray_tracing_plotters.TracerPlotter(
            tracer=self.tracer,
            grid=self.fit.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
        )

    def inversion_plotter_of_plane(self, plane_index):

        inversion_plotter = inversion_plotters.InversionPlotter(
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
        image=False,
        noise_map=False,
        signal_to_noise_map=False,
        model_image=False,
        residual_map=False,
        normalized_residual_map=False,
        chi_squared_map=False,
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

        super(FitImagingPlotter, self).figures_2d(
            image=image,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_image=model_image,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
        )

    def plane_indexes_from_plane_index(self, plane_index):

        if plane_index is None:
            return range(len(self.fit.tracer.planes))
        else:
            return [plane_index]

    def figures_2d_of_planes(
        self,
        subtracted_image=False,
        model_image=False,
        plane_image=False,
        plane_index=None,
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

        plane_indexes = self.plane_indexes_from_plane_index(plane_index=plane_index)

        for plane_index in plane_indexes:

            if subtracted_image:

                try:
                    vmin = self.mat_plot_2d.cmap.kwargs["vmin"]
                    vmin_store = vmin
                except KeyError:
                    vmin = 0.0
                    vmin_store = None

                try:
                    vmax = self.mat_plot_2d.cmap.kwargs["vmax"]
                    vmax_store = vmax
                except KeyError:
                    vmax = np.max(self.fit.model_images_of_planes[plane_index])
                    vmax_store = None

                self.mat_plot_2d.cmap.kwargs["vmin"] = vmin
                self.mat_plot_2d.cmap.kwargs["vmax"] = vmax

                self.mat_plot_2d.plot_array(
                    array=self.fit.subtracted_images_of_planes[plane_index],
                    visuals_2d=self.visuals_with_include_2d,
                    auto_labels=mp.AutoLabels(
                        title=f"Subtracted Image of Plane {plane_index}",
                        filename=f"subtracted_image_of_plane_{plane_index}",
                    ),
                )

                self.mat_plot_2d.cmap.kwargs["vmin"] = vmin_store
                self.mat_plot_2d.cmap.kwargs["vmax"] = vmax_store

            if model_image:

                if self.fit.inversion is None or plane_index == 0:

                    self.mat_plot_2d.plot_array(
                        array=self.fit.model_images_of_planes[plane_index],
                        visuals_2d=self.visuals_with_include_2d,
                        auto_labels=mp.AutoLabels(
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
                    inversion_plotter.figures_2d(reconstruction=True)

    def subplot_of_planes(self, plane_index=None):
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

        plane_indexes = self.plane_indexes_from_plane_index(plane_index=plane_index)

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

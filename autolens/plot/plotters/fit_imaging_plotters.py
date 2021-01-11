from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.plotters import inversion_plotters
from autoarray.plot.plotters import fit_imaging_plotters
from autolens.plot.plotters import ray_tracing_plotters
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autolens.fit import fit as f

import numpy as np


class FitImagingPlotter(fit_imaging_plotters.AbstractFitImagingPlotter):
    def __init__(
        self,
        fit: f.FitImaging,
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

        visuals_2d = super(FitImagingPlotter, self).visuals_with_include_2d

        visuals_2d.mask = None

        return visuals_2d + visuals_2d.__class__(
            light_profile_centres=self.extract_2d(
                "light_profile_centres", self.tracer.planes[0].light_profile_centres
            ),
            mass_profile_centres=self.extract_2d(
                "mass_profile_centres", self.tracer.planes[0].mass_profile_centres
            ),
            critical_curves=self.extract_2d(
                "critical_curves", self.tracer.critical_curves, "critical_curves"
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

    @property
    def inversion_plotter(self):
        inversion_plotter = inversion_plotters.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.tracer_plotter.visuals_with_include_2d_of_plane(
                plane_index=1
            ),
            include_2d=self.include_2d,
        )
        inversion_plotter.visuals_2d.border = None
        return inversion_plotter

    @abstract_plotters.for_figure_with_index
    def figure_subtracted_image_of_plane(self, plane_index):
        """Plot the model image of a specific plane of a lens fit.
    
        Set *autolens.datas.arrays.plotter.plotter* for a description of all input parameters not described below.
    
        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
        image_index : int
            The index of the datas in the datas-set of which the model image is plotted.
        plane_indexes : int
            The plane from which the model image is generated.
        """

        if self.tracer.total_planes > 1:

            other_planes_model_images = [
                model_image
                for i, model_image in enumerate(self.fit.model_images_of_planes)
                if i != plane_index
            ]

            subtracted_image = self.fit.image - sum(other_planes_model_images)

        else:

            subtracted_image = self.fit.image

        try:
            vmin = self.mat_plot_2d.cmap.kwargs["vmin"]
        except KeyError:
            vmin = None

        try:
            vmax = self.mat_plot_2d.cmap.kwargs["vmax"]
        except KeyError:
            vmax = None

        self.mat_plot_2d.cmap.kwargs["vmin"] = np.max(
            self.fit.model_images_of_planes[plane_index]
        )
        self.mat_plot_2d.cmap.kwargs["vmax"] = np.min(
            self.fit.model_images_of_planes[plane_index]
        )

        self.mat_plot_2d.plot_array(
            array=subtracted_image, visuals_2d=self.visuals_with_include_2d
        )

        self.mat_plot_2d.cmap.kwargs["vmin"] = vmin
        self.mat_plot_2d.cmap.kwargs["vmax"] = vmax

    @abstract_plotters.for_figure_with_index
    def figure_model_image_of_plane(self, plane_index):
        """Plot the model image of a specific plane of a lens fit.
    
        Set *autolens.datas.arrays.plotter.plotter* for a description of all input parameters not described below.
    
        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
        plane_indexes : [int]
            The plane from which the model image is generated.
        """

        if self.fit.inversion is None or plane_index == 0:

            self.mat_plot_2d.plot_array(
                array=self.fit.model_images_of_planes[plane_index],
                visuals_2d=self.visuals_with_include_2d,
            )

        else:

            self.inversion_plotter.figure_reconstructed_image()

    @abstract_plotters.for_figure_with_index
    def figure_plane_image_of_plane(self, plane_index):
        """Plot the model image of a specific plane of a lens fit.

        Set *autolens.datas.arrays.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
        plane_indexes : [int]
            The plane from which the model image is generated.
        """

        if not self.tracer.planes[plane_index].has_pixelization:

            self.tracer_plotter.figure_plane_image_of_plane(plane_index=plane_index)

        elif self.tracer.planes[plane_index].has_pixelization:

            self.inversion_plotter.figure_reconstruction()

    def figure_individuals(
        self,
        plot_image=False,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_image=False,
        plot_residual_map=False,
        plot_normalized_residual_map=False,
        plot_chi_squared_map=False,
        plot_subtracted_images_of_planes=False,
        plot_model_images_of_planes=False,
        plot_plane_images_of_planes=False,
    ):
        """Plot the model datas_ of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.

        Parameters
        -----------
        fit : autolens.lens.fitting.Fitter
            Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
        output_path : str
            The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
        output_format : str
            How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
            in the python interpreter window.
        """

        super(FitImagingPlotter, self).figure_individuals(
            plot_image=plot_image,
            plot_noise_map=plot_noise_map,
            plot_signal_to_noise_map=plot_signal_to_noise_map,
            plot_model_image=plot_model_image,
            plot_residual_map=plot_residual_map,
            plot_normalized_residual_map=plot_normalized_residual_map,
            plot_chi_squared_map=plot_chi_squared_map,
        )

        if plot_subtracted_images_of_planes:

            for plane_index in range(self.tracer.total_planes):
                self.figure_subtracted_image_of_plane(plane_index=plane_index)

        if plot_model_images_of_planes:

            for plane_index in range(self.tracer.total_planes):
                self.figure_model_image_of_plane(plane_index=plane_index)

        if plot_plane_images_of_planes:

            for plane_index in range(self.tracer.total_planes):
                self.figure_plane_image_of_plane(plane_index=plane_index)

    def subplots_of_all_planes(self):

        for plane_index in range(self.tracer.total_planes):

            if (
                self.tracer.planes[plane_index].has_light_profile
                or self.tracer.planes[plane_index].has_pixelization
            ):
                self.subplot_of_plane(plane_index=plane_index)

    @abstract_plotters.for_subplot_with_index
    def subplot_of_plane(self, plane_index):
        """Plot the model datas_ of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.

        Parameters
        -----------
        fit : autolens.lens.fitting.Fitter
            Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
        output_path : str
            The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
        output_filename : str
            The name of the file that is output, if the output_type is a file format (e.g. png, fits)
        output_format : str
            How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
            in the python interpreter window.
        """

        number_subplots = 4

        self.open_subplot_figure(number_subplots=number_subplots)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        self.figure_image()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=2)

        self.figure_subtracted_image_of_plane(plane_index=plane_index)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=3)

        self.figure_model_image_of_plane(plane_index=plane_index)

        if not self.tracer.planes[plane_index].has_pixelization:

            self.setup_subplot(number_subplots=number_subplots, subplot_index=4)

        else:

            aspect_inv = self.mat_plot_2d.figure.aspect_for_subplot_from_grid(
                grid=self.fit.inversion.mapper.source_full_grid
            )

            self.setup_subplot(
                number_subplots=number_subplots,
                subplot_index=4,
                aspect=float(aspect_inv),
            )

        self.figure_plane_image_of_plane(plane_index=plane_index)

        self.mat_plot_2d.output.subplot_to_figure()

        self.mat_plot_2d.figure.close()

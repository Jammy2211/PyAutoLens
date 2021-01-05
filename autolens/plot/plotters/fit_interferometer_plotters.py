from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.plotters import fit_interferometer_plotters
from autoarray.plot.plotters import inversion_plotters
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autogalaxy.plot.plotters import plane_plotters
from autolens.plot.plotters import ray_tracing_plotters
from autolens.fit import fit as f


class FitInterferometerPlotter(fit_interferometer_plotters.FitInterferometerPlotter):
    def __init__(
        self,
        fit: f.FitInterferometer,
        mat_plot_1d: lensing_mat_plot.MatPlot1D = lensing_mat_plot.MatPlot1D(),
        visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
        include_1d: lensing_include.Include1D = lensing_include.Include1D(),
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):

        super().__init__(
            fit=fit,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

    @property
    def visuals_with_include_2d(self) -> "vis.Visuals2D":
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

        visuals_2d = super(FitInterferometerPlotter, self).visuals_with_include_2d

        visuals_2d.mask = None

        return visuals_2d + lensing_visuals.Visuals2D(
            light_profile_centres=self.extract_2d(
                "light_profile_centres", self.tracer.planes[0].light_profile_centres
            ),
            mass_profile_centres=self.extract_2d(
                "mass_profile_centres", self.tracer.planes[0].mass_profile_centres
            ),
            critical_curves=self.extract_2d(
                "critical_curves", self.tracer.critical_curves
            ),
        )

    @property
    def tracer(self):
        return self.fit.tracer

    @property
    def tracer_plotter(self):
        return ray_tracing_plotters.TracerPlotter(
            tracer=self.tracer,
            grid=self.fit.masked_interferometer.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
        )

    @property
    def inversion_plotter(self):
        return inversion_plotters.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.tracer_plotter.visuals_with_include_2d_of_plane(
                plane_index=1
            ),
            include_2d=self.include_2d,
        )

    @abstract_plotters.for_figure
    def figure_image(self):
        """Plot the model image of a specific plane of a lens fit.

        Set *autolens.datas.arrays.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
        plane_indexes : [int]
            The plane from which the model image is generated.
        """

        if self.fit.inversion is None:
            self.tracer_plotter.figure_image()
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

    @abstract_plotters.for_subplot
    def subplot_fit_real_space(self):

        number_subplots = 2

        self.open_subplot_figure(number_subplots=number_subplots)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        self.figure_image()

        if not self.tracer.planes[-1].has_pixelization:

            self.setup_subplot(number_subplots=number_subplots, subplot_index=2)

        else:

            aspect_inv = self.mat_plot_2d.figure.aspect_for_subplot_from_grid(
                grid=self.fit.inversion.mapper.source_full_grid
            )

            self.setup_subplot(
                number_subplots=number_subplots,
                subplot_index=2,
                aspect=float(aspect_inv),
            )

        self.figure_plane_image_of_plane(plane_index=-1)

        self.mat_plot_2d.output.subplot_to_figure()

        self.mat_plot_2d.figure.close()

from autoarray.plot.mat_wrap import mat_plot as mp
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

        visuals_2d = super(FitInterferometerPlotter, self).visuals_with_include_2d

        visuals_2d.mask = None

        return visuals_2d + visuals_2d.__class__(
            light_profile_centres=self.extract_2d(
                "light_profile_centres", self.tracer.planes[0].light_profile_centres
            ),
            mass_profile_centres=self.extract_2d(
                "mass_profile_centres", self.tracer.planes[0].mass_profile_centres
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

    def figures(
        self,
        visibilities=False,
        noise_map=False,
        signal_to_noise_map=False,
        model_visibilities=False,
        residual_map_real=False,
        residual_map_imag=False,
        normalized_residual_map_real=False,
        normalized_residual_map_imag=False,
        chi_squared_map_real=False,
        chi_squared_map_imag=False,
        image=False,
    ):

        super().figures(
            visibilities=visibilities,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_visibilities=model_visibilities,
            residual_map_real=residual_map_real,
            residual_map_imag=residual_map_imag,
            normalized_residual_map_real=normalized_residual_map_real,
            normalized_residual_map_imag=normalized_residual_map_imag,
            chi_squared_map_real=chi_squared_map_real,
            chi_squared_map_imag=chi_squared_map_imag,
        )

        if image:

            if self.fit.inversion is None:
                self.tracer_plotter.figures(image=True)
            else:
                self.inversion_plotter.figures(reconstructed_image=True)

    def figures_of_planes(self, plane_image, plane_index=None):

        if plane_image:

            if not self.tracer.planes[plane_index].has_pixelization:

                self.tracer_plotter.figures_of_planes(
                    plane_image=True, plane_index=plane_index
                )

            elif self.tracer.planes[plane_index].has_pixelization:

                self.inversion_plotter.figures(reconstruction=True)

    def subplot_fit_real_space(self):

        if self.fit.inversion is None:

            self.tracer_plotter.subplot(
                image=True, source_plane=True, auto_filename="subplot_fit_real_space"
            )

        elif self.fit.inversion is not None:

            self.inversion_plotter.subplot(
                reconstructed_image=True,
                reconstruction=True,
                auto_filename="subplot_fit_real_space",
            )

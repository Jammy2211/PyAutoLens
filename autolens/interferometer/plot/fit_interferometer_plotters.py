from typing import Optional

import autoarray as aa
import autogalaxy as ag
import autogalaxy.plot as aplt

from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.lens.ray_tracing import Tracer
from autolens.lens.plot.ray_tracing_plotters import TracerPlotter


class FitInterferometerPlotter(aplt.FitInterferometerPlotter):
    def __init__(
        self,
        fit: FitInterferometer,
        mat_plot_1d: aplt.MatPlot1D = aplt.MatPlot1D(),
        visuals_1d: aplt.Visuals1D = aplt.Visuals1D(),
        include_1d: aplt.Include1D = aplt.Include1D(),
        mat_plot_2d: aplt.MatPlot2D = aplt.MatPlot2D(),
        visuals_2d: aplt.Visuals2D = aplt.Visuals2D(),
        include_2d: aplt.Include2D = aplt.Include2D(),
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

        visuals_2d = super(FitInterferometerPlotter, self).visuals_with_include_2d

        visuals_2d.mask = None

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
    def tracer(self) -> Tracer:
        return self.fit.tracer

    @property
    def tracer_plotter(self) -> TracerPlotter:
        return TracerPlotter(
            tracer=self.tracer,
            grid=self.fit.interferometer.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
        )

    @property
    def inversion_plotter(self) -> aplt.InversionPlotter:
        return aplt.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.tracer_plotter.visuals_with_include_2d_of_plane(
                plane_index=1
            ),
            include_2d=self.include_2d,
        )

    def figures_2d(
        self,
        visibilities: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_visibilities: bool = False,
        residual_map_real: bool = False,
        residual_map_imag: bool = False,
        normalized_residual_map_real: bool = False,
        normalized_residual_map_imag: bool = False,
        chi_squared_map_real: bool = False,
        chi_squared_map_imag: bool = False,
        image: bool = False,
        dirty_image: bool = False,
        dirty_noise_map: bool = False,
        dirty_signal_to_noise_map: bool = False,
        dirty_model_image: bool = False,
        dirty_residual_map: bool = False,
        dirty_normalized_residual_map: bool = False,
        dirty_chi_squared_map: bool = False,
    ):

        super().figures_2d(
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
            dirty_image=dirty_image,
            dirty_noise_map=dirty_noise_map,
            dirty_signal_to_noise_map=dirty_signal_to_noise_map,
            dirty_model_image=dirty_model_image,
            dirty_residual_map=dirty_residual_map,
            dirty_normalized_residual_map=dirty_normalized_residual_map,
            dirty_chi_squared_map=dirty_chi_squared_map,
        )

        if image:

            if self.fit.inversion is None:
                self.tracer_plotter.figures_2d(image=True)
            else:
                self.inversion_plotter.figures_2d(reconstructed_image=True)

    def figures_2d_of_planes(
        self, plane_image: aa.Array2D, plane_index: Optional[int] = None
    ):

        if plane_image:

            if not self.tracer.planes[plane_index].has_pixelization:

                self.tracer_plotter.figures_2d_of_planes(
                    plane_image=True, plane_index=plane_index
                )

            elif self.tracer.planes[plane_index].has_pixelization:

                self.inversion_plotter.figures_2d_of_mapper(
                    mapper_index=0, reconstruction=True
                )

    def subplot_fit_real_space(self):

        if self.fit.inversion is None:

            self.tracer_plotter.subplot(
                image=True, source_plane=True, auto_filename="subplot_fit_real_space"
            )

        elif self.fit.inversion is not None:

            self.open_subplot_figure(number_subplots=2)

            self.inversion_plotter.figures_2d_of_mapper(
                mapper_index=0, reconstructed_image=True
            )
            self.inversion_plotter.figures_2d_of_mapper(
                mapper_index=0, reconstruction=True
            )

            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename="subplot_fit_real_space"
            )
            self.close_subplot_figure()

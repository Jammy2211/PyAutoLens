from typing import Optional

import autoarray as aa
import autogalaxy as ag
import autogalaxy.plot as aplt

from autoarray.fit.plot.fit_interferometer_plotters import FitInterferometerPlotterMeta

from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.lens.ray_tracing import Tracer
from autolens.lens.plot.ray_tracing_plotters import TracerPlotter
from autolens.plot.abstract_plotters import Plotter


class FitInterferometerPlotter(Plotter):
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
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.fit = fit

        self._fit_interferometer_meta_plotter = FitInterferometerPlotterMeta(
            fit=self.fit,
            get_visuals_2d_real_space=self.get_visuals_2d_real_space,
            mat_plot_1d=self.mat_plot_1d,
            include_1d=self.include_1d,
            visuals_1d=self.visuals_1d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

        self.subplot = self._fit_interferometer_meta_plotter.subplot
        self.subplot_fit_interferometer = (
            self._fit_interferometer_meta_plotter.subplot_fit_interferometer
        )
        self.subplot_fit_dirty_images = (
            self._fit_interferometer_meta_plotter.subplot_fit_dirty_images
        )

    def get_visuals_2d_real_space(self) -> aplt.Visuals2D:
        return self.get_2d.via_mask_from(mask=self.fit.interferometer.real_space_mask)

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
            visuals_2d=self.tracer_plotter.get_visuals_2d_of_plane(plane_index=1),
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

        self._fit_interferometer_meta_plotter.figures_2d(
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

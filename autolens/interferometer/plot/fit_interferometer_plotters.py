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
        """
        Plots the attributes of `FitInterferometer` objects using the matplotlib method `imshow()` and many
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2d` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `FitInterferometer` and plotted via the visuals object, if the corresponding entry is `True` in
        the `Include1D` or `Include2D` object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        fit
            The fit to an interferometer dataset the plotter plots.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `FitInterferometer` are extracted and plotted as visuals for 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `FitInterferometer` are extracted and plotted as visuals for 2D plots.
        """
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
        """
        Returns an `TracerPlotter` corresponding to the `Tracer` in the `FitImaging`.
        """
        return TracerPlotter(
            tracer=self.tracer,
            grid=self.fit.interferometer.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
        )

    def inversion_plotter_of_plane(self, plane_index: int) -> aplt.InversionPlotter:
        """
        Returns an `InversionPlotter` corresponding to one of the `Inversion`'s in the fit, which is specified via
        the index of the `Plane` that inversion was performed on.

        Parameters
        ----------
        plane_index
            The index of the inversion in the inversion which is used to create the `InversionPlotter`.

        Returns
        -------
        InversionPlotter
            An object that plots inversions which is used for plotting attributes of the inversion.
        """
        return aplt.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.tracer_plotter.get_visuals_2d_of_plane(
                plane_index=plane_index
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
        """
        Plots the individual attributes of the plotter's `FitInterferometer` object in 1D and 2D.

        The API is such that every plottable attribute of the `Interferometer` object is an input parameter of type
        bool of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        visibilities
            Whether or not to make a 2D plot (via `scatter`) of the visibility data.
        noise_map
            Whether or not to make a 2D plot (via `scatter`) of the noise-map.
        signal_to_noise_map
            Whether or not to make a 2D plot (via `scatter`) of the signal-to-noise-map.
        model_visibilities
            Whether or not to make a 2D plot (via `scatter`) of the model visibility data.
        residual_map_real
            Whether or not to make a 1D plot (via `plot`) of the real component of the residual map.
        residual_map_imag
            Whether or not to make a 1D plot (via `plot`) of the imaginary component of the residual map.
        normalized_residual_map_real
            Whether or not to make a 1D plot (via `plot`) of the real component of the normalized residual map.
        normalized_residual_map_imag
            Whether or not to make a 1D plot (via `plot`) of the imaginary component of the normalized residual map.
        chi_squared_map_real
            Whether or not to make a 1D plot (via `plot`) of the real component of the chi-squared map.
        chi_squared_map_imag
            Whether or not to make a 1D plot (via `plot`) of the imaginary component of the chi-squared map.
        image
            Whether or not to make a 2D plot (via `imshow`) of the source-plane image.
        dirty_image
            Whether or not to make a 2D plot (via `imshow`) of the dirty image.
        dirty_noise_map
            Whether or not to make a 2D plot (via `imshow`) of the dirty noise map.
        dirty_model_image
            Whether or not to make a 2D plot (via `imshow`) of the dirty model image.
        dirty_residual_map
            Whether or not to make a 2D plot (via `imshow`) of the dirty residual map.
        dirty_normalized_residual_map
            Whether or not to make a 2D plot (via `imshow`) of the dirty normalized residual map.
        dirty_chi_squared_map
            Whether or not to make a 2D plot (via `imshow`) of the dirty chi-squared map.
        """
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
                inversion_plotter = self.inversion_plotter_of_plane(plane_index=1)
                inversion_plotter.figures_2d(reconstructed_image=True)

    def figures_2d_of_planes(
        self, plane_index: Optional[int] = None, plane_image: bool = False
    ):
        """
        Plots images representing each individual `Plane` in the fit's `Tracer` in 2D, which are computed via the
        plotter's 2D grid object.

        These images subtract or omit the contribution of other planes in the plane, such that plots showing
        each individual plane are made.

        The API is such that every plottable attribute of the `Plane` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        plane_index
            The index of the plane which figures are plotted for.
        plane_image
            Whether or not to make a 2D plot (via `imshow`) of the image of a plane in its source-plane (e.g. unlensed).
            Depending on how the fit is performed, this could either be an image of light profiles of the reconstruction
            of an `Inversion`.
        """
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

    def subplot_fit_real_space(self):
        """
        Standard subplot of the real-space attributes of the plotter's `FitInterferometer` object.

        Depending on whether `LightProfile`'s or an `Inversion` are used to represent galaxies in the `Tracer`,
        different methods are called to create these real-space images.
        """
        if self.fit.inversion is None:

            self.tracer_plotter.subplot(
                image=True, source_plane=True, auto_filename="subplot_fit_real_space"
            )

        elif self.fit.inversion is not None:

            self.open_subplot_figure(number_subplots=2)

            inversion_plotter = self.inversion_plotter_of_plane(plane_index=1)

            inversion_plotter.figures_2d_of_mapper(
                mapper_index=0, reconstructed_image=True
            )
            inversion_plotter.figures_2d_of_mapper(mapper_index=0, reconstruction=True)

            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename="subplot_fit_real_space"
            )
            self.close_subplot_figure()

from typing import Optional

from autoconf import conf

import autoarray as aa
import autogalaxy.plot as aplt

from autoarray.fit.plot.fit_interferometer_plotters import FitInterferometerPlotterMeta
from autoarray.plot.auto_labels import AutoLabels

from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.lens.tracer import Tracer
from autolens.lens.plot.tracer_plotters import TracerPlotter
from autolens.plot.abstract_plotters import Plotter

from autolens.lens import tracer_util


class FitInterferometerPlotter(Plotter):
    def __init__(
        self,
        fit: FitInterferometer,
        mat_plot_1d: aplt.MatPlot1D = None,
        visuals_1d: aplt.Visuals1D = None,
        mat_plot_2d: aplt.MatPlot2D = None,
        visuals_2d: aplt.Visuals2D = None,
        residuals_symmetric_cmap: bool = True,
        visuals_2d_of_planes_list: Optional = None,
    ):
        """
        Plots the attributes of `FitInterferometer` objects using the matplotlib method `imshow()` and many
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2d` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `FitInterferometer` and plotted via the visuals object.

        Parameters
        ----------
        fit
            The fit to an interferometer dataset the plotter plots.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        residuals_symmetric_cmap
            If true, the `residual_map` and `normalized_residual_map` are plotted with a symmetric color map such
            that `abs(vmin) = abs(vmax)`.
        """
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
        )

        self.fit = fit

        self._fit_interferometer_meta_plotter = FitInterferometerPlotterMeta(
            fit=self.fit,
            mat_plot_1d=self.mat_plot_1d,
            visuals_1d=self.visuals_1d,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            residuals_symmetric_cmap=residuals_symmetric_cmap,
        )

        self.subplot = self._fit_interferometer_meta_plotter.subplot
        self.subplot_fit_dirty_images = (
            self._fit_interferometer_meta_plotter.subplot_fit_dirty_images
        )

        self._visuals_2d_of_planes_list = visuals_2d_of_planes_list

    @property
    def visuals_2d_of_planes_list(self):

        if self._visuals_2d_of_planes_list is None:
            self._visuals_2d_of_planes_list = (
                tracer_util.visuals_2d_of_planes_list_from(
                    tracer=self.fit.tracer,
                    grid=self.fit.grids.lp.mask.derive_grid.all_false,
                )
            )

        return self._visuals_2d_of_planes_list

    def visuals_2d_from(
        self, plane_index: Optional[int] = None, remove_critical_caustic: bool = False
    ) -> aplt.Visuals2D:
        """
        Returns the `Visuals2D` of the plotter with critical curves and caustics added, which are used to plot
        the critical curves and caustics of the `Tracer` object.

        If `remove_critical_caustic` is `True`, critical curves and caustics are not included in the visuals.

        Parameters
        ----------
        plane_index
            The index of the plane in the tracer which is used to extract quantities, as only one plane is plotted
            at a time.
        remove_critical_caustic
            Whether to remove critical curves and caustics from the visuals.
        """
        if remove_critical_caustic:
            return self.visuals_2d

        return self.visuals_2d + self.visuals_2d_of_planes_list[plane_index]

    @property
    def tracer(self) -> Tracer:
        return self.fit.tracer_linear_light_profiles_to_light_profiles

    def tracer_plotter_of_plane(
        self, plane_index: int, remove_critical_caustic: bool = False
    ) -> TracerPlotter:
        """
        Returns an `TracerPlotter` corresponding to the `Tracer` in the `FitImaging`.
        """

        zoom = aa.Zoom2D(mask=self.fit.dataset.real_space_mask)

        grid = aa.Grid2D.from_extent(
            extent=zoom.extent_from(buffer=0), shape_native=zoom.shape_native
        )
        return TracerPlotter(
            tracer=self.tracer,
            grid=grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d_from(
                plane_index=plane_index, remove_critical_caustic=remove_critical_caustic
            ),
        )

    def inversion_plotter_of_plane(
        self, plane_index: int, remove_critical_caustic: bool = False
    ) -> aplt.InversionPlotter:
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

        inversion_plotter = aplt.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d_from(
                plane_index=plane_index, remove_critical_caustic=remove_critical_caustic
            ),
        )
        return inversion_plotter

    def plane_indexes_from(self, plane_index: int):
        """
        Returns a list of all indexes of the planes in the fit, which is iterated over in figures that plot
        individual figures of each plane in a tracer.

        Parameters
        ----------
        plane_index
            A specific plane index which when input means that only a single plane index is returned.

        Returns
        -------
        list
            A list of galaxy indexes corresponding to planes in the plane.
        """
        if plane_index is None:
            return range(len(self.fit.tracer.planes))
        return [plane_index]

    def figures_2d(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        amplitudes_vs_uv_distances: bool = False,
        model_data: bool = False,
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
        data
            Whether to make a 2D plot (via `scatter`) of the visibility data.
        noise_map
            Whether to make a 2D plot (via `scatter`) of the noise-map.
        signal_to_noise_map
            Whether to make a 2D plot (via `scatter`) of the signal-to-noise-map.
        model_data
            Whether to make a 2D plot (via `scatter`) of the model visibility data.
        residual_map_real
            Whether to make a 1D plot (via `plot`) of the real component of the residual map.
        residual_map_imag
            Whether to make a 1D plot (via `plot`) of the imaginary component of the residual map.
        normalized_residual_map_real
            Whether to make a 1D plot (via `plot`) of the real component of the normalized residual map.
        normalized_residual_map_imag
            Whether to make a 1D plot (via `plot`) of the imaginary component of the normalized residual map.
        chi_squared_map_real
            Whether to make a 1D plot (via `plot`) of the real component of the chi-squared map.
        chi_squared_map_imag
            Whether to make a 1D plot (via `plot`) of the imaginary component of the chi-squared map.
        image
            Whether to make a 2D plot (via `imshow`) of the source-plane image.
        dirty_image
            Whether to make a 2D plot (via `imshow`) of the dirty image.
        dirty_noise_map
            Whether to make a 2D plot (via `imshow`) of the dirty noise map.
        dirty_model_image
            Whether to make a 2D plot (via `imshow`) of the dirty model image.
        dirty_residual_map
            Whether to make a 2D plot (via `imshow`) of the dirty residual map.
        dirty_normalized_residual_map
            Whether to make a 2D plot (via `imshow`) of the dirty normalized residual map.
        dirty_chi_squared_map
            Whether to make a 2D plot (via `imshow`) of the dirty chi-squared map.
        """
        self._fit_interferometer_meta_plotter.figures_2d(
            data=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            amplitudes_vs_uv_distances=amplitudes_vs_uv_distances,
            model_data=model_data,
            residual_map_real=residual_map_real,
            residual_map_imag=residual_map_imag,
            normalized_residual_map_real=normalized_residual_map_real,
            normalized_residual_map_imag=normalized_residual_map_imag,
            chi_squared_map_real=chi_squared_map_real,
            chi_squared_map_imag=chi_squared_map_imag,
            dirty_image=dirty_image,
            dirty_noise_map=dirty_noise_map,
            dirty_signal_to_noise_map=dirty_signal_to_noise_map,
            dirty_residual_map=dirty_residual_map,
            dirty_normalized_residual_map=dirty_normalized_residual_map,
            dirty_chi_squared_map=dirty_chi_squared_map,
        )

        if image:
            plane_index = len(self.tracer.planes) - 1

            if not self.tracer.planes[plane_index].has(cls=aa.Pixelization):

                tracer_plotter = self.tracer_plotter_of_plane(plane_index=plane_index)

                tracer_plotter.figures_2d(image=True)

            elif self.tracer.planes[plane_index].has(cls=aa.Pixelization):
                inversion_plotter = self.inversion_plotter_of_plane(
                    plane_index=plane_index
                )
                inversion_plotter.figures_2d(reconstructed_image=True)

        if dirty_model_image:
            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_model_image,
                visuals_2d=self.visuals_2d_of_planes_list[0],
                auto_labels=AutoLabels(
                    title="Dirty Model Image", filename="dirty_model_image_2d"
                ),
            )

    def figures_2d_of_planes(
        self,
        plane_index: Optional[int] = None,
        plane_image: bool = False,
        plane_noise_map: bool = False,
        plane_signal_to_noise_map: bool = False,
        zoom_to_brightest: bool = True,
        interpolate_to_uniform: bool = False,
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
            Whether to make a 2D plot (via `imshow`) of the image of a plane in its source-plane (e.g. unlensed).
            Depending on how the fit is performed, this could either be an image of light profiles of the reconstruction
            of an `Inversion`.
        plane_noise_map
            Whether to make a 2D plot of the noise-map of a plane in its source-plane, where the
            noise map can only be computed when a pixelized source reconstruction is performed and they correspond to
            the noise map in each reconstructed pixel as given by the inverse curvature matrix.
        plane_signal_to_noise_map
            Whether to make a 2D plot of the signal-to-noise map of a plane in its source-plane,
            where the signal-to-noise map values can only be computed when a pixelized source reconstruction and they
            are the ratio of reconstructed flux to error in each pixel.
        zoom_to_brightest
            For images not in the image-plane (e.g. the `plane_image`), whether to automatically zoom the plot to
            the brightest regions of the galaxies being plotted as opposed to the full extent of the grid.
        interpolate_to_uniform
            If `True`, the mapper's reconstruction is interpolated to a uniform grid before plotting, for example
            meaning that an irregular Delaunay grid can be plotted as a uniform grid.
        """
        if plane_image:
            if not self.tracer.planes[plane_index].has(cls=aa.Pixelization):

                tracer_plotter = self.tracer_plotter_of_plane(plane_index=plane_index)

                tracer_plotter.figures_2d_of_planes(
                    plane_image=True,
                    plane_index=plane_index,
                    zoom_to_brightest=zoom_to_brightest,
                )

            elif self.tracer.planes[plane_index].has(cls=aa.Pixelization):
                inversion_plotter = self.inversion_plotter_of_plane(plane_index=1)
                inversion_plotter.figures_2d_of_pixelization(
                    pixelization_index=0,
                    reconstruction=True,
                    zoom_to_brightest=zoom_to_brightest,
                    interpolate_to_uniform=interpolate_to_uniform,
                )

        if plane_noise_map:
            if self.tracer.planes[plane_index].has(cls=aa.Pixelization):
                inversion_plotter = self.inversion_plotter_of_plane(
                    plane_index=plane_index
                )

                inversion_plotter.figures_2d_of_pixelization(
                    pixelization_index=0,
                    reconstruction_noise_map=True,
                    zoom_to_brightest=zoom_to_brightest,
                    interpolate_to_uniform=interpolate_to_uniform,
                )

        if plane_signal_to_noise_map:
            if self.tracer.planes[plane_index].has(cls=aa.Pixelization):
                inversion_plotter = self.inversion_plotter_of_plane(
                    plane_index=plane_index
                )

                inversion_plotter.figures_2d_of_pixelization(
                    pixelization_index=0,
                    signal_to_noise_map=True,
                    zoom_to_brightest=zoom_to_brightest,
                    interpolate_to_uniform=interpolate_to_uniform,
                )

    def subplot_fit(self):
        """
        Standard subplot of the attributes of the plotter's `FitImaging` object.
        """

        self.open_subplot_figure(number_subplots=12)

        self.figures_2d(amplitudes_vs_uv_distances=True)

        self.mat_plot_1d.subplot_index = 2
        self.mat_plot_2d.subplot_index = 2

        self.figures_2d(dirty_image=True)
        self.figures_2d(dirty_signal_to_noise_map=True)
        self.figures_2d(dirty_model_image=True)
        self.figures_2d(image=True)

        self.mat_plot_1d.subplot_index = 6
        self.mat_plot_2d.subplot_index = 6

        self.figures_2d(normalized_residual_map_real=True)
        self.figures_2d(normalized_residual_map_imag=True)

        self.mat_plot_1d.subplot_index = 8
        self.mat_plot_2d.subplot_index = 8

        final_plane_index = len(self.fit.tracer.planes) - 1

        self.set_title(label="Source Plane (Zoomed)")
        self.figures_2d_of_planes(plane_index=final_plane_index, plane_image=True)
        self.set_title(label=None)

        self.figures_2d(dirty_normalized_residual_map=True)

        self.mat_plot_2d.cmap.kwargs["vmin"] = -1.0
        self.mat_plot_2d.cmap.kwargs["vmax"] = 1.0

        self.set_title(label=r"Normalized Residual Map $1\sigma$")
        self.figures_2d(dirty_normalized_residual_map=True)
        self.set_title(label=None)

        self.mat_plot_2d.cmap.kwargs.pop("vmin")
        self.mat_plot_2d.cmap.kwargs.pop("vmax")

        self.figures_2d(dirty_chi_squared_map=True)

        self.set_title(label="Source Plane (No Zoom)")
        self.figures_2d_of_planes(
            plane_index=final_plane_index,
            plane_image=True,
            zoom_to_brightest=False,
        )

        self.set_title(label=None)

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_fit")
        self.close_subplot_figure()

    def subplot_mappings_of_plane(
        self, plane_index: Optional[int] = None, auto_filename: str = "subplot_mappings"
    ):
        if self.fit.inversion is None:
            return

        plane_indexes = self.plane_indexes_from(plane_index=plane_index)

        for plane_index in plane_indexes:
            pixelization_index = 0

            inversion_plotter = self.inversion_plotter_of_plane(plane_index=0)

            inversion_plotter.open_subplot_figure(number_subplots=4)

            self.figures_2d(dirty_image=True)

            total_pixels = conf.instance["visualize"]["general"]["inversion"][
                "total_mappings_pixels"
            ]

            mapper = inversion_plotter.inversion.cls_list_from(cls=aa.AbstractMapper)[0]
            mapper_valued = aa.MapperValued(
                values=inversion_plotter.inversion.reconstruction_dict[mapper],
                mapper=mapper,
            )
            pix_indexes = mapper_valued.max_pixel_list_from(
                total_pixels=total_pixels, filter_neighbors=True
            )

            inversion_plotter.visuals_2d.source_plane_mesh_indexes = [
                [index] for index in pix_indexes[pixelization_index]
            ]

            inversion_plotter.visuals_2d.tangential_critical_curves = None
            inversion_plotter.visuals_2d.radial_critical_curves = None

            inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=pixelization_index, reconstructed_image=True
            )

            self.visuals_2d.source_plane_mesh_indexes = [
                [index] for index in pix_indexes[pixelization_index]
            ]

            self.figures_2d_of_planes(
                plane_index=plane_index,
                plane_image=True,
            )

            self.set_title(label="Source Reconstruction (Unzoomed)")
            self.figures_2d_of_planes(
                plane_index=plane_index,
                plane_image=True,
                zoom_to_brightest=False,
            )
            self.set_title(label=None)

            self.visuals_2d.source_plane_mesh_indexes = None

            inversion_plotter.mat_plot_2d.output.subplot_to_figure(
                auto_filename=f"{auto_filename}_{pixelization_index}"
            )

            inversion_plotter.close_subplot_figure()

    def subplot_fit_real_space(self):
        """
        Standard subplot of the real-space attributes of the plotter's `FitInterferometer` object.

        Depending on whether `LightProfile`'s or an `Inversion` are used to represent galaxies in the `Tracer`,
        different methods are called to create these real-space images.
        """
        if self.fit.inversion is None:

            tracer_plotter = self.tracer_plotter_of_plane(plane_index=0)

            tracer_plotter.subplot(
                image=True, source_plane=True, auto_filename="subplot_fit_real_space"
            )

        elif self.fit.inversion is not None:
            self.open_subplot_figure(number_subplots=2)

            inversion_plotter = self.inversion_plotter_of_plane(plane_index=1)

            inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=0, reconstructed_image=True
            )
            inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=0, reconstruction=True
            )

            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename="subplot_fit_real_space"
            )
            self.close_subplot_figure()

import copy
import numpy as np
from typing import Optional

from autoconf import conf

import autoarray as aa
import autogalaxy.plot as aplt

from autoarray.plot.auto_labels import AutoLabels
from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta

from autolens.plot.abstract_plotters import Plotter
from autolens.imaging.fit_imaging import FitImaging
from autolens.lens.plot.tracer_plotters import TracerPlotter

from autolens.lens import tracer_util


class FitImagingPlotter(Plotter):
    def __init__(
        self,
        fit: FitImaging,
        mat_plot_2d: aplt.MatPlot2D = None,
        visuals_2d: aplt.Visuals2D = None,
        residuals_symmetric_cmap: bool = True,
        visuals_2d_of_planes_list : Optional = None
    ):
        """
        Plots the attributes of `FitImaging` objects using the matplotlib method `imshow()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `FitImaging` and plotted via the visuals object.

        Parameters
        ----------
        fit
            The fit to an imaging dataset the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make the plot.
        visuals_2d
            Contains visuals that can be overlaid on the plot.
        residuals_symmetric_cmap
            If true, the `residual_map` and `normalized_residual_map` are plotted with a symmetric color map such
            that `abs(vmin) = abs(vmax)`.
        """
        super().__init__(mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d)

        self.fit = fit

        self._fit_imaging_meta_plotter = FitImagingPlotterMeta(
            fit=self.fit,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            residuals_symmetric_cmap=residuals_symmetric_cmap,
        )

        self.residuals_symmetric_cmap = residuals_symmetric_cmap

        self._visuals_2d_of_planes_list = visuals_2d_of_planes_list

    @property
    def visuals_2d_of_planes_list(self):

        if self._visuals_2d_of_planes_list is None:

            self._visuals_2d_of_planes_list = tracer_util.visuals_2d_of_planes_list_from(
                tracer=self.fit.tracer,
                grid=self.fit.grids.lp.mask.derive_grid.all_false,
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

        return (
            self.visuals_2d
            + self.visuals_2d_of_planes_list[plane_index]
        )

    @property
    def tracer(self):
        return self.fit.tracer_linear_light_profiles_to_light_profiles

    def tracer_plotter_of_plane(
        self, plane_index: int, remove_critical_caustic: bool = False
    ) -> TracerPlotter:
        """
        Returns an `TracerPlotter` corresponding to the `Tracer` in the `FitImaging`.
        """

        zoom = aa.Zoom2D(mask=self.fit.mask)

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

    def figures_2d_of_planes(
        self,
        plane_index: Optional[int] = None,
        subtracted_image: bool = False,
        model_image: bool = False,
        plane_image: bool = False,
        plane_noise_map: bool = False,
        plane_signal_to_noise_map: bool = False,
        use_source_vmax: bool = False,
        zoom_to_brightest: bool = True,
        interpolate_to_uniform: bool = False,
        remove_critical_caustic: bool = False,
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
        subtracted_image
            Whether to make a 2D plot (via `imshow`) of the subtracted image of a plane, where this image is
            the fit's `data` minus the model images of all other planes, thereby showing an individual plane in the
            data.
        model_image
            Whether to make a 2D plot (via `imshow`) of the model image of a plane, where this image is the
            model image of one plane, thereby showing how much it contributes to the overall model image.
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
        use_source_vmax
            If `True`, the maximum value of the lensed source (e.g. in the image-plane) is used to set the `vmax` of
            certain plots (e.g. the `data`) in order to ensure the lensed source is visible compared to the lens.
        zoom_to_brightest
            For images not in the image-plane (e.g. the `plane_image`), whether to automatically zoom the plot to
            the brightest regions of the galaxies being plotted as opposed to the full extent of the grid.
        interpolate_to_uniform
            If `True`, the mapper's reconstruction is interpolated to a uniform grid before plotting, for example
            meaning that an irregular Delaunay grid can be plotted as a uniform grid.
        remove_critical_caustic
            Whether to remove critical curves and caustics from the plot.
        """

        plane_indexes = self.plane_indexes_from(plane_index=plane_index)

        for plane_index in plane_indexes:

            if use_source_vmax:
                self.mat_plot_2d.cmap.kwargs["vmax"] = np.max(
                    self.fit.model_images_of_planes_list[plane_index].array
                )

            if subtracted_image:

                title = f"Subtracted Image of Plane {plane_index}"
                filename = f"subtracted_image_of_plane_{plane_index}"

                if len(self.tracer.planes) == 2:

                    if plane_index == 0:

                        title = "Source Subtracted Image"
                        filename = "source_subtracted_image"

                    elif plane_index == 1:

                        title = "Lens Subtracted Image"
                        filename = "lens_subtracted_image"

                self.mat_plot_2d.plot_array(
                    array=self.fit.subtracted_images_of_planes_list[plane_index],
                    visuals_2d=self.visuals_2d_from(
                        plane_index=plane_index,
                        remove_critical_caustic=remove_critical_caustic,
                    ),
                    auto_labels=aplt.AutoLabels(title=title, filename=filename),
                )

            if model_image:

                title = f"Model Image of Plane {plane_index}"
                filename = f"model_image_of_plane_{plane_index}"

                if len(self.tracer.planes) == 2:

                    if plane_index == 0:

                        title = "Lens Model Image"
                        filename = "lens_model_image"

                    elif plane_index == 1:

                        title = "Source Model Image"
                        filename = "source_model_image"

                self.mat_plot_2d.plot_array(
                    array=self.fit.model_images_of_planes_list[plane_index],
                    visuals_2d=self.visuals_2d_from(
                        plane_index=plane_index,
                        remove_critical_caustic=remove_critical_caustic,
                    ),
                    auto_labels=aplt.AutoLabels(title=title, filename=filename),
                )

            if plane_image:

                if not self.tracer.planes[plane_index].has(cls=aa.Pixelization):

                    tracer_plotter = self.tracer_plotter_of_plane(
                        plane_index=plane_index,
                        remove_critical_caustic=remove_critical_caustic,
                    )

                    tracer_plotter.figures_2d_of_planes(
                        plane_image=True,
                        plane_index=plane_index,
                        zoom_to_brightest=zoom_to_brightest,
                        retain_visuals=True,
                    )

                elif self.tracer.planes[plane_index].has(cls=aa.Pixelization):

                    inversion_plotter = self.inversion_plotter_of_plane(
                        plane_index=plane_index,
                        remove_critical_caustic=remove_critical_caustic,
                    )

                    inversion_plotter.figures_2d_of_pixelization(
                        pixelization_index=0,
                        reconstruction=True,
                        zoom_to_brightest=zoom_to_brightest,
                        interpolate_to_uniform=interpolate_to_uniform,
                    )

            if use_source_vmax:
                try:
                    self.mat_plot_2d.cmap.kwargs.pop("vmax")
                except KeyError:
                    pass

            if plane_noise_map:

                if self.tracer.planes[plane_index].has(cls=aa.Pixelization):

                    inversion_plotter = self.inversion_plotter_of_plane(
                        plane_index=plane_index,
                        remove_critical_caustic=remove_critical_caustic,
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
                        plane_index=plane_index,
                        remove_critical_caustic=remove_critical_caustic,
                    )

                    inversion_plotter.figures_2d_of_pixelization(
                        pixelization_index=0,
                        signal_to_noise_map=True,
                        zoom_to_brightest=zoom_to_brightest,
                        interpolate_to_uniform=interpolate_to_uniform,
                    )

    def subplot(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_data: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        auto_filename: str = "subplot_fit",
    ):
        """
        Plots the individual attributes of the plotter's `FitImaging` object in 2D on a subplot.

        The API is such that every plottable attribute of the `FitImaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        data
            Whether to include a 2D plot (via `imshow`) of the image data.
        noise_map
            Whether to include a 2D plot (via `imshow`) of the noise map.
        psf
            Whether to include a 2D plot (via `imshow`) of the psf.
        signal_to_noise_map
            Whether to include a 2D plot (via `imshow`) of the signal-to-noise map.
        model_data
            Whether to include a 2D plot (via `imshow`) of the model image.
        residual_map
            Whether to include a 2D plot (via `imshow`) of the residual map.
        normalized_residual_map
            Whether to include a 2D plot (via `imshow`) of the normalized residual map.
        chi_squared_map
            Whether to include a 2D plot (via `imshow`) of the chi-squared map.
        auto_filename
            The default filename of the output subplot if written to hard-disk.
        """
        self._subplot_custom_plot(
            data=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_image=model_data,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
            auto_labels=AutoLabels(filename=auto_filename),
        )


    def subplot_fit_x1_plane(self):
        """
        Standard subplot of the attributes of the plotter's `FitImaging` object.
        """

        self.open_subplot_figure(number_subplots=6)

        self.mat_plot_2d.cmap.kwargs["vmax"] = np.max(self.fit.model_images_of_planes_list[0].array)
        self.figures_2d(data=True)
        self.mat_plot_2d.cmap.kwargs.pop("vmax")

        self.figures_2d(signal_to_noise_map=True)

        self.mat_plot_2d.cmap.kwargs["vmax"] = np.max(self.fit.model_images_of_planes_list[0].array)
        self.figures_2d(model_image=True)
        self.mat_plot_2d.cmap.kwargs.pop("vmax")

        self.residuals_symmetric_cmap = False
        self.set_title(label="Lens Light Subtracted")
        self.figures_2d(normalized_residual_map=True)

        self.mat_plot_2d.cmap.kwargs["vmin"] = 0.0
        self.set_title(label="Subtracted Image Zero Minimum")
        self.figures_2d(normalized_residual_map=True)
        self.mat_plot_2d.cmap.kwargs.pop("vmin")

        self.residuals_symmetric_cmap = True
        self.set_title(label="Normalized Residual Map")
        self.figures_2d(normalized_residual_map=True)
        self.set_title(label=None)

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_fit_x1_plane")
        self.close_subplot_figure()

    def subplot_fit_log10_x1_plane(self):
        """
        Standard subplot of the attributes of the plotter's `FitImaging` object.
        """

        contour_original = copy.copy(self.mat_plot_2d.contour)
        use_log10_original = self.mat_plot_2d.use_log10

        self.open_subplot_figure(number_subplots=6)

        self.mat_plot_2d.contour = False
        self.mat_plot_2d.use_log10 = True

        self.mat_plot_2d.cmap.kwargs["vmax"] = np.max(self.fit.model_images_of_planes_list[0].array)
        self.figures_2d(data=True)
        self.mat_plot_2d.cmap.kwargs.pop("vmax")

        self.figures_2d(signal_to_noise_map=True)

        self.mat_plot_2d.cmap.kwargs["vmax"] = np.max(self.fit.model_images_of_planes_list[0].array)
        self.figures_2d(model_image=True)
        self.mat_plot_2d.cmap.kwargs.pop("vmax")

        self.residuals_symmetric_cmap = False
        self.set_title(label="Lens Light Subtracted")
        self.figures_2d(normalized_residual_map=True)

        self.residuals_symmetric_cmap = True
        self.set_title(label="Normalized Residual Map")
        self.figures_2d(normalized_residual_map=True)
        self.set_title(label=None)

        self.figures_2d(chi_squared_map=True)

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_fit_log10")
        self.close_subplot_figure()

        self.mat_plot_2d.use_log10 = use_log10_original
        self.mat_plot_2d.contour = contour_original

    def subplot_fit(self, plane_index: Optional[int] = None):
        """
        Standard subplot of the attributes of the plotter's `FitImaging` object.
        """

        if len(self.fit.tracer.planes) == 1:
            return self.subplot_fit_x1_plane()

        self.open_subplot_figure(number_subplots=12)

        self.figures_2d(data=True)

        self.set_title(label="Data (Source Scale)")
        self.figures_2d(data=True, use_source_vmax=True)
        self.set_title(label=None)

        self.figures_2d(signal_to_noise_map=True)
        self.figures_2d(model_image=True)

        self.set_title(label="Lens Light Model Image")
        self.figures_2d_of_planes(
            plane_index=0, model_image=True, remove_critical_caustic=True
        )

        # If the lens light is not included the subplot index does not increase, so we must manually set it to 4
        self.mat_plot_2d.subplot_index = 6

        plane_index_tag = "" if plane_index is None else f"_{plane_index}"

        plane_index = (
            len(self.fit.tracer.planes) - 1 if plane_index is None else plane_index
        )

        self.mat_plot_2d.cmap.kwargs["vmin"] = 0.0

        self.set_title(label="Lens Light Subtracted")
        self.figures_2d_of_planes(
            plane_index=plane_index,
            subtracted_image=True,
            use_source_vmax=True,
            remove_critical_caustic=True,
        )

        self.set_title(label="Source Model Image")
        self.figures_2d_of_planes(
            plane_index=plane_index,
            model_image=True,
            use_source_vmax=True,
            remove_critical_caustic=True,
        )

        self.mat_plot_2d.cmap.kwargs.pop("vmin")

        self.set_title(label="Source Plane (Zoomed)")
        self.figures_2d_of_planes(
            plane_index=plane_index, plane_image=True, use_source_vmax=True
        )

        self.set_title(label=None)

        self.mat_plot_2d.subplot_index = 9

        self.figures_2d(normalized_residual_map=True)

        self.mat_plot_2d.cmap.kwargs["vmin"] = -1.0
        self.mat_plot_2d.cmap.kwargs["vmax"] = 1.0

        self.set_title(label=r"Normalized Residual Map $1\sigma$")
        self.figures_2d(normalized_residual_map=True)
        self.set_title(label=None)

        self.mat_plot_2d.cmap.kwargs.pop("vmin")
        self.mat_plot_2d.cmap.kwargs.pop("vmax")

        self.figures_2d(chi_squared_map=True)

        self.set_title(label="Source Plane (No Zoom)")
        self.figures_2d_of_planes(
            plane_index=plane_index,
            plane_image=True,
            zoom_to_brightest=False,
            use_source_vmax=True,
        )

        self.set_title(label=None)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"subplot_fit{plane_index_tag}",
           # also_show=self.mat_plot_2d.quick_update
        )
        self.close_subplot_figure()

    def subplot_fit_log10(self, plane_index: Optional[int] = None):
        """
        Standard subplot of the attributes of the plotter's `FitImaging` object.
        """

        if len(self.fit.tracer.planes) == 1:
            return self.subplot_fit_log10_x1_plane()

        contour_original = copy.copy(self.mat_plot_2d.contour)
        use_log10_original = self.mat_plot_2d.use_log10

        self.open_subplot_figure(number_subplots=12)

        self.mat_plot_2d.contour = False
        self.mat_plot_2d.use_log10 = True

        self.figures_2d(data=True)

        self.set_title(label="Data (Source Scale)")

        try:
            self.figures_2d(data=True, use_source_vmax=True)
        except ValueError:
            pass

        self.set_title(label=None)

        try:
            self.figures_2d(signal_to_noise_map=True)
        except ValueError:
            pass

        self.figures_2d(model_image=True)

        self.set_title(label="Lens Light Model Image")
        self.figures_2d_of_planes(plane_index=0, model_image=True, remove_critical_caustic=True)

        # If the lens light is not included the subplot index does not increase, so we must manually set it to 4
        self.mat_plot_2d.subplot_index = 6

        plane_index_tag = "" if plane_index is None else f"_{plane_index}"

        plane_index = (
            len(self.fit.tracer.planes) - 1 if plane_index is None else plane_index
        )

        self.mat_plot_2d.cmap.kwargs["vmin"] = 0.0

        self.set_title(label="Lens Light Subtracted")
        self.figures_2d_of_planes(
            plane_index=plane_index, subtracted_image=True, use_source_vmax=True, remove_critical_caustic=True
        )

        self.set_title(label="Source Model Image")
        self.figures_2d_of_planes(
            plane_index=plane_index, model_image=True, use_source_vmax=True, remove_critical_caustic=True
        )

        self.mat_plot_2d.cmap.kwargs.pop("vmin")

        self.set_title(label="Source Plane (Zoomed)")
        self.figures_2d_of_planes(
            plane_index=plane_index, plane_image=True, use_source_vmax=True
        )

        self.set_title(label=None)

        self.mat_plot_2d.use_log10 = False

        self.mat_plot_2d.subplot_index = 9

        self.figures_2d(normalized_residual_map=True)

        self.mat_plot_2d.cmap.kwargs["vmin"] = -1.0
        self.mat_plot_2d.cmap.kwargs["vmax"] = 1.0

        self.set_title(label=r"Normalized Residual Map $1\sigma$")
        self.figures_2d(normalized_residual_map=True)
        self.set_title(label=None)

        self.mat_plot_2d.cmap.kwargs.pop("vmin")
        self.mat_plot_2d.cmap.kwargs.pop("vmax")

        self.mat_plot_2d.use_log10 = True

        self.figures_2d(chi_squared_map=True)

        self.set_title(label="Source Plane (No Zoom)")
        self.figures_2d_of_planes(
            plane_index=plane_index,
            plane_image=True,
            zoom_to_brightest=False,
            use_source_vmax=True,
        )

        self.set_title(label=None)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"subplot_fit_log10{plane_index_tag}"
        )
        self.close_subplot_figure()

        self.mat_plot_2d.use_log10 = use_log10_original
        self.mat_plot_2d.contour = contour_original

    def subplot_of_planes(self, plane_index: Optional[int] = None):
        """
        Plots images representing each individual `Plane` in the plotter's `Tracer` in 2D on a subplot, which are
        computed via the plotter's 2D grid object.

        These images subtract or omit the contribution of other planes in the plane, such that plots showing
        each individual plane are made.

        The subplot plots the subtracted image, model image and plane image of each plane, where are described in the
        `figures_2d_of_planes` function.

        Parameters
        ----------
        plane_index
            The index of the plane whose images are included on the subplot.
        """

        plane_indexes = self.plane_indexes_from(plane_index=plane_index)

        for plane_index in plane_indexes:

            self.open_subplot_figure(number_subplots=4)

            self.figures_2d(data=True)

            self.figures_2d_of_planes(subtracted_image=True, plane_index=plane_index)
            self.figures_2d_of_planes(model_image=True, plane_index=plane_index)
            self.figures_2d_of_planes(plane_image=True, plane_index=plane_index)

            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename=f"subplot_of_plane_{plane_index}"
            )
            self.close_subplot_figure()

    def subplot_tracer(self):
        """
        Standard subplot of a Tracer.

        The `subplot_tracer` method in the `Tracer` class cannot plot the images of galaxies which are computed
        via an `Inversion`. Therefore, using the `subplot_tracer` method of the `FitImagingPLotter` can plot
        more information.

        Returns
        -------

        """

        use_log10_original = self.mat_plot_2d.use_log10

        final_plane_index = len(self.fit.tracer.planes) - 1

        self.open_subplot_figure(number_subplots=9)

        self.figures_2d(model_image=True)

        self.set_title(label="Lensed Source Image")
        self.figures_2d_of_planes(
            plane_index=final_plane_index, model_image=True, use_source_vmax=True
        )
        self.set_title(label=None)

        self.set_title(label="Source Plane")
        self.figures_2d_of_planes(
            plane_index=final_plane_index,
            plane_image=True,
            zoom_to_brightest=False,
            use_source_vmax=True,
        )

        tracer_plotter = self.tracer_plotter_of_plane(plane_index=0)

        tracer_plotter._subplot_lens_and_mass()

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_tracer")
        self.close_subplot_figure()

        self.mat_plot_2d.use_log10 = use_log10_original

    def subplot_mappings_of_plane(
        self, plane_index: Optional[int] = None, auto_filename: str = "subplot_mappings"
    ):

        try:

            plane_indexes = self.plane_indexes_from(plane_index=plane_index)

            for plane_index in plane_indexes:

                pixelization_index = 0

                inversion_plotter = self.inversion_plotter_of_plane(plane_index=0)

                inversion_plotter.open_subplot_figure(number_subplots=4)

                inversion_plotter.figures_2d_of_pixelization(
                    pixelization_index=pixelization_index, data_subtracted=True
                )

                total_pixels = conf.instance["visualize"]["general"]["inversion"][
                    "total_mappings_pixels"
                ]

                mapper = inversion_plotter.inversion.cls_list_from(
                    cls=aa.AbstractMapper
                )[0]
                mapper_valued = aa.MapperValued(
                    values=inversion_plotter.inversion.reconstruction_dict[mapper],
                    mapper=mapper,
                )

                pix_indexes = mapper_valued.max_pixel_list_from(
                    total_pixels=total_pixels, filter_neighbors=True
                )

                indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

                inversion_plotter.visuals_2d.indexes = indexes

                inversion_plotter.figures_2d_of_pixelization(
                    pixelization_index=pixelization_index, reconstructed_image=True
                )

                self.visuals_2d.source_plane_mesh_indexes = [
                    [index] for index in pix_indexes[pixelization_index]
                ]

                self.figures_2d_of_planes(
                    plane_index=plane_index, plane_image=True, use_source_vmax=True
                )

                self.set_title(label="Source Reconstruction (Unzoomed)")
                self.figures_2d_of_planes(
                    plane_index=plane_index,
                    plane_image=True,
                    zoom_to_brightest=False,
                    use_source_vmax=True,
                )
                self.set_title(label=None)

                self.visuals_2d.source_plane_mesh_indexes = None

                inversion_plotter.mat_plot_2d.output.subplot_to_figure(
                    auto_filename=f"{auto_filename}_{pixelization_index}"
                )

                inversion_plotter.close_subplot_figure()

        except (IndexError, AttributeError, ValueError):

            pass

    def figures_2d(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        residual_flux_fraction_map: bool = False,
        use_source_vmax: bool = False,
        suffix: str = "",
    ):
        """
        Plots the individual attributes of the plotter's `FitImaging` object in 2D.

        The API is such that every plottable attribute of the `FitImaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        data
            Whether to make a 2D plot (via `imshow`) of the image data.
        noise_map
            Whether to make a 2D plot (via `imshow`) of the noise map.
        signal_to_noise_map
            Whether to make a 2D plot (via `imshow`) of the signal-to-noise map.
        model_image
            Whether to make a 2D plot (via `imshow`) of the model image.
        residual_map
            Whether to make a 2D plot (via `imshow`) of the residual map.
        normalized_residual_map
            Whether to make a 2D plot (via `imshow`) of the normalized residual map.
        chi_squared_map
            Whether to make a 2D plot (via `imshow`) of the chi-squared map.
        residual_flux_fraction_map
            Whether to make a 2D plot (via `imshow`) of the residual flux fraction map.
        use_source_vmax
            If `True`, the maximum value of the lensed source (e.g. in the image-plane) is used to set the `vmax` of
            certain plots (e.g. the `data`) in order to ensure the lensed source is visible compared to the lens.
        """

        if use_source_vmax:
            try:
                source_vmax = np.max(
                    [
                        model_image.array
                        for model_image in self.fit.model_images_of_planes_list[1:]
                    ]
                )
            except ValueError:
                source_vmax = None

        if data:

            if use_source_vmax:
                self.mat_plot_2d.cmap.kwargs["vmax"] = source_vmax

            self.mat_plot_2d.plot_array(
                array=self.fit.data,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(title="Data", filename=f"data{suffix}"),
            )

            if use_source_vmax:
                self.mat_plot_2d.cmap.kwargs.pop("vmax")

        if noise_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.noise_map,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(
                    title="Noise-Map", filename=f"noise_map{suffix}"
                ),
            )

        if signal_to_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.signal_to_noise_map,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(
                    title="Signal-To-Noise Map",
                    cb_unit=" S/N",
                    filename=f"signal_to_noise_map{suffix}",
                ),
            )

        if model_image:

            if use_source_vmax:
                self.mat_plot_2d.cmap.kwargs["vmax"] = source_vmax

            self.mat_plot_2d.plot_array(
                array=self.fit.model_data,
                visuals_2d=self.visuals_2d_from(plane_index=0),
                auto_labels=AutoLabels(
                    title="Model Image", filename=f"model_image{suffix}"
                ),
            )

            if use_source_vmax:
                self.mat_plot_2d.cmap.kwargs.pop("vmax")

        cmap_original = self.mat_plot_2d.cmap

        if self.residuals_symmetric_cmap:

            self.mat_plot_2d.cmap = self.mat_plot_2d.cmap.symmetric_cmap_from()

        if residual_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.residual_map,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(
                    title="Residual Map", filename=f"residual_map{suffix}"
                ),
            )

        if normalized_residual_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.normalized_residual_map,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(
                    title="Normalized Residual Map",
                    cb_unit=r" $\sigma$",
                    filename=f"normalized_residual_map{suffix}",
                ),
            )

        self.mat_plot_2d.cmap = cmap_original

        if chi_squared_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.chi_squared_map,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(
                    title="Chi-Squared Map",
                    cb_unit=r" $\chi^2$",
                    filename=f"chi_squared_map{suffix}",
                ),
            )

        if residual_flux_fraction_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.residual_flux_fraction_map,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(
                    title="Residual Flux Fraction Map",
                    filename=f"residual_flux_fraction_map{suffix}",
                ),
            )

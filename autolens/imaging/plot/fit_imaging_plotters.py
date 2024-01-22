import copy
import numpy as np
from typing import Optional

import autoarray as aa
import autogalaxy.plot as aplt

from autoarray.plot.auto_labels import AutoLabels
from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta

from autolens.plot.abstract_plotters import Plotter
from autolens.imaging.fit_imaging import FitImaging
from autolens.lens.plot.ray_tracing_plotters import TracerPlotter


class FitImagingPlotter(Plotter):
    def __init__(
        self,
        fit: FitImaging,
        mat_plot_2d: aplt.MatPlot2D = aplt.MatPlot2D(),
        visuals_2d: aplt.Visuals2D = aplt.Visuals2D(),
        include_2d: aplt.Include2D = aplt.Include2D(),
        residuals_symmetric_cmap: bool = True
    ):
        """
        Plots the attributes of `FitImaging` objects using the matplotlib method `imshow()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `FitImaging` and plotted via the visuals object, if the corresponding entry is `True` in the `Include2D`
        object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        fit
            The fit to an imaging dataset the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make the plot.
        visuals_2d
            Contains visuals that can be overlaid on the plot.
        include_2d
            Specifies which attributes of the `Array2D` are extracted and plotted as visuals.
        residuals_symmetric_cmap
            If true, the `residual_map` and `normalized_residual_map` are plotted with a symmetric color map such
            that `abs(vmin) = abs(vmax)`.
        """
        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.fit = fit

        self._fit_imaging_meta_plotter = FitImagingPlotterMeta(
            fit=self.fit,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
            residuals_symmetric_cmap=residuals_symmetric_cmap
        )

        self.residuals_symmetric_cmap = residuals_symmetric_cmap

    def get_visuals_2d(self) -> aplt.Visuals2D:
        return self.get_2d.via_fit_imaging_from(fit=self.fit)

    @property
    def tracer(self):
        return self.fit.tracer_linear_light_profiles_to_light_profiles

    @property
    def tracer_plotter(self) -> TracerPlotter:
        """
        Returns an `TracerPlotter` corresponding to the `Tracer` in the `FitImaging`.
        """

        extent = self.fit.data.extent_of_zoomed_array(buffer=0)
        shape_native = self.fit.data.zoomed_around_mask(buffer=0).shape_native

        grid = aa.Grid2D.from_extent(
            extent=extent,
            shape_native=shape_native
        )

        return TracerPlotter(
            tracer=self.tracer,
            grid=grid,
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
        inversion_plotter = aplt.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.tracer_plotter.get_visuals_2d_of_plane(
                plane_index=plane_index
            ),
            include_2d=self.include_2d,
        )
        inversion_plotter.visuals_2d.mask = None
        inversion_plotter.visuals_2d.border = None
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
        use_source_vmax: bool = False,
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
        use_source_vmax
            If `True`, the maximum value of the lensed source (e.g. in the image-plane) is used to set the `vmax` of
            certain plots (e.g. the `data`) in order to ensure the lensed source is visible compared to the lens.
        zoom_to_brightest
            For images not in the image-plane (e.g. the `plane_image`), whether to automatically zoom the plot to
            the brightest regions of the galaxies being plotted as opposed to the full extent of the grid.
        interpolate_to_uniform
            If `True`, the mapper's reconstruction is interpolated to a uniform grid before plotting, for example
            meaning that an irregular Delaunay grid can be plotted as a uniform grid.
        """

        visuals_2d = self.get_visuals_2d()

        visuals_2d_no_critical_caustic = self.get_visuals_2d()
        visuals_2d_no_critical_caustic.tangential_critical_curves = None
        visuals_2d_no_critical_caustic.radial_critical_curves = None
        visuals_2d_no_critical_caustic.tangential_caustics = None
        visuals_2d_no_critical_caustic.radial_caustics = None

        plane_indexes = self.plane_indexes_from(plane_index=plane_index)

        if use_source_vmax:
            self.mat_plot_2d.cmap.kwargs["vmax"] = np.max(self.fit.model_images_of_planes_list[-1])

        for plane_index in plane_indexes:

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
                    visuals_2d=visuals_2d_no_critical_caustic,
                    auto_labels=aplt.AutoLabels(
                        title=title,
                        filename=filename
                    ),
                )

            if model_image:

                if self.tracer.planes[plane_index].has(cls=aa.Pixelization):

                    # Overwrite plane_index=0 so that model image uses critical curves -- improve via config cutomization

                    visuals_2d_model_image = self.inversion_plotter_of_plane(plane_index=0).get_visuals_2d_for_data()

                else:

                    visuals_2d_model_image = visuals_2d

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
                    visuals_2d=visuals_2d_model_image,
                    auto_labels=aplt.AutoLabels(
                        title=title,
                        filename=filename
                    ),
                )

            if plane_image:

                if not self.tracer.planes[plane_index].has(cls=aa.Pixelization):

                    self.tracer_plotter.figures_2d_of_planes(
                        plane_image=True,
                        plane_index=plane_index,
                        zoom_to_brightest=zoom_to_brightest
                    )

                elif self.tracer.planes[plane_index].has(cls=aa.Pixelization):

                    inversion_plotter = self.inversion_plotter_of_plane(
                        plane_index=plane_index
                    )

                    inversion_plotter.figures_2d_of_pixelization(
                        pixelization_index=0,
                        reconstruction=True,
                        zoom_to_brightest=zoom_to_brightest,
                        interpolate_to_uniform=interpolate_to_uniform
                    )

        if use_source_vmax:
            self.mat_plot_2d.cmap.kwargs.pop("vmax")

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

    def subplot_fit(self):
        """
        Standard subplot of the attributes of the plotter's `FitImaging` object.
        """

        self.open_subplot_figure(number_subplots=12)

        self.figures_2d(data=True)

        self.set_title(label="Data (Source Scale)")
        self.figures_2d(data=True, use_source_vmax=True)
        self.set_title(label=None)

        self.figures_2d(signal_to_noise_map=True)
        self.figures_2d(model_image=True)

        self.set_title(label="Lens Light Model Image")
        self.figures_2d_of_planes(plane_index=0, model_image=True)

        # If the lens light is not included the subplot index does not increase, so we must manually set it to 4
        self.mat_plot_2d.subplot_index = 6

        final_plane_index = len(self.fit.tracer.planes) - 1

        self.mat_plot_2d.cmap.kwargs["vmin"] = 0.0

        self.set_title(label="Lens Light Subtracted Image")
        self.figures_2d_of_planes(plane_index=final_plane_index, subtracted_image=True, use_source_vmax=True)

        self.set_title(label="Source Model Image (Image Plane)")
        self.figures_2d_of_planes(plane_index=final_plane_index, model_image=True, use_source_vmax=True)

        self.mat_plot_2d.cmap.kwargs.pop("vmin")

        self.set_title(label="Source Plane (Zoomed)")
        self.figures_2d_of_planes(plane_index=final_plane_index, plane_image=True, use_source_vmax=True)


        self.set_title(label=None)

        self.mat_plot_2d.subplot_index = 9

        self.figures_2d(normalized_residual_map=True)

        self.mat_plot_2d.cmap.kwargs["vmin"] = -1.0
        self.mat_plot_2d.cmap.kwargs["vmax"] = 1.0

        self.set_title(label="Normalized Residual Map (1 sigma)")
        self.figures_2d(normalized_residual_map=True)
        self.set_title(label=None)

        self.mat_plot_2d.cmap.kwargs.pop("vmin")
        self.mat_plot_2d.cmap.kwargs.pop("vmax")

        self.figures_2d(chi_squared_map=True)

        self.set_title(label="Source Plane (No Zoom)")
        self.figures_2d_of_planes(
            plane_index=final_plane_index,
            plane_image=True,
            zoom_to_brightest=False,
            use_source_vmax=True
        )

        self.set_title(label=None)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_fit"
        )
        self.close_subplot_figure()

    def subplot_fit_log10(self):
        """
        Standard subplot of the attributes of the plotter's `FitImaging` object.
        """

        contour_original = copy.copy(self.mat_plot_2d.contour)
        use_log10_original = self.mat_plot_2d.use_log10

        self.open_subplot_figure(number_subplots=12)

        self.mat_plot_2d.contour = False
        self.mat_plot_2d.use_log10 = True

        self.figures_2d(data=True)

        self.set_title(label="Data (Source Scale)")

        self.figures_2d(data=True, use_source_vmax=True)
        self.set_title(label=None)

        self.figures_2d(signal_to_noise_map=True)
        self.figures_2d(model_image=True)

        self.set_title(label="Lens Light Model Image")
        self.figures_2d_of_planes(plane_index=0, model_image=True)

        # If the lens light is not included the subplot index does not increase, so we must manually set it to 4
        self.mat_plot_2d.subplot_index = 6

        final_plane_index = len(self.fit.tracer.planes) - 1

        self.mat_plot_2d.cmap.kwargs["vmin"] = 0.0

        self.set_title(label="Lens Light Subtracted Image")
        self.figures_2d_of_planes(plane_index=final_plane_index, subtracted_image=True, use_source_vmax=True)

        self.set_title(label="Source Model Image (Image Plane)")
        self.figures_2d_of_planes(plane_index=final_plane_index, model_image=True, use_source_vmax=True)

        self.mat_plot_2d.cmap.kwargs.pop("vmin")

        self.set_title(label="Source Plane (Zoomed)")
        self.figures_2d_of_planes(plane_index=final_plane_index, plane_image=True, use_source_vmax=True)


        self.set_title(label=None)

        self.mat_plot_2d.use_log10 = False

        self.mat_plot_2d.subplot_index = 9

        self.figures_2d(normalized_residual_map=True)

        self.mat_plot_2d.cmap.kwargs["vmin"] = -1.0
        self.mat_plot_2d.cmap.kwargs["vmax"] = 1.0

        self.set_title(label="Normalized Residual Map (1 sigma)")
        self.figures_2d(normalized_residual_map=True)
        self.set_title(label=None)

        self.mat_plot_2d.cmap.kwargs.pop("vmin")
        self.mat_plot_2d.cmap.kwargs.pop("vmax")

        self.mat_plot_2d.use_log10 = True

        self.figures_2d(chi_squared_map=True)

        self.set_title(label="Source Plane (No Zoom)")
        self.figures_2d_of_planes(
            plane_index=final_plane_index,
            plane_image=True,
            zoom_to_brightest=False,
            use_source_vmax=True
        )

        self.set_title(label=None)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_fit_log10"
        )
        self.close_subplot_figure()

        self.mat_plot_2d.use_log10 = use_log10_original
        self.mat_plot_2d.contour = contour_original

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
        self.figures_2d_of_planes(plane_index=final_plane_index, model_image=True, use_source_vmax=True)
        self.set_title(label=None)

        self.set_title(label="Source Plane")
        self.figures_2d_of_planes(
            plane_index=final_plane_index,
            plane_image=True,
            zoom_to_brightest=False,
            use_source_vmax=True
        )

        tracer_plotter = self.tracer_plotter

        include_tangential_critical_curves_original = tracer_plotter.include_2d._tangential_critical_curves
        include_radial_critical_curves_original = tracer_plotter.include_2d._radial_critical_curves

        tracer_plotter._subplot_lens_and_mass()

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_tracer"
        )
        self.close_subplot_figure()

        self.include_2d._tangential_critical_curves = include_tangential_critical_curves_original
        self.include_2d._radial_critical_curves = include_radial_critical_curves_original
        self.mat_plot_2d.use_log10 = use_log10_original

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
        use_source_vmax : bool = False,
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

        visuals_2d = self.get_visuals_2d()

        visuals_2d_no_critical_caustic = self.get_visuals_2d()
        visuals_2d_no_critical_caustic.tangential_critical_curves = None
        visuals_2d_no_critical_caustic.radial_critical_curves = None
        visuals_2d_no_critical_caustic.tangential_caustics = None
        visuals_2d_no_critical_caustic.radial_caustics = None
        visuals_2d_no_critical_caustic.origin = None
        visuals_2d_no_critical_caustic.light_profile_centres = None
        visuals_2d_no_critical_caustic.mass_profile_centres = None

        if data:

            if use_source_vmax:
                self.mat_plot_2d.cmap.kwargs["vmax"] = np.max(self.fit.model_images_of_planes_list[-1])

            self.mat_plot_2d.plot_array(
                array=self.fit.data,
                visuals_2d=visuals_2d_no_critical_caustic,
                auto_labels=AutoLabels(title="Data", filename=f"data{suffix}"),
            )

            if use_source_vmax:
                self.mat_plot_2d.cmap.kwargs.pop("vmax")

        if noise_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.noise_map,
                visuals_2d=visuals_2d_no_critical_caustic,
                auto_labels=AutoLabels(
                    title="Noise-Map", filename=f"noise_map{suffix}"
                ),
            )

        if signal_to_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.signal_to_noise_map,
                visuals_2d=visuals_2d_no_critical_caustic,
                auto_labels=AutoLabels(
                    title="Signal-To-Noise Map", cb_unit=" S/N", filename=f"signal_to_noise_map{suffix}"
                ),
            )

        if model_image:

            if use_source_vmax:
                self.mat_plot_2d.cmap.kwargs["vmax"] = np.max(self.fit.model_images_of_planes_list[-1])

            self.mat_plot_2d.plot_array(
                array=self.fit.model_data,
                visuals_2d=visuals_2d,
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
                visuals_2d=visuals_2d_no_critical_caustic,
                auto_labels=AutoLabels(
                    title="Residual Map", filename=f"residual_map{suffix}"
                ),
            )

        if normalized_residual_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.normalized_residual_map,
                visuals_2d=visuals_2d_no_critical_caustic,
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
                visuals_2d=visuals_2d_no_critical_caustic,
                auto_labels=AutoLabels(
                    title="Chi-Squared Map", cb_unit=r" $\chi^2$",  filename=f"chi_squared_map{suffix}"
                ),
            )

        if residual_flux_fraction_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.residual_flux_fraction_map,
                visuals_2d=visuals_2d_no_critical_caustic,
                auto_labels=AutoLabels(
                    title="Residual Flux Fraction Map", filename=f"residual_flux_fraction_map{suffix}"
                ),
            )
import numpy as np
from typing import Optional

import autogalaxy.plot as aplt

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
        )

        self.figures_2d = self._fit_imaging_meta_plotter.figures_2d
        self.subplot = self._fit_imaging_meta_plotter.subplot
        self.subplot_fit_imaging = self._fit_imaging_meta_plotter.subplot_fit_imaging

    def get_visuals_2d(self) -> aplt.Visuals2D:
        return self.get_2d.via_fit_imaging_from(fit=self.fit)

    @property
    def tracer(self):
        return self.fit.tracer

    @property
    def tracer_plotter(self) -> TracerPlotter:
        """
        Returns an `TracerPlotter` corresponding to the `Tracer` in the `FitImaging`.
        """
        return TracerPlotter(
            tracer=self.tracer,
            grid=self.fit.grid,
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
            Whether or not to make a 2D plot (via `imshow`) of the subtracted image of a plane, where this image is
            the fit's `data` minus the model images of all other planes, thereby showing an individual plane in the
            data.
        model_image
            Whether or not to make a 2D plot (via `imshow`) of the model image of a plane, where this image is the
            model image of one plane, thereby showing how much it contributes to the overall model image.
        plane_image
            Whether or not to make a 2D plot (via `imshow`) of the image of a plane in its source-plane (e.g. unlensed).
            Depending on how the fit is performed, this could either be an image of light profiles of the reconstruction
            of an `Inversion`.
        """

        plane_indexes = self.plane_indexes_from(plane_index=plane_index)

        for plane_index in plane_indexes:

            if subtracted_image:

                if "vmin" in self.mat_plot_2d.cmap.kwargs:
                    vmin_stored = True
                else:
                    self.mat_plot_2d.cmap.kwargs["vmin"] = 0.0
                    vmin_stored = False

                if "vmax" in self.mat_plot_2d.cmap.kwargs:
                    vmax_stored = True
                else:
                    self.mat_plot_2d.cmap.kwargs["vmax"] = np.max(
                        self.fit.model_images_of_planes_list[plane_index]
                    )
                    vmax_stored = False

                self.mat_plot_2d.plot_array(
                    array=self.fit.subtracted_images_of_planes_list[plane_index],
                    visuals_2d=self.get_visuals_2d(),
                    auto_labels=aplt.AutoLabels(
                        title=f"Subtracted Image of Plane {plane_index}",
                        filename=f"subtracted_image_of_plane_{plane_index}",
                    ),
                )

                if not vmin_stored:
                    self.mat_plot_2d.cmap.kwargs.pop("vmin")

                if not vmax_stored:
                    self.mat_plot_2d.cmap.kwargs.pop("vmax")

            if model_image:

                if self.fit.inversion is None or plane_index == 0:

                    self.mat_plot_2d.plot_array(
                        array=self.fit.model_images_of_planes_list[plane_index],
                        visuals_2d=self.get_visuals_2d(),
                        auto_labels=aplt.AutoLabels(
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
                    inversion_plotter.figures_2d_of_mapper(
                        mapper_index=0, reconstruction=True
                    )

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

            self.figures_2d(image=True)
            self.figures_2d_of_planes(subtracted_image=True, plane_index=plane_index)
            self.figures_2d_of_planes(model_image=True, plane_index=plane_index)
            self.figures_2d_of_planes(plane_image=True, plane_index=plane_index)

            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename=f"subplot_of_plane_{plane_index}"
            )
            self.close_subplot_figure()

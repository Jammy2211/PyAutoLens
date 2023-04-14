import autogalaxy.plot as aplt

from autolens.lens.plot.ray_tracing_plotters import TracerPlotter as TracerPlotterBase


class TracerPlotter(TracerPlotterBase):

    def figures_2d(
        self,
        image: bool = False,
        source_plane: bool = False,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
        contribution_map: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `Tracer` object in 2D, which are computed via the plotter's 2D
        grid object.

        The API is such that every plottable attribute of the `Tracer` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 2D plot (via `imshow`) of the image of tracer in its image-plane (e.g. after
            lensing).
        source_plane
            Whether to make a 2D plot (via `imshow`) of the image of the tracer in the source-plane (e.g. its
            unlensed light).
        convergence
            Whether to make a 2D plot (via `imshow`) of the convergence.
        potential
            Whether to make a 2D plot (via `imshow`) of the potential.
        deflections_y
            Whether to make a 2D plot (via `imshow`) of the y component of the deflection angles.
        deflections_x
            Whether to make a 2D plot (via `imshow`) of the x component of the deflection angles.
        magnification
            Whether to make a 2D plot (via `imshow`) of the magnification.
        contribution_map
            Whether to make a 2D plot (via `imshow`) of the contribution map.
        """

        if image:

            self.mat_plot_2d.plot_array(
                array=self.tracer.image_2d_from(grid=self.grid),
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(title="Image", filename="image_2d"),
            )

        if source_plane:
            self.figures_2d_of_planes(
                plane_image=True, plane_index=len(self.tracer.planes) - 1
            )

        self._mass_plotter.figures_2d(
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
        )

        if contribution_map:

            self.mat_plot_2d.plot_array(
                array=self.tracer.contribution_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title="Contribution Map", filename="contribution_map_2d"
                ),
            )

    def subplot(
        self,
        image: bool = False,
        source_plane: bool = False,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
        contribution_map: bool = False,
        auto_filename: str = "subplot_tracer",
    ):
        """
        Plots the individual attributes of the plotter's `Tracer` object in 2D on a subplot, which are computed via
        the plotter's 2D grid object.

        The API is such that every plottable attribute of the `Tracer` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        image
            Whether or not to include a 2D plot (via `imshow`) of the image of tracer in its image-plane (e.g. after
            lensing).
        source_plane
            Whether or not to include a 2D plot (via `imshow`) of the image of the tracer in the source-plane (e.g. its
            unlensed light).
        convergence
            Whether or not to include a 2D plot (via `imshow`) of the convergence.
        potential
            Whether or not to include a 2D plot (via `imshow`) of the potential.
        deflections_y
            Whether or not to include a 2D plot (via `imshow`) of the y component of the deflection angles.
        deflections_x
            Whether or not to include a 2D plot (via `imshow`) of the x component of the deflection angles.
        magnification
            Whether or not to include a 2D plot (via `imshow`) of the magnification.
        contribution_map
            Whether or not to include a 2D plot (via `imshow`) of the contribution map.
        auto_filename
            The default filename of the output subplot if written to hard-disk.
        """

        self._subplot_custom_plot(
            image=image,
            source_plane=source_plane,
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
            contribution_map=contribution_map,
            auto_labels=aplt.AutoLabels(filename=auto_filename),
        )

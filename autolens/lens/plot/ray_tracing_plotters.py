from typing import Optional, List

import autoarray as aa
import autogalaxy as ag
import autogalaxy.plot as aplt

from autogalaxy.plot.mass_plotter import MassPlotter

from autolens.plot.abstract_plotters import Plotter
from autolens.lens.ray_tracing import Tracer


class TracerPlotter(Plotter):
    def __init__(
        self,
        tracer: Tracer,
        grid: aa.type.Grid2DLike,
        mat_plot_1d: aplt.MatPlot1D = aplt.MatPlot1D(),
        visuals_1d: aplt.Visuals1D = aplt.Visuals1D(),
        include_1d: aplt.Include1D = aplt.Include1D(),
        mat_plot_2d: aplt.MatPlot2D = aplt.MatPlot2D(),
        visuals_2d: aplt.Visuals2D = aplt.Visuals2D(),
        include_2d: aplt.Include2D = aplt.Include2D(),
    ):
        """
        Plots the attributes of `Tracer` objects using the matplotlib methods `plot()` and `imshow()` and many 
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default, 
        the settings passed to every matplotlib function called are those specified in 
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to 
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be 
        extracted from the `MassProfile` and plotted via the visuals object, if the corresponding entry is `True` in 
        the `Include1D` or `Include2D` object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        tracer
            The tracer the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the tracer's light and mass quantities that are plotted.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `MassProfile` are extracted and plotted as visuals for 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `MassProfile` are extracted and plotted as visuals for 2D plots.
        """
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.tracer = tracer
        self.grid = grid

        self._mass_plotter = MassPlotter(
            mass_obj=self.tracer,
            grid=self.grid,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

    def get_visuals_2d(self) -> aplt.Visuals2D:
        return self.get_visuals_2d_of_plane(plane_index=0)

    def get_visuals_2d_of_plane(self, plane_index: int) -> aplt.Visuals2D:
        return self.get_2d.via_tracer_from(
            tracer=self.tracer, grid=self.grid, plane_index=plane_index
        )

    def plane_plotter_from(self, plane_index: int) -> aplt.PlanePlotter:
        """
        Returns an `PlanePlotter` corresponding to a `Plane` in the `Tracer`.

        Returns
        -------
        plane_index
            The index of the plane in the `Tracer` used to make the `PlanePlotter`.
        """
        plane_grid = self.tracer.traced_grid_2d_list_from(grid=self.grid)[plane_index]

        return aplt.PlanePlotter(
            plane=self.tracer.planes[plane_index],
            grid=plane_grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.get_visuals_2d_of_plane(plane_index=plane_index),
            include_2d=self.include_2d,
        )

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
            Whether or not to make a 2D plot (via `imshow`) of the image of tracer in its image-plane (e.g. after
            lensing).
        source_plane
            Whether or not to make a 2D plot (via `imshow`) of the image of the tracer in the source-plane (e.g. its
            unlensed light).
        convergence
            Whether or not to make a 2D plot (via `imshow`) of the convergence.
        potential
            Whether or not to make a 2D plot (via `imshow`) of the potential.
        deflections_y
            Whether or not to make a 2D plot (via `imshow`) of the y component of the deflection angles.
        deflections_x
            Whether or not to make a 2D plot (via `imshow`) of the x component of the deflection angles.
        magnification
            Whether or not to make a 2D plot (via `imshow`) of the magnification.
        contribution_map
            Whether or not to make a 2D plot (via `imshow`) of the contribution map.
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

    def plane_indexes_from(self, plane_index: Optional[int]) -> List[int]:
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
            return list(range(len(self.tracer.planes)))
        return [plane_index]

    def figures_2d_of_planes(
        self,
        plane_image: bool = False,
        plane_grid: bool = False,
        plane_index: Optional[int] = None,
    ):
        """
        Plots source-plane images (e.g. the unlensed light) each individual `Plane` in the plotter's `Tracer` in 2D, 
        which are computed via the plotter's 2D grid object.

        The API is such that every plottable attribute of the `Plane` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        plane_image
            Whether or not to make a 2D plot (via `imshow`) of the image of the plane in the soure-plane (e.g. its
            unlensed light).
        plane_grid
            Whether or not to make a 2D plot (via `scatter`) of the lensed (y,x) coordinates of the plane in the 
            source-plane.
        plane_index
            If input, plots for only a single plane based on its index in the tracer are created.
        """
        plane_indexes = self.plane_indexes_from(plane_index=plane_index)

        for plane_index in plane_indexes:

            plane_plotter = self.plane_plotter_from(plane_index=plane_index)

            if plane_image:

                plane_plotter.figures_2d(
                    plane_image=True,
                    title_suffix=f" Of Plane {plane_index}",
                    filename_suffix=f"_of_plane_{plane_index}",
                )

            if plane_grid:

                plane_plotter.figures_2d(
                    plane_grid=True,
                    title_suffix=f" Of Plane {plane_index}",
                    filename_suffix=f"_of_plane_{plane_index}",
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

    def subplot_tracer(self):
        """
        Standard subplot of the attributes of the plotter's `Tracer` object.
        """
        return self.subplot(
            image=True,
            source_plane=True,
            convergence=True,
            potential=True,
            deflections_y=True,
            deflections_x=True,
        )

    def subplot_plane_images(self):
        """
        Subplot of the image of every plane in its own plane.

        For example, for a 2 plane `Tracer`, this creates a subplot with 2 panels, one for the image-plane image
        and one for the source-plane (e.g. unlensed) image. If there are 3 planes, 3 panels are created, showing
        images at each plane.
        """
        number_subplots = 2 * self.tracer.total_planes - 1

        self.open_subplot_figure(number_subplots=number_subplots)

        plane_plotter = self.plane_plotter_from(plane_index=0)
        plane_plotter.figures_2d(image=True, title_suffix=" Of Plane 0")

        self.mat_plot_2d.subplot_index += 1

        for plane_index in range(1, self.tracer.total_planes):

            plane_plotter = self.plane_plotter_from(plane_index=plane_index)
            plane_plotter.figures_2d(
                image=True, title_suffix=f" Of Plane {plane_index}"
            )
            plane_plotter.figures_2d(
                plane_image=True, title_suffix=f" Of Plane {plane_index}"
            )

        self.mat_plot_2d.output.subplot_to_figure(auto_filename=f"subplot_plane_images")
        self.close_subplot_figure()

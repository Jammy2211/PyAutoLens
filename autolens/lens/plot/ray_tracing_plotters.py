from typing import Optional, List

import autoarray as aa
import autogalaxy as ag
import autogalaxy.plot as aplt

from autogalaxy.plot.lensing_obj_plotter import LensingObjPlotter

from autolens.lens.ray_tracing import Tracer


class TracerPlotter(LensingObjPlotter):
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

    @property
    def lensing_obj(self) -> Tracer:
        return self.tracer

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

        return self.visuals_with_include_2d_of_plane(plane_index=0)

    def visuals_with_include_2d_of_plane(self, plane_index) -> aplt.Visuals2D:
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

        border = self.extract_2d(
            "border", value=self.grid.mask.border_grid_sub_1.binned
        )

        if border is not None:
            if plane_index > 0:
                border = self.tracer.traced_grids_of_planes_from(grid=border)[
                    plane_index
                ]

        if plane_index == 0:
            critical_curves = self.extract_2d(
                "critical_curves",
                self.tracer.critical_curves_from(grid=self.grid),
                "critical_curves",
            )
        else:
            critical_curves = None

        if plane_index == 1:
            caustics = self.extract_2d(
                "caustics", self.tracer.caustics_from(grid=self.grid), "caustics"
            )
        else:
            caustics = None

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin", value=aa.Grid2DIrregular(grid=[self.grid.origin])
            ),
            border=border,
            light_profile_centres=self.extract_2d(
                "light_profile_centres",
                self.tracer.planes[plane_index].extract_attribute(
                    cls=ag.lp.LightProfile, attr_name="centre"
                ),
            ),
            mass_profile_centres=self.extract_2d(
                "mass_profile_centres",
                self.tracer.planes[plane_index].extract_attribute(
                    cls=ag.mp.MassProfile, attr_name="centre"
                ),
            ),
            critical_curves=critical_curves,
            caustics=caustics,
        )

    def plane_plotter_from(self, plane_index: int) -> aplt.PlanePlotter:

        plane_grid = self.tracer.traced_grids_of_planes_from(grid=self.grid)[
            plane_index
        ]

        return aplt.PlanePlotter(
            plane=self.tracer.planes[plane_index],
            grid=plane_grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_with_include_2d_of_plane(plane_index=plane_index),
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
        """Plot the observed _tracer of an analysis, using the `Imaging` class object.
    
        The visualization and output type can be fully customized.
    
        Parameters
        -----------
        tracer : autolens.imaging.tracer.Imaging
            Class containing the _tracer, noise_mappers and PSF that are to be plotted.
            The font size of the figure ylabel.
        output_path : str
            The path where the _tracer is output if the output_type is a file format (e.g. png, fits)
        output_format : str
            How the _tracer is output. File formats (e.g. png, fits) output the _tracer to harddisk. 'show' displays the _tracer \
            in the python interpreter window.
        """

        if image:

            self.mat_plot_2d.plot_array(
                array=self.tracer.image_2d_from(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=aplt.AutoLabels(title="Image", filename="image_2d"),
            )

        if source_plane:
            self.figures_2d_of_planes(
                plane_image=True, plane_index=len(self.tracer.planes) - 1
            )

        super().figures_2d(
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
        )

        if contribution_map:

            self.mat_plot_2d.plot_array(
                array=self.tracer.contribution_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=aplt.AutoLabels(
                    title="Contribution Map", filename="contribution_map_2d"
                ),
            )

    def plane_indexes_from(self, plane_index: int) -> List[int]:

        if plane_index is None:
            return list(range(len(self.tracer.planes)))
        else:
            return [plane_index]

    def figures_2d_of_planes(
        self,
        plane_image: bool = False,
        plane_grid: bool = False,
        plane_index: Optional[int] = None,
    ):

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
        """Plot the observed _tracer of an analysis, using the `Imaging` class object.

        The visualization and output type can be fully customized.

        Parameters
        -----------
        tracer : autolens.imaging.tracer.Imaging
            Class containing the _tracer,  noise_mappers and PSF that are to be plotted.
            The font size of the figure ylabel.
        output_path : str
            The path where the _tracer is output if the output_type is a file format (e.g. png, fits)
        output_format : str
            How the _tracer is output. File formats (e.g. png, fits) output the _tracer to harddisk. 'show' displays the _tracer \
            in the python interpreter window.
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
        return self.subplot(
            image=True,
            source_plane=True,
            convergence=True,
            potential=True,
            deflections_y=True,
            deflections_x=True,
        )

    def subplot_plane_images(self):

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

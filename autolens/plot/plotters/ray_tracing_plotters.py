from autoarray.structures import arrays, grids
from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import mat_plot as mp
from autogalaxy.plot.plotters import lensing_obj_plotter
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autogalaxy.plot.plotters import plane_plotters
from autolens.lens import ray_tracing


class TracerPlotter(lensing_obj_plotter.LensingObjPlotter):
    def __init__(
        self,
        tracer: ray_tracing.Tracer,
        grid: grids.Grid2D,
        mat_plot_1d: lensing_mat_plot.MatPlot1D = lensing_mat_plot.MatPlot1D(),
        visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
        include_1d: lensing_include.Include1D = lensing_include.Include1D(),
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):
        super().__init__(
            lensing_obj=tracer,
            grid=grid,
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

    @property
    def visuals_with_include_2d(self) -> lensing_visuals.Visuals2D:
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

    def visuals_with_include_2d_of_plane(
        self, plane_index
    ) -> lensing_visuals.Visuals2D:
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
            "border", value=self.grid.mask.border_grid_sub_1.slim_binned
        )

        if border is not None:
            if plane_index > 0:
                border = self.tracer.traced_grids_of_planes_from_grid(grid=border)[
                    plane_index
                ]

        if plane_index == 0:
            critical_curves = self.extract_2d(
                "critical_curves",
                self.lensing_obj.critical_curves_from_grid(grid=self.grid),
                "critical_curves",
            )
        else:
            critical_curves = None

        if plane_index == 1:
            caustics = self.extract_2d(
                "caustics",
                self.lensing_obj.caustics_from_grid(grid=self.grid),
                "caustics",
            )
        else:
            caustics = None

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin", value=grids.Grid2DIrregular(grid=[self.grid.origin])
            ),
            border=border,
            light_profile_centres=self.extract_2d(
                "light_profile_centres",
                self.tracer.planes[plane_index].light_profile_centres,
            ),
            mass_profile_centres=self.extract_2d(
                "mass_profile_centres",
                self.tracer.planes[plane_index].mass_profile_centres,
            ),
            critical_curves=critical_curves,
            caustics=caustics,
        )

    @property
    def tracer(self):
        return self.lensing_obj

    def plane_plotter_from(self, plane_index):

        plane_grid = self.tracer.traced_grids_of_planes_from_grid(grid=self.grid)[
            plane_index
        ]

        return plane_plotters.PlanePlotter(
            plane=self.tracer.planes[plane_index],
            grid=plane_grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_with_include_2d_of_plane(plane_index=plane_index),
            include_2d=self.include_2d,
        )

    def figures(
        self,
        image=False,
        source_plane=False,
        convergence=False,
        potential=False,
        deflections_y=False,
        deflections_x=False,
        magnification=False,
        contribution_map=False,
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
                array=self.tracer.image_from_grid(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(title="Image", filename="image"),
            )

        if source_plane:
            self.figures_of_planes(
                plane_image=True, plane_index=len(self.tracer.planes) - 1
            )

        super().figures(
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
                auto_labels=mp.AutoLabels(
                    title="Contribution Map", filename="contribution_map"
                ),
            )

    def plane_indexes_from_plane_index(self, plane_index):

        if plane_index is None:
            return range(len(self.tracer.planes))
        else:
            return [plane_index]

    def figures_of_planes(self, plane_image=False, plane_grid=False, plane_index=None):

        plane_indexes = self.plane_indexes_from_plane_index(plane_index=plane_index)

        for plane_index in plane_indexes:

            plane_plotter = self.plane_plotter_from(plane_index=plane_index)

            if plane_image:

                plane_plotter.figures(
                    plane_image=True,
                    title_suffix=f" Of Plane {plane_index}",
                    filename_suffix=f"_of_plane_{plane_index}",
                )

            if plane_grid:

                plane_plotter.figures(
                    plane_grid=True,
                    title_suffix=f" Of Plane {plane_index}",
                    filename_suffix=f"_of_plane_{plane_index}",
                )

    def subplot(
        self,
        image=False,
        source_plane=False,
        convergence=False,
        potential=False,
        deflections_y=False,
        deflections_x=False,
        magnification=False,
        contribution_map=False,
        auto_filename="subplot_tracer",
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
            auto_labels=mp.AutoLabels(filename=auto_filename),
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

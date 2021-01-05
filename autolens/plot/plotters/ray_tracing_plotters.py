from autoarray.structures import arrays, grids
from autoarray.plot.plotters import abstract_plotters
from autogalaxy.plot.plotters import lensing_obj_plotter
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autogalaxy.plot.plotters import plane_plotters
from autolens.lens import ray_tracing


class TracerPlotter(lensing_obj_plotter.LensingObjPlotter):
    def __init__(
        self,
        tracer: ray_tracing.Tracer,
        grid: grids.Grid,
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
            "border", value=self.grid.mask.geometry.border_grid_sub_1.in_1d_binned
        )

        if border is not None:
            if plane_index > 0:
                border = self.tracer.traced_grids_of_planes_from_grid(grid=border)[
                    plane_index
                ]

        if plane_index == 0:
            critical_curves = self.extract_2d(
                "critical_curves", self.lensing_obj.critical_curves
            )
        else:
            critical_curves = None

        if plane_index == 1:
            caustics = self.extract_2d("caustics", self.lensing_obj.caustics)
        else:
            caustics = None

        return self.visuals_2d + lensing_visuals.Visuals2D(
            origin=self.extract_2d(
                "origin", value=grids.GridIrregular(grid=[self.grid.origin])
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

    @abstract_plotters.for_figure
    def figure_image(self):
        self.mat_plot_2d.plot_array(
            array=self.tracer.image_from_grid(grid=self.grid),
            visuals_2d=self.visuals_with_include_2d,
        )

    @abstract_plotters.for_figure
    def figure_contribution_map(self):

        self.mat_plot_2d.plot_array(
            array=self.tracer.contribution_map, visuals_2d=self.visuals_with_include_2d
        )

    @abstract_plotters.for_figure_with_index
    def figure_plane_image_of_plane(self, plane_index):

        plane_plotter = self.plane_plotter_from(plane_index=plane_index)

        plane_plotter.figure_plane_image()

    def figure_individuals(
        self,
        plot_image=False,
        plot_source_plane=False,
        plot_convergence=False,
        plot_potential=False,
        plot_deflections=False,
        plot_magnification=False,
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

        if plot_image:
            self.figure_image()

        if plot_source_plane:
            self.figure_plane_image_of_plane(plane_index=len(self.tracer.planes) - 1)

        if plot_convergence:
            self.figure_convergence()

        if plot_potential:
            self.figure_potential()

        if plot_deflections:
            self.figure_deflections_y()
            self.figure_deflections_x()

        if plot_magnification:
            self.figure_magnification()

    @abstract_plotters.for_subplot
    def subplot_tracer(self):
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

        number_subplots = 6

        self.open_subplot_figure(number_subplots=number_subplots)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        self.figure_image()

        if self.tracer.has_mass_profile:
            self.setup_subplot(number_subplots=number_subplots, subplot_index=2)

            self.figure_convergence()

            self.setup_subplot(number_subplots=number_subplots, subplot_index=3)

            self.figure_potential()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=4)

        self.figure_plane_image_of_plane(plane_index=len(self.tracer.planes) - 1)

        if self.tracer.has_mass_profile:
            self.setup_subplot(number_subplots=number_subplots, subplot_index=5)

            self.figure_deflections_y()

            self.setup_subplot(number_subplots=number_subplots, subplot_index=6)

            self.figure_deflections_x()

        self.mat_plot_2d.output.subplot_to_figure()

        self.mat_plot_2d.figure.close()

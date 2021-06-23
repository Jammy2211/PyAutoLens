from autoarray.plot.mat_wrap import mat_plot as mp
from autoarray.plot import abstract_plotters
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autolens.dataset import point_dataset


class PointDictPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        point_dict: point_dataset.PointDict,
        mat_plot_1d: lensing_mat_plot.MatPlot1D = lensing_mat_plot.MatPlot1D(),
        visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
        include_1d: lensing_include.Include1D = lensing_include.Include1D(),
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.point_dict = point_dict

    @property
    def visuals_with_include_1d(self) -> lensing_visuals.Visuals1D:
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

        return self.visuals_1d

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

        return self.visuals_2d

    def point_dataset_plotter_from(self, name):

        return PointDatasetPlotter(
            point_dataset=self.point_dict[name],
            mat_plot_1d=self.mat_plot_1d,
            include_1d=self.include_1d,
            visuals_1d=self.visuals_1d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

    def subplot(self):

        self.open_subplot_figure(number_subplots=len(self.point_dict))

        for name in self.point_dict.keys():

            point_dataset_plotter = self.point_dataset_plotter_from(name=name)

            point_dataset_plotter.figures_2d(positions=True, fluxes=True)

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_point_dict")

        self.close_subplot_figure()

    def subplot_positions(self):

        self.open_subplot_figure(number_subplots=len(self.point_dict))

        for name in self.point_dict.keys():

            point_dataset_plotter = self.point_dataset_plotter_from(name=name)

            point_dataset_plotter.figures_2d(positions=True)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_point_dict_positions"
        )

        self.close_subplot_figure()

    def subplot_fluxes(self):

        self.open_subplot_figure(number_subplots=len(self.point_dict))

        for name in self.point_dict.keys():

            point_dataset_plotter = self.point_dataset_plotter_from(name=name)

            point_dataset_plotter.figures_2d(fluxes=True)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_point_dict_fluxes"
        )

        self.close_subplot_figure()


class PointDatasetPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        point_dataset: point_dataset.PointDataset,
        mat_plot_1d: lensing_mat_plot.MatPlot1D = lensing_mat_plot.MatPlot1D(),
        visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
        include_1d: lensing_include.Include1D = lensing_include.Include1D(),
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.point_dataset = point_dataset

    @property
    def visuals_with_include_1d(self) -> lensing_visuals.Visuals1D:
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

        return self.visuals_1d

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

        return self.visuals_2d

    def figures_2d(self, positions: bool = False, fluxes: bool = False):

        if positions:

            self.mat_plot_2d.plot_grid(
                grid=self.point_dataset.positions,
                y_errors=self.point_dataset.positions_noise_map,
                x_errors=self.point_dataset.positions_noise_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title=f"{self.point_dataset.name} Positions",
                    filename="point_dataset_positions",
                ),
                buffer=0.1,
            )

        # nasty hack to ensure subplot index between 2d and 1d plots are syncs. Need a refactor that mvoes subplot
        # functionality out of mat_plot and into plotter.

        if (
            self.mat_plot_1d.subplot_index is not None
            and self.mat_plot_2d.subplot_index is not None
        ):

            self.mat_plot_1d.subplot_index = max(
                self.mat_plot_1d.subplot_index, self.mat_plot_2d.subplot_index
            )

        if fluxes:

            if self.point_dataset.fluxes is not None:

                self.mat_plot_1d.plot_yx(
                    y=self.point_dataset.fluxes,
                    y_errors=self.point_dataset.fluxes_noise_map,
                    visuals_1d=self.visuals_with_include_1d,
                    auto_labels=mp.AutoLabels(
                        title=f" {self.point_dataset.name} Fluxes",
                        filename="point_dataset_fluxes",
                        xlabel="Point Number",
                    ),
                    plot_axis_type_override="errorbar",
                )

    def subplot(
        self,
        positions: bool = False,
        fluxes: bool = False,
        auto_filename="subplot_point_dataset",
    ):

        self._subplot_custom_plot(
            positions=positions,
            fluxes=fluxes,
            auto_labels=mp.AutoLabels(filename=auto_filename),
        )

    def subplot_point_dataset(self):
        self.subplot(positions=True, fluxes=True)

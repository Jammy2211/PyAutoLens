from os import path

from autolens.analysis.plotter_interface import PlotterInterface

from autolens.point.fit.dataset import FitPointDataset
from autolens.point.plot.fit_point_plotters import FitPointDatasetPlotter
from autolens.point.dataset import PointDataset
from autolens.point.plot.point_dataset_plotters import PointDatasetPlotter

from autolens.analysis.plotter_interface import plot_setting


class PlotterInterfacePoint(PlotterInterface):
    def dataset_point(self, dataset: PointDataset):
        """
        Output visualization of an `PointDataset` dataset, typically before a model-fit is performed.

        Images are output to the `image` folder of the `image_path` in a subfolder called `dataset`. When used with
        a non-linear search the `image_path` is the output folder of the non-linear search.
        `.
        Visualization includes individual images of the different points of the dataset (e.g. the positions and fluxes)

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `dataset` header.

        Parameters
        ----------
        dataset
            The imaging dataset which is visualized.
        """

        def should_plot(name):
            return plot_setting(section=["point_dataset"], name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="dataset")

        dataset_plotter = PointDatasetPlotter(
            dataset=dataset, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        dataset_plotter.figures_2d(
            positions=should_plot("positions"),
            fluxes=should_plot("fluxes"),
        )

        mat_plot_2d = self.mat_plot_2d_from(subfolders="")

        dataset_plotter = PointDatasetPlotter(
            dataset=dataset, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("subplot_dataset"):
            dataset_plotter.subplot_dataset()

    def fit_point(
        self,
        fit: FitPointDataset,
        during_analysis: bool,
        subfolders: str = "fit_dataset",
    ):
        """
        Visualizes a `FitPointDataset` object, which fits an imaging dataset.

        Images are output to the `image` folder of the `image_path` in a subfolder called `fit`. When
        used with a non-linear search the `image_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `FitImaging` inferred by the search so far.

        Visualization includes individual images of attributes of the `FitPointDataset` (e.g. the model data and data)
        and a subplot of all `FitPointDataset`'s images on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under the
        [fit] header.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitPointDataset` of the non-linear search which is used to plot the fit.
        during_analysis
            Whether visualization is performed during a non-linear search or once it is completed.
        visuals_2d
            An object containing attributes which may be plotted over the figure (e.g. the centres of mass and light
            profiles).
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_point_dataset"], name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders=subfolders)

        fit_plotter = FitPointDatasetPlotter(
            fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        fit_plotter.figures_2d(
            positions=should_plot("positions"),
            fluxes=should_plot("fluxes"),
        )

        mat_plot_2d = self.mat_plot_2d_from(subfolders="")

        fit_plotter = FitPointDatasetPlotter(
            fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("subplot_fit"):
            fit_plotter.subplot_fit()

        if not during_analysis and should_plot("all_at_end_png"):
            mat_plot_2d = self.mat_plot_2d_from(
                subfolders=path.join("fit_dataset", "end"),
            )

            fit_plotter = FitPointDatasetPlotter(
                fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
            )

            fit_plotter.figures_2d(positions=True, fluxes=True)

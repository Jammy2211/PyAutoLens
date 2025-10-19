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

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        is the output folder of the non-linear search.

        Visualization includes individual images of the different points of the dataset (e.g. the positions and fluxes)

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `point_dataset` header.

        Parameters
        ----------
        dataset
            The imaging dataset which is visualized.
        """

        def should_plot(name):
            return plot_setting(section=["point_dataset"], name=name)

        mat_plot_2d = self.mat_plot_2d_from()

        dataset_plotter = PointDatasetPlotter(dataset=dataset, mat_plot_2d=mat_plot_2d)

        if should_plot("subplot_dataset"):
            dataset_plotter.subplot_dataset()

    def fit_point(
        self,
        fit: FitPointDataset,
        quick_update: bool = False,
    ):
        """
        Visualizes a `FitPointDataset` object, which fits an imaging dataset.

        Images are output to the `image` folder of the `image_path` in a subfolder called `fit`. When
        used with a non-linear search the `image_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `FitImaging` inferred by the search so far.

        Visualization includes a subplot of individual images of attributes of the `FitPointDataset` (e.g. the model
        data and data) and .fits files containing its attributes grouped together.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `fit` and `fit_point_dataset` headers.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitPointDataset` of the non-linear search which is used to plot the fit.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_point_dataset"], name=name)

        mat_plot_2d = self.mat_plot_2d_from()

        fit_plotter = FitPointDatasetPlotter(fit=fit, mat_plot_2d=mat_plot_2d)

        if should_plot("subplot_fit") or quick_update:
            fit_plotter.subplot_fit()

        if quick_update:
            return

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

        Parameters
        ----------
        dataset
            The imaging dataset which is visualized.
        """

        def should_plot(name):
            return plot_setting(section=["point_dataset"], name=name)

        output = self.output_from()

        dataset_plotter = PointDatasetPlotter(dataset=dataset, output=output)

        if should_plot("subplot_dataset"):
            dataset_plotter.subplot_dataset()

    def fit_point(
        self,
        fit: FitPointDataset,
        quick_update: bool = False,
    ):
        """
        Visualizes a `FitPointDataset` object, which fits an imaging dataset.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitPointDataset` of the non-linear search which is used to plot the fit.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_point_dataset"], name=name)

        output = self.output_from()

        fit_plotter = FitPointDatasetPlotter(fit=fit, output=output)

        if should_plot("subplot_fit") or quick_update:
            fit_plotter.subplot_fit()

        if quick_update:
            return

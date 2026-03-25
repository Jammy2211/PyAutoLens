from autolens.analysis.plotter import Plotter

from autolens.point.fit.dataset import FitPointDataset
from autolens.point.plot.fit_point_plots import subplot_fit as subplot_fit_point
from autolens.point.dataset import PointDataset
from autolens.point.plot.point_dataset_plots import subplot_dataset

from autolens.analysis.plotter import plot_setting


class PlotterPoint(Plotter):
    def dataset_point(self, dataset: PointDataset):
        """
        Output visualization of a `PointDataset` dataset.

        Parameters
        ----------
        dataset
            The point dataset which is visualized.
        """

        def should_plot(name):
            return plot_setting(section=["point_dataset"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        if should_plot("subplot_dataset"):
            subplot_dataset(dataset, output_path=output_path, output_format=fmt)

    def fit_point(
        self,
        fit: FitPointDataset,
        quick_update: bool = False,
    ):
        """
        Visualizes a `FitPointDataset` object.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitPointDataset` of the non-linear search.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_point_dataset"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        if should_plot("subplot_fit") or quick_update:
            subplot_fit_point(fit, output_path=output_path, output_format=fmt)

        if quick_update:
            return

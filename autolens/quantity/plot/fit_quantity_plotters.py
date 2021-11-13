from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta

from autogalaxy.quantity.fit_quantity import FitQuantity

from autogalaxy.plot.mat_wrap.mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D
from autogalaxy.plot.mat_wrap.include import Include2D

from autolens.plot.abstract_plotters import Plotter


class FitQuantityPlotter(Plotter):
    def __init__(
        self,
        fit: FitQuantity,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):

        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.fit = fit

        self._fit_imaging_meta_plotter = FitImagingPlotterMeta(
            fit=self.fit,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

        self.figures_2d = self._fit_imaging_meta_plotter.figures_2d
        self.subplot = self._fit_imaging_meta_plotter.subplot

    def get_visuals_2d(self) -> Visuals2D:
        return self.get_2d.via_fit_imaging_from(fit=self.fit)

    def subplot_fit_quantity(self):
        return self.subplot(
            image=True,
            signal_to_noise_map=True,
            model_image=True,
            residual_map=True,
            normalized_residual_map=True,
            chi_squared_map=True,
            auto_filename="subplot_fit_quantity",
        )

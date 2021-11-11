import autoarray.plot as aplt
from autoarray.fit.plot.fit_imaging_plotters import AbstractFitImagingPlotter

from autogalaxy.quantity.fit_quantity import FitQuantity

from autogalaxy.plot.mat_wrap.mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D
from autogalaxy.plot.mat_wrap.include import Include2D


class FitQuantityPlotter(AbstractFitImagingPlotter):
    def __init__(
        self,
        fit: FitQuantity,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):

        super().__init__(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

    @property
    def visuals_with_include_2d(self) -> Visuals2D:

        return self.visuals_2d + self.visuals_2d.__class__()

    def figures_2d(
        self,
        image: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
    ):

        if image:

            self.mat_plot_2d.plot_array(
                array=self.fit.data,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=aplt.AutoLabels(title="Data", filename="data"),
            )

        super().figures_2d(
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_image=model_image,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
        )

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

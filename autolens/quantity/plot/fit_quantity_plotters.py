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
        """
        Plots the attributes of `FitQuantity` objects using the matplotlib method `imshow()` and many
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2d` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `FitQuantity` and plotted via the visuals object, if the corresponding entry is `True` in
        the `Include1D` or `Include2D` object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        fit
            The fit to an interferometer dataset the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `FitQuantity` are extracted and plotted as visuals for 2D plots.
        """
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
        """
        Standard subplot of the attributes of the plotter's `FitQuantity` object.
        """
        return self.subplot(
            image=True,
            signal_to_noise_map=True,
            model_image=True,
            residual_map=True,
            normalized_residual_map=True,
            chi_squared_map=True,
            auto_filename="subplot_fit_quantity",
        )

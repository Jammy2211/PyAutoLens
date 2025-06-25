from autoarray.plot.wrap.base.abstract import set_backend

set_backend()

from autolens.plot.get_visuals.one_d import GetVisuals1D
from autolens.plot.get_visuals.two_d import GetVisuals2D

from autogalaxy.plot.abstract_plotters import AbstractPlotter

from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.plot.visuals.two_d import Visuals2D
from autogalaxy.plot.include.one_d import Include1D
from autogalaxy.plot.include.two_d import Include2D


class Plotter(AbstractPlotter):

    def __init__(
        self,
        mat_plot_1d: MatPlot1D = None,
        visuals_1d: Visuals1D = None,
        include_1d: Include1D = None,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
        include_2d: Include2D = None,
    ):

        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

        self.visuals_1d = visuals_1d or Visuals1D()
        self.include_1d = include_1d or Include1D()
        self.mat_plot_1d = mat_plot_1d or MatPlot1D()

        self.visuals_2d = visuals_2d or Visuals2D()
        self.include_2d = include_2d or Include2D()
        self.mat_plot_2d = mat_plot_2d or MatPlot2D()

    @property
    def get_1d(self):
        return GetVisuals1D(visuals=self.visuals_1d, include=self.include_1d)

    @property
    def get_2d(self):
        return GetVisuals2D(visuals=self.visuals_2d, include=self.include_2d)

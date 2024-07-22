import numpy as np

from autofit.mock import *  # noqa
from autoarray.mock import *  # noqa
from autogalaxy.mock import *  # noqa
from autolens import Tracer

from autolens.imaging.mock.mock_fit_imaging import MockFitImaging  # noqa
from autolens.lens.mock.mock_tracer import MockTracer  # noqa
from autolens.lens.mock.mock_tracer import MockTracerPoint  # noqa
from autolens.point.mock.mock_point_solver import MockPointSolver  # noqa


class NullTracer(Tracer):
    def __init__(self):
        super().__init__([])

    def deflections_yx_2d_from(self, grid):
        return np.zeros_like(grid)

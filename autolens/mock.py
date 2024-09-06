import numpy as np

from autofit.jax_wrapper import register_pytree_node_class
from autofit.mock import *  # noqa
from autoarray.mock import *  # noqa
from autogalaxy.mock import *  # noqa
from autolens import Tracer

from autolens.imaging.mock.mock_fit_imaging import MockFitImaging  # noqa
from autolens.lens.mock.mock_tracer import MockTracer  # noqa
from autolens.lens.mock.mock_tracer import MockTracerPoint  # noqa
from autolens.point.mock.mock_solver import MockPointSolver  # noqa


@register_pytree_node_class
class NullTracer(Tracer):
    def __init__(self):
        super().__init__([])

    def deflections_yx_2d_from(self, grid):
        return np.zeros_like(grid.array)

    def deflections_between_planes_from(self, grid, plane_i=0, plane_j=-1):
        return np.zeros_like(grid.array)

    def tree_flatten(self):
        """
        Flatten this model as a PyTree.
        """
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a PyTree into a model.
        """
        return cls()

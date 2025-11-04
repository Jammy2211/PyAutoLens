import jax.numpy as jnp

from autoconf.jax_wrapper import register_pytree_node_class
from autofit.mock import *  # noqa
from autoarray.mock import *  # noqa
from autogalaxy.mock import *  # noqa
import autoarray as aa
from autolens import Tracer

from autolens.imaging.mock.mock_fit_imaging import MockFitImaging  # noqa
from autolens.lens.mock.mock_tracer import MockTracer  # noqa
from autolens.lens.mock.mock_tracer import MockTracerPoint  # noqa
from autolens.point.mock.mock_solver import MockPointSolver  # noqa


class NullTracer(Tracer):
    def __init__(self):
        super().__init__([])

    def deflections_yx_2d_from(self, grid):
        return jnp.zeros_like(grid.array)

    def deflections_between_planes_from(self, grid, plane_i=0, plane_j=-1):
        return jnp.zeros_like(grid.array)

    def magnification_2d_via_hessian_from(
        self, grid, buffer: float = 0.01, deflections_func=None
    ) -> aa.ArrayIrregular:
        return aa.ArrayIrregular(values=jnp.ones(grid.shape[0]))

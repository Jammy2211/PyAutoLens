import pytest

from autoarray.structures.triangles.shape import Circle
from autolens.mock import NullTracer
from autolens.point.solver.abstract_solver import ShapeSolver


@pytest.fixture
def solver(grid):
    return ShapeSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.01,
    )


def test_solver_basic(solver):
    result = solver.solve(
        tracer=NullTracer(),
        shape=Circle(
            0.0,
            0.0,
            radius=0.01,
        ),
    )
    assert list(map(tuple, result)) == [
        (-0.012003766846269881, 0.0078125),
        (-0.0029826688901819823, -0.0078125),
        (0.015059527021993818, -0.0078125),
        (0.010548978043949867, 0.015625),
    ]

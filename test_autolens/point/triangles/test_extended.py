import pytest

from autoarray.structures.triangles.shape import Circle
from autolens.mock import NullTracer
from autolens.point.solver.shape_solver import ShapeSolver


@pytest.fixture
def solver(grid):
    return ShapeSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.01,
    )


def test_solver_basic(solver):
    assert solver.find_magnification(
        tracer=NullTracer(),
        shape=Circle(
            0.0,
            0.0,
            radius=0.01,
        ),
    ) == pytest.approx(1.0, abs=0.01)

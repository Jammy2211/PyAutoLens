from typing import Tuple

import pytest

import autolens as al
import autogalaxy as ag
from autolens.mock import NullTracer
from autolens.point.solver import PointSolver


@pytest.fixture
def solver():
    return PointSolver(
        pixel_scale_precision=0.01,
    )


def test_solver_simple(solver, grid):

    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=ag.mp.Isothermal(
                    centre=(0.0, 0.0),
                    einstein_radius=1.0,
                ),
            )
        ]
    )

    assert solver.solve(
        lensing_obj=tracer,
        grid=grid,
        source_plane_coordinate=(0.0, 0.0),
    )


def test_steps(solver):
    assert solver.n_steps == 7


@pytest.mark.parametrize(
    "source_plane_coordinate",
    [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (0.1, 0.1),
        (-1.0, -1.0),
    ],
)
def test_trivial(
    source_plane_coordinate: Tuple[float, float],
    grid,
):
    solver = PointSolver.for_grid(
        lensing_obj=NullTracer(),
        grid=grid,
        pixel_scale_precision=0.01,
    )
    (coordinates,) = solver.solve(
        source_plane_coordinate=source_plane_coordinate,
    )
    assert coordinates == pytest.approx(source_plane_coordinate, abs=1.0e-1)


def test_real_example(grid, tracer):
    solver = PointSolver.for_grid(
        grid=grid,
        lensing_obj=tracer,
        pixel_scale_precision=0.001,
    )
    result = solver.solve((0.07, 0.07))
    for r in result:
        print(r)
    assert len(result) == 5

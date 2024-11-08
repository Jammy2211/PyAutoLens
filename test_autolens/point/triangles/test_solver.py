from typing import Tuple

import pytest

import autolens as al
import autogalaxy as ag
from autolens.mock import NullTracer
from autolens.point.solver import PointSolver


@pytest.fixture
def solver(grid):
    return PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.01,
    )


def test_solver_basic(solver):
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=ag.mp.Isothermal(
                    centre=(0.0, 0.0),
                    einstein_radius=1.0,
                ),
            ),
            al.Galaxy(
                redshift=1.0,
            ),
        ]
    )

    assert solver.solve(
        tracer=tracer,
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
        grid=grid,
        pixel_scale_precision=0.01,
    )
    coordinates = solver.solve(
        tracer=NullTracer(),
        source_plane_coordinate=source_plane_coordinate,
    )

    assert coordinates[0] == pytest.approx(source_plane_coordinate, abs=1.0e-1)


def test_real_example(grid, tracer):
    solver = PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.001,
    )

    result = solver.solve(tracer=tracer, source_plane_coordinate=(0.07, 0.07))

    assert len(result) == 5


@pytest.mark.parametrize(
    "neighbor_degree, expected",
    [
        (0, 1),
        (1, 4),
        (2, 10),
    ],
)
def test_neighbor_order(
    solver,
    neighbor_degree,
    expected,
):
    assert (
        len(
            solver.solve(
                NullTracer(),
                source_plane_coordinate=(0.0, 0.0),
                neighbor_order=neighbor_degree,
            )
        )
        == expected
    )

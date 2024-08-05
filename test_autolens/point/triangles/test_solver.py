from typing import Tuple

import pytest

import autolens as al
import autogalaxy as ag
from autoarray.structures.triangles.array import ArrayTriangles
from autolens.mock import NullTracer
from autolens.point.triangles.triangle_solver import TriangleSolver
from autolens.point.triangles.visualise import visualise


@pytest.fixture
def solver(grid):
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

    return TriangleSolver.for_grid(
        lensing_obj=tracer,
        grid=grid,
        pixel_scale_precision=0.01,
    )


def test_solver(solver):
    assert solver.solve(
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
    solver = TriangleSolver.for_grid(
        lensing_obj=NullTracer(),
        grid=grid,
        pixel_scale_precision=0.01,
        array_triangles_cls=ArrayTriangles,
    )
    (coordinates,) = solver.solve(
        source_plane_coordinate=source_plane_coordinate,
    )
    assert coordinates == pytest.approx(source_plane_coordinate, abs=1.0e-1)


def test_real_example(grid, tracer):
    solver = TriangleSolver.for_grid(
        grid=grid,
        lensing_obj=tracer,
        pixel_scale_precision=0.001,
        array_triangles_cls=ArrayTriangles,
    )

    for step in solver.steps((0.07, 0.07)):
        visualise(step)

    result = solver.solve((0.07, 0.07))
    for r in result:
        print(r)
    assert len(result) == 5

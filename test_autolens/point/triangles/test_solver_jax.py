import time
from typing import Tuple

import pytest

import autolens as al
import autogalaxy as ag
import autofit as af
from autofit.jax_wrapper import use_jax
from autolens import PointSolver

try:
    if use_jax:
        from autoarray.structures.triangles.jax_array import ArrayTriangles
    else:
        from autoarray.structures.triangles.array import ArrayTriangles
except ImportError:
    from autoarray.structures.triangles.array import ArrayTriangles

from autolens.mock import NullTracer


pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def register(tracer):
    af.Model.from_instance(tracer)


@pytest.fixture
def solver(grid):
    return PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.01,
        array_triangles_cls=ArrayTriangles,
    )


def test_solver(solver):
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
        tracer,
        source_plane_coordinate=(0.0, 0.0),
    )


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
    solver,
):
    coordinates = solver.solve(
        NullTracer(),
        source_plane_coordinate=source_plane_coordinate,
    )
    assert coordinates[0] == pytest.approx(source_plane_coordinate, abs=1.0e-1)


def test_real_example(grid, tracer):
    solver = PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.001,
        array_triangles_cls=ArrayTriangles,
    )

    result = solver.solve(tracer, (0.07, 0.07))
    assert len(result) == 5


def _test_jax(grid):
    solver = PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.001,
        array_triangles_cls=ArrayTriangles,
    )

    solver.solve(NullTracer(), (0.07, 0.07))

    repeats = 1000
    start = time.time()
    for _ in range(repeats):
        result = solver.solve(NullTracer(), (0.07, 0.07))

    print(result)

    print(f"Time taken: {(time.time() - start) / repeats}")

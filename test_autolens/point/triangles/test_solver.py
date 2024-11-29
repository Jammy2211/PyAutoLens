from typing import Tuple

import numpy as np
import pytest

import autolens as al
import autogalaxy as ag
from autoarray.structures.triangles.coordinate_array import CoordinateArrayTriangles
from autoarray.structures.triangles.coordinate_array.jax_coordinate_array import (
    CoordinateArrayTriangles as JAXTriangles,
)
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


def triangle_set(triangles):
    return {
        tuple(sorted([tuple(np.round(pair, 4)) for pair in triangle]))
        for triangle in triangles.triangles.tolist()
        if not np.isnan(triangle).any()
    }


def test_real_example_jax(grid, tracer):
    jax_solver = PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.001,
        array_triangles_cls=JAXTriangles,
    )

    result = jax_solver.solve(
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
    )

    assert len(result) == 5


def test_real_example_normal(grid, tracer):
    jax_solver = PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.001,
        array_triangles_cls=CoordinateArrayTriangles,
    )

    result = jax_solver.solve(
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
    )

    assert len(result) == 5

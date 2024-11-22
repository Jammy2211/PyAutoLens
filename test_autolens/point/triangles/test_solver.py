from typing import Tuple

import numpy as np
import pytest

import autolens as al
import autogalaxy as ag
from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoarray.structures.triangles.coordinate_array import CoordinateArrayTriangles
from autoarray.structures.triangles.jax_coordinate_array import (
    CoordinateArrayTriangles as JAXTriangles,
)
from autoarray.structures.triangles.shape import Point
from autolens.mock import NullTracer
from autolens.point.solver import PointSolver
from autolens.point.visualise import visualise, plot_triangles_compare


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


def test_real_example(grid, tracer):
    solver = PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.001,
    )
    jax_solver = PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.001,
        array_triangles_cls=JAXTriangles,
    )

    for step, jax_step in zip(
        solver.steps(tracer=tracer, shape=Point(0.07, 0.07)),
        jax_solver.steps(tracer=tracer, shape=Point(0.07, 0.07)),
    ):
        point = Point(2.0, 0.0)

        triangles = step.initial_triangles
        jax_triangles = jax_step.initial_triangles

        indices = step.initial_triangles.with_vertices(
            step.initial_triangles.vertices
        ).containing_indices(point)
        jax_indices = jax_step.initial_triangles.with_vertices(
            jax_step.initial_triangles.vertices
        ).containing_indices(point)

        new_triangles = triangles.for_indexes(indices)
        new_jax_triangles = jax_triangles.for_indexes(jax_indices)

        print(indices)
        print(jax_indices)

    result = solver.solve(tracer=tracer, source_plane_coordinate=(0.07, 0.07))

    assert len(result) == 5


def test_broken_step(grid, tracer):
    solver = PointSolver(
        scale=0.5,
        pixel_scale_precision=0.001,
        initial_triangles=JAXTriangles(
            coordinates=np.array([[6.0, 3.0]]),
            side_length=0.5,
            flipped=True,
            y_offset=-0.25 * HEIGHT_FACTOR,
        ),
    )
    step = next(
        solver.steps(
            tracer=tracer,
            shape=Point(0.07, 0.07),
        )
    )
    visualise(step)

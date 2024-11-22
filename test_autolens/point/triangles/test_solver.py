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
from autolens.point.visualise import visualise, plot_triangles_compare, plot_triangles


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

    point = Point(0.07, 0.07)

    for step, jax_step in zip(
        solver.steps(tracer=tracer, shape=point),
        jax_solver.steps(tracer=tracer, shape=point),
    ):
        initial_triangles = step.initial_triangles
        jax_initial_triangles = jax_step.initial_triangles

        initial_triangle_set = triangle_set(initial_triangles)
        jax_initial_triangle_set = triangle_set(jax_initial_triangles)

        print(
            "difference in initial",
            initial_triangle_set.difference(jax_initial_triangle_set),
        )

        print("Difference in vertices")
        print(
            {
                tuple(map(float, np.round(v, 3))) for v in initial_triangles.vertices
            }.difference(
                {
                    tuple(map(float, np.round(v, 3)))
                    for v in jax_initial_triangles.vertices
                    if not np.isnan(v).any()
                }
            )
        )

        source_triangles = triangle_set(step.source_triangles)
        jax_source_triangles = triangle_set(jax_step.source_triangles)

        print(
            "in source but not jax", source_triangles.difference(jax_source_triangles)
        )
        print(
            "in jax but not source", jax_source_triangles.difference(source_triangles)
        )

        if step.number == 2:
            break


def test_real_example_jax_only(grid, tracer):
    jax_solver = PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.001,
        array_triangles_cls=JAXTriangles,
    )

    for step in jax_solver.steps(
        tracer=tracer,
        shape=Point(
            0.07,
            0.07,
        ),
    ):
        triangles = step.initial_triangles

        print(triangles)
        visualise(step)


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

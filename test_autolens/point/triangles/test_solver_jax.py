import time
from typing import Tuple

import pytest

import autolens as al
import autogalaxy as ag
import autofit as af
import numpy as np
from autolens import PointSolver

try:
    from autoarray.structures.triangles.jax_array import ArrayTriangles
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
    tracer = ag.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.0,
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
    coordinates = coordinates.array[~np.isnan(coordinates.array).any(axis=1)]
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
    sizes = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
    run_times = []
    init_times = []

    for size in sizes:
        start = time.time()
        solver = PointSolver.for_grid(
            grid=grid,
            pixel_scale_precision=0.001,
            array_triangles_cls=ArrayTriangles,
            max_containing_size=size,
        )

        solver.solve(NullTracer(), (0.07, 0.07))

        repeats = 100

        done_init_time = time.time()
        init_time = done_init_time - start
        for _ in range(repeats):
            _ = solver.solve(NullTracer(), (0.07, 0.07))

        # print(result)

        init_times.append(init_time)

        run_time = (time.time() - done_init_time) / repeats
        run_times.append(run_time)

        print(f"Time taken for {size}: {run_time} ({init_time} to init)")

    from matplotlib import pyplot as plt

    plt.plot(sizes, run_times)
    plt.show()

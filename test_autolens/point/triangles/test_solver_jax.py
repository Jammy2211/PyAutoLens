import pytest

import autolens as al
import autogalaxy as ag
from autoarray.structures.triangles.jax_array import ArrayTriangles
from autolens.point.triangles.triangle_solver import TriangleSolver


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
        ArrayTriangles=ArrayTriangles,
    )


def test_solver(solver):
    assert solver.solve(
        source_plane_coordinate=(0.0, 0.0),
    )

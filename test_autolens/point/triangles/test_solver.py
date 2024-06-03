import pytest

import autolens as al
import autogalaxy as ag
from autolens.point.triangles.triangle_solver import TriangleSolver


@pytest.fixture
def solver():
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=ag.mp.Isothermal(
                    centre=(2.0, 1.0),
                    einstein_radius=1.0,
                ),
            )
        ]
    )
    grid = al.Grid2D.uniform(
        shape_native=(100, 100),
        pixel_scales=0.05,
    )

    return TriangleSolver(
        tracer=tracer,
        grid=grid,
        min_pixel_scale=0.01,
    )


def test_solver(solver):
    assert solver.solve(
        source_plane_coordinate=(0.0, 0.0),
    )


def test_steps(solver):
    assert solver.n_steps == 3

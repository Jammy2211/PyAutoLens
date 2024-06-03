from typing import Tuple

import numpy as np
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
                    centre=(0.0, 0.0),
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
        lensing_obj=tracer,
        grid=grid,
        pixel_scale_precision=0.01,
    )


def test_solver(solver):
    assert solver.solve(
        source_plane_coordinate=(0.0, 0.0),
    )


def test_steps(solver):
    assert solver.n_steps == 3


class NullTracer(al.Tracer):
    def __init__(self):
        super().__init__([])

    def deflections_yx_2d_from(self, grid):
        return np.zeros_like(grid)


@pytest.mark.parametrize(
    "source_plane_coordinate",
    [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (0.1, 0.1),
        (-1.0, -1.0),
    ],
)
def test_trivial(
    source_plane_coordinate: Tuple[float, float],
):
    solver = TriangleSolver(
        lensing_obj=NullTracer(),
        grid=al.Grid2D.uniform(
            shape_native=(100, 100),
            pixel_scales=0.05,
        ),
        pixel_scale_precision=0.01,
    )
    (coordinates,) = solver.solve(
        source_plane_coordinate=source_plane_coordinate,
    )
    assert coordinates == pytest.approx(source_plane_coordinate, abs=1.0e-2)


def test_real_example():
    grid = al.Grid2D.uniform(
        shape_native=(100, 100),
        pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
    )

    isothermal_mass_profile = al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=isothermal_mass_profile,
    )

    point_source = al.ps.PointSourceChi(centre=(0.07, 0.07))

    source_galaxy = al.Galaxy(redshift=1.0, point_0=point_source)

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    solver = TriangleSolver(
        grid=grid,
        lensing_obj=tracer,
        pixel_scale_precision=0.001,
    )
    assert solver.solve((0.07, 0.07))

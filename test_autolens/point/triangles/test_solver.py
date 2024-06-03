import autolens as al
import autogalaxy as ag
from autolens.point.triangles.triangle_solver import TriangleSolver


def test_solver():
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

    solver = TriangleSolver(
        tracer=tracer,
        grid=grid,
    )

    assert solver.solve(
        source_plane_coordinate=(0.0, 0.0),
    )

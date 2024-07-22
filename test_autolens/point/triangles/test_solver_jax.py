import pytest

import autolens as al
import autogalaxy as ag
from autolens.point.triangles.triangle_solver import TriangleSolver


def test_solver(solver):
    assert solver.solve(
        source_plane_coordinate=(0.0, 0.0),
    )

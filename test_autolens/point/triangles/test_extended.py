import numpy as np
import pytest

from autoarray.structures.triangles.shape import Circle
from autolens.mock import NullTracer
from autolens.point.solver.shape_solver import ShapeSolver


@pytest.fixture
def solver(grid):
    return ShapeSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.001,
    )


def test_solver_basic(solver):
    """
    The identity lens: `NullTracer` deflects nothing, so the source plane *is* the image
    plane and a shape's image has exactly the shape's own area -- a magnification of 1.

    This is the solver's only original test, commented out since ~2024 and revived by the
    `image-source-mappings` validation suite. It is the one case where the answer is known
    without any lensing at all, so it isolates the tiling and the area accounting from the
    ray tracing: a magnification of 1 here means the kept triangles cover the shape and
    nothing else.
    """
    assert solver.find_magnification(
        tracer=NullTracer(),
        shape=Circle(
            0.0,
            0.0,
            radius=0.1,
        ),
    ) == pytest.approx(1.0, abs=0.1)


def test_solver_basic_is_one_image(solver):
    """
    With no deflection there is one image, not several, and its magnification is the total.
    """
    magnifications = solver.find_magnification(
        tracer=NullTracer(),
        shape=Circle(0.0, 0.0, radius=0.1),
        per_image=True,
    )

    assert len(magnifications) == 1
    assert magnifications[0] == pytest.approx(1.0, abs=0.1)


def test_solver_basic_regions_sit_on_the_shape(solver, grid):
    """
    The identity lens again, this time through `image_regions_from`: the one region must
    sit on top of the shape it was built from.
    """
    regions = solver.image_regions_from(
        tracer=NullTracer(),
        shape=Circle(0.0, 0.0, radius=0.1),
        grid=grid,
    )

    assert len(regions) == 1
    assert regions[0].centre == pytest.approx((0.0, 0.0), abs=0.05)
    assert np.asarray(regions[0].slim_indexes).size >= 1

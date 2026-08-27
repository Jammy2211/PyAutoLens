"""
Regression tests for PyAutoLens#480: ``PointSolver.solve`` found no images for a source on an
**intermediate** plane of a multi-plane tracer.

The triangle search honoured ``plane_redshift`` and traced to the requested plane, but the
magnification filter did not: it measured magnification against the whole tracer, i.e. the last
plane. Candidate images of an intermediate-plane source are strongly de-magnified when measured
that way (~1e-3-1e-5 in the issue's configuration), so ``magnification_threshold`` discarded every
one of them and the solve returned an empty grid.

That made any point-source modeling involving a source off the last plane impossible -- the
simulator returned 0 positions and ``AnalysisPoint`` likelihood evaluations could not pair model
images to data.

No test passed ``plane_redshift`` to the solver before this module existed, which is why the bug
shipped; the multi-plane solve path had no coverage at all.
"""

import numpy as np
import pytest

import autolens as al


@pytest.fixture
def solver_grid():
    return al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)


@pytest.fixture
def solver(solver_grid):
    return al.PointSolver.for_grid(
        grid=solver_grid, pixel_scale_precision=0.001, magnification_threshold=0.1
    )


@pytest.fixture
def lens_galaxy():
    return al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        ),
    )


@pytest.fixture
def intermediate_source():
    """
    A source at z=1.0 which is itself a deflector for the z=2.0 source behind it.
    """
    return al.Galaxy(
        redshift=1.0,
        mass=al.mp.Isothermal(
            centre=(0.02, 0.03),
            einstein_radius=0.2,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        ),
        point_0=al.ps.Point(centre=(0.02, 0.03)),
    )


@pytest.fixture
def multi_plane_tracer(lens_galaxy, intermediate_source):
    far_source = al.Galaxy(redshift=2.0, point_1=al.ps.Point(centre=(0.0, 0.0)))
    return al.Tracer(galaxies=[lens_galaxy, intermediate_source, far_source])


def test__intermediate_plane_source__images_are_found(solver, multi_plane_tracer):
    """
    The issue's repro. Before the fix this returned a (0, 2) empty grid.
    """
    result = solver.solve(
        tracer=multi_plane_tracer,
        source_plane_coordinate=(0.02, 0.03),
        plane_redshift=1.0,
    )

    assert len(result) == 4


def test__intermediate_plane__filter_is_what_used_to_reject_the_images(
    solver_grid, multi_plane_tracer
):
    """
    Disabling the magnification filter was the issue's diagnostic: the triangle search always found
    candidates, so any images lost were lost in the filter. With the filter measuring at the right
    plane it now keeps them, rather than the threshold having to be turned off.
    """
    kwargs = dict(
        tracer=multi_plane_tracer,
        source_plane_coordinate=(0.02, 0.03),
        plane_redshift=1.0,
    )

    unfiltered = al.PointSolver.for_grid(
        grid=solver_grid, pixel_scale_precision=0.001, magnification_threshold=0.0
    ).solve(**kwargs)

    filtered = al.PointSolver.for_grid(
        grid=solver_grid, pixel_scale_precision=0.001, magnification_threshold=0.1
    ).solve(**kwargs)

    assert len(unfiltered) > 0
    assert len(filtered) > 0


def test__last_plane_solve_is_unchanged(solver, multi_plane_tracer):
    """
    The fix must be a no-op for the last plane, which was never broken.
    """
    result = solver.solve(
        tracer=multi_plane_tracer,
        source_plane_coordinate=(0.0, 0.0),
        plane_redshift=2.0,
    )

    assert len(result) == 4


def test__plane_redshift_none_matches_last_plane_redshift(solver, multi_plane_tracer):
    """
    ``plane_redshift=None`` means "the last plane", so it must agree exactly with naming that
    plane's redshift explicitly -- the two now resolve to the same plane index and therefore the
    same magnification callable.
    """
    implicit = solver.solve(
        tracer=multi_plane_tracer, source_plane_coordinate=(0.0, 0.0)
    )
    explicit = solver.solve(
        tracer=multi_plane_tracer,
        source_plane_coordinate=(0.0, 0.0),
        plane_redshift=2.0,
    )

    assert np.array_equal(np.asarray(implicit.array), np.asarray(explicit.array))


def test__single_plane_solve_is_unchanged(solver, lens_galaxy):
    """
    The ordinary two-plane case shares the changed code path, so it is pinned here too.
    """
    tracer = al.Tracer(
        galaxies=[
            lens_galaxy,
            al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.0, 0.0))),
        ]
    )

    result = solver.solve(tracer=tracer, source_plane_coordinate=(0.0, 0.0))

    assert len(result) == 4


def test__magnification_is_measured_at_the_requested_plane(solver, multi_plane_tracer):
    """
    The heart of #480, asserted directly rather than through the image count: the same points get
    different magnifications depending on the plane, so the filter's verdict must depend on
    ``plane_redshift``.

    A threshold of 5.0 sits between the two planes' magnifications at these points, so measuring at
    z=1.0 keeps points the last plane's measurement rejects. If ``plane_redshift`` were ignored the
    two calls would be identical and this test would fail.
    """
    points = np.array([[1.0, 0.4], [-0.9, 0.7], [0.3, -1.2]])

    solver.magnification_threshold = 5.0

    at_intermediate = solver._filter_low_magnification(
        tracer=multi_plane_tracer, points=points, xp=np, plane_redshift=1.0
    )
    at_last = solver._filter_low_magnification(
        tracer=multi_plane_tracer, points=points, xp=np, plane_redshift=None
    )

    kept_intermediate = ~np.isnan(np.asarray(at_intermediate)[:, 0])
    kept_last = ~np.isnan(np.asarray(at_last)[:, 0])

    assert not np.array_equal(kept_intermediate, kept_last)


def test__plane_redshift_matching_no_plane__raises_value_error(
    solver, multi_plane_tracer
):
    """
    ``plane_index_via_redshift_from`` returns ``None`` when no plane matches within its 1e-8
    tolerance, which used to reach ``traced_grids_list[None]`` and raise a bare ``TypeError``
    naming nothing the caller controls. A mistyped redshift now says which redshifts exist.
    """
    with pytest.raises(ValueError) as exc_info:
        solver.solve(
            tracer=multi_plane_tracer,
            source_plane_coordinate=(0.02, 0.03),
            plane_redshift=1.5,
        )

    message = str(exc_info.value)

    assert "1.5" in message
    assert "0.5" in message and "1.0" in message and "2.0" in message

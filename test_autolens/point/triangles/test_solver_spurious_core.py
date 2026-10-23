"""
Regression tests for PyAutoLens#799: spurious candidates on a zero-core mass profile's centre.

A ``dPIEMassSph`` with ``r_core=0`` has a deflection which is discontinuous at its centre: a
triangle around the centre traces to a triangle inscribed in a circle of the profile's Einstein
radius, which can contain the source at every refinement step without containing a root. On the
cluster draws of autolens_profiling#439 this returned a candidate ~0.001" from a scaling member's
centre with a source-plane residual of 0.04". The magnification filter passes it, because in a
near-critical host region ``|mu|`` there is ~0.1-0.3.

The fixture reproduces that with one member inside a cored host whose lens-equation Jacobian at
the member is ~ -0.05 I (overcritical): two true images (``mu`` ~ 210 and ~ -155) and one spurious
candidate at ~6e-4" from the member centre, residual 0.066", ``mu`` 0.29.

``residual_threshold`` (default 10.0; ``None`` disables it and restores the pre-#799 solve) removes
it. The JAX tests skip
on the NumPy-only (no ``[optional]`` extras) CI leg.
"""

import importlib

import numpy as np
import pytest

import autolens as al

MEMBER_CENTRE = (0.3, 0.2)
SOURCE = (-0.0165, -0.011)
PRECISION = 0.001


def _core_tracer(host_sigma=480.0):
    host = al.Galaxy(
        redshift=0.5,
        mass=al.mp.dPIEMassSph(
            centre=(0.0, 0.0),
            sigma=host_sigma,
            r_core=2.0,
            r_cut=50.0,
            redshift_object=0.5,
            redshift_source=1.0,
        ),
    )
    member = al.Galaxy(
        redshift=0.5,
        mass=al.mp.dPIEMassSph(
            centre=MEMBER_CENTRE,
            sigma=60.0,
            r_core=0.0,
            r_cut=2.0,
            redshift_object=0.5,
            redshift_source=1.0,
        ),
    )
    return al.Tracer(galaxies=[host, member, al.Galaxy(redshift=1.0)])


def _isothermal_tracer(einstein_radius=1.0):
    return al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.0, 0.0),
                    ell_comps=(0.1, 0.0),
                    einstein_radius=einstein_radius,
                ),
            ),
            al.Galaxy(redshift=1.0),
        ]
    )


def _solver(use_jax=False, **kwargs):
    """A solver over the fixture grid; ``residual_threshold`` is only passed when given."""
    return al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.1),
        pixel_scale_precision=PRECISION,
        use_jax=use_jax,
        **kwargs,
    )


def _residuals(tracer, positions, source):
    traced = np.asarray(
        tracer.traced_grid_2d_list_from(grid=al.Grid2DIrregular(positions))[-1]
    )
    return np.linalg.norm(traced - np.asarray(source), axis=1)


def _near_member(positions):
    return np.linalg.norm(positions - np.asarray(MEMBER_CENTRE), axis=1) < 0.002


def _finite_sorted(positions):
    positions = np.asarray(positions)
    positions = positions[np.all(np.isfinite(positions), axis=1)]
    return positions[np.lexsort((positions[:, 1], positions[:, 0]))]


def test__disabled_filter_returns_a_candidate_on_the_zero_core_centre():
    """The baseline the filter exists for: with it disabled the solver keeps the spurious point."""
    tracer = _core_tracer()

    positions = np.asarray(
        _solver(residual_threshold=None)
        .solve(tracer, source_plane_coordinate=SOURCE)
        .array
    )

    assert positions.shape == (3, 2)
    near = _near_member(positions)
    assert near.sum() == 1
    residuals = _residuals(tracer, positions, SOURCE)
    assert residuals[near][0] > 0.05
    assert (residuals[~near] < PRECISION).all()


def test__default_solver_removes_the_spurious_candidate_only():
    tracer = _core_tracer()

    unfiltered = np.asarray(
        _solver(residual_threshold=None)
        .solve(tracer, source_plane_coordinate=SOURCE)
        .array
    )
    default = np.asarray(_solver().solve(tracer, source_plane_coordinate=SOURCE).array)
    explicit = np.asarray(
        _solver(residual_threshold=10.0)
        .solve(tracer, source_plane_coordinate=SOURCE)
        .array
    )

    assert default.shape == (2, 2)
    assert not _near_member(default).any()
    np.testing.assert_array_equal(default, unfiltered[~_near_member(unfiltered)])
    np.testing.assert_array_equal(explicit, default)


@pytest.mark.parametrize("source", [(0.05, 0.02), (0.0, 0.0), (0.3, -0.2)])
def test__residual_threshold_leaves_a_normal_lens_unchanged(source):
    tracer = _isothermal_tracer()

    unfiltered = np.asarray(
        _solver(residual_threshold=None)
        .solve(tracer, source_plane_coordinate=source)
        .array
    )
    default = np.asarray(_solver().solve(tracer, source_plane_coordinate=source).array)

    assert len(unfiltered) > 0
    np.testing.assert_array_equal(default, unfiltered)


def test__residual_threshold_defaults_to_10_and_threads_through_constructors():
    grid = al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=1.0)

    assert (
        al.PointSolver.for_grid(
            grid=grid, pixel_scale_precision=0.01
        ).residual_threshold
        == 10.0
    )
    assert (
        al.PointSolver.for_limits_and_scale(
            pixel_scale_precision=0.01
        ).residual_threshold
        == 10.0
    )
    assert (
        al.PointSolver(
            y_min=-1.0,
            y_max=1.0,
            x_min=-1.0,
            x_max=1.0,
            scale=0.1,
            pixel_scale_precision=0.01,
        ).residual_threshold
        == 10.0
    )
    assert (
        al.PointSolver.for_grid(
            grid=grid, pixel_scale_precision=0.01, residual_threshold=None
        ).residual_threshold
        is None
    )
    assert (
        al.PointSolver.for_limits_and_scale(
            pixel_scale_precision=0.01, residual_threshold=5.0
        ).residual_threshold
        == 5.0
    )


def test__tree_flatten_roundtrips_residual_threshold():
    for residual_threshold, kwargs in (
        (10.0, {}),
        (None, {"residual_threshold": None}),
    ):
        solver = _solver(**kwargs)

        children, aux = solver.tree_flatten()
        rebuilt = al.PointSolver.tree_unflatten(aux, children)

        assert rebuilt.residual_threshold == residual_threshold
        assert rebuilt.magnification_threshold == solver.magnification_threshold


if importlib.util.find_spec("jax") is not None:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from autolens.jax.registration import register_tracer_classes

    def _jit_solve(solver, tracer_from, source, *args):
        return np.asarray(
            jax.jit(
                lambda s, *a: solver.solve(
                    tracer_from(*a), source_plane_coordinate=(s[0], s[1])
                ).array
            )(jnp.array(source), *args)
        )

    def _point_tracer(einstein_radius):
        lens = al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=(0.0, 0.0),
                einstein_radius=einstein_radius,
                ell_comps=(0.05, 0.03),
            ),
        )
        return al.Tracer(
            galaxies=[
                lens,
                al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.07, 0.07))),
            ]
        )

    def _log_likelihood_and_grad(solver):
        positions = al.Grid2DIrregular([(1.53, 0.35), (-1.24, -0.28)])
        noise_map = al.ArrayIrregular([0.05, 0.05])

        def log_likelihood(einstein_radius):
            fit = al.FitPositionsImagePairAll(
                name="point_0",
                data=positions,
                noise_map=noise_map,
                tracer=_point_tracer(einstein_radius),
                solver=solver,
                xp=jnp,
            )
            return fit.log_likelihood

        value, grad = jax.jit(jax.value_and_grad(log_likelihood))(1.6)
        return float(value), float(grad)


requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (the [optional] extras)",
)


@requires_jax
def test__jit_default_residual_threshold_removes_the_spurious_candidate():
    register_tracer_classes(_core_tracer())

    unfiltered = _jit_solve(
        _solver(residual_threshold=None, use_jax=True), _core_tracer, SOURCE
    )
    filtered = _jit_solve(_solver(use_jax=True), _core_tracer, SOURCE)

    assert unfiltered.shape == filtered.shape
    assert _near_member(_finite_sorted(unfiltered)).sum() == 1

    numpy = np.asarray(
        _solver().solve(_core_tracer(), source_plane_coordinate=SOURCE).array
    )
    np.testing.assert_allclose(
        _finite_sorted(filtered), _finite_sorted(numpy), rtol=0.0, atol=1e-10
    )


@requires_jax
def test__vmap_matches_scalar_solves_with_residual_threshold():
    solver = _solver(residual_threshold=10.0, use_jax=True)
    register_tracer_classes(_core_tracer())

    sources = jnp.array([SOURCE, (0.0034, -0.011), (-0.0165, -0.031)])

    batched = np.asarray(
        jax.jit(
            jax.vmap(
                lambda s: solver.solve(
                    _core_tracer(), source_plane_coordinate=(s[0], s[1])
                ).array
            )
        )(sources)
    )

    for i, source in enumerate(np.asarray(sources)):
        scalar = _finite_sorted(_jit_solve(solver, _core_tracer, tuple(source)))
        assert not _near_member(scalar).any()
        np.testing.assert_allclose(
            _finite_sorted(batched[i]), scalar, rtol=0.0, atol=1e-10
        )


@requires_jax
def test__grad_through_the_solver_is_unchanged_by_residual_threshold():
    """
    The filter runs in the ``custom_jvp`` primal only, so at true images the implicit
    fixed-point gradient is untouched.
    """
    register_tracer_classes(_point_tracer(1.6))

    grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2)

    def solver(residual_threshold):
        return al.PointSolver.for_grid(
            grid=grid,
            pixel_scale_precision=PRECISION,
            residual_threshold=residual_threshold,
            use_jax=True,
        )

    value, grad = _log_likelihood_and_grad(solver(None))
    value_filtered, grad_filtered = _log_likelihood_and_grad(solver(10.0))

    assert np.isfinite(value) and np.isfinite(grad)
    assert grad != 0.0, "jax.grad through the solver is silently zero"
    assert value_filtered == value
    assert grad_filtered == grad


@requires_jax
def test__grad_at_the_true_images_of_the_core_fixture_is_finite():
    solver = _solver(residual_threshold=10.0, use_jax=True)
    register_tracer_classes(_core_tracer())

    def summed_positions(host_sigma):
        positions = solver.solve(
            _core_tracer(host_sigma), source_plane_coordinate=SOURCE
        ).array
        finite = jnp.all(jnp.isfinite(positions), axis=1)
        return jnp.sum(jnp.where(finite[:, None], positions, 0.0))

    value, grad = jax.jit(jax.value_and_grad(summed_positions))(480.0)

    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0

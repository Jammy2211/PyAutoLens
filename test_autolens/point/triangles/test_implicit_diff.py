"""
NumPy-only tests of the implicit-diff tangent rule (`autolens.point.solver.implicit_diff`).

The JAX behaviour (custom_jvp wiring, value_and_grad through a solver-chained likelihood,
finite-difference certification) is validated by the workspace_test scripts
(`scripts/point_source/jax_grad/gradient.py`) per the no-JAX-in-unit-tests rule; these
tests pin the pure linear algebra and the numpy-path invariants.
"""

import numpy as np
import pytest

import autogalaxy as ag
from autolens import PointSolver, Tracer
from autolens.point.solver import implicit_diff


def test_implicit_tangents_solve_the_linear_system():
    rng = np.random.default_rng(3)
    n = 5
    jac_alpha = 0.3 * rng.standard_normal((n, 2, 2))
    dalpha = rng.standard_normal((n, 2))
    dbeta = rng.standard_normal(2)
    finite = np.array([True, True, False, True, False])

    dtheta = implicit_diff.implicit_tangents_from(
        jac_alpha=jac_alpha, dalpha=dalpha, dbeta=dbeta, finite=finite, xp=np
    )

    a_mat = np.eye(2)[None] - jac_alpha
    for i in range(n):
        if finite[i]:
            np.testing.assert_allclose(
                a_mat[i] @ dtheta[i], dalpha[i] + dbeta, rtol=1e-12
            )
        else:
            np.testing.assert_array_equal(dtheta[i], 0.0)


def test_implicit_tangents_padded_nan_rows_never_reach_the_solve():
    """
    Padded rows carry whatever the Jacobian evaluated at their placeholder position
    produced — NaN when that position sits on a profile centre. The rule must
    sanitize those rows before the solve (identity ``a_mat``, zero ``rhs``): in
    reverse mode the transpose solves against ``a_mat`` row-by-row and sums the
    cotangents, so a NaN padded row would contaminate every parameter's gradient
    even though the forward output masks it (#678 phase B, cluster cells).
    """
    jac_alpha = np.array(
        [
            [[0.5, 0.0], [0.0, 0.5]],
            [[np.nan, np.nan], [np.nan, np.nan]],
        ]
    )
    dalpha = np.array([[1.0, 2.0], [np.nan, np.nan]])
    dbeta = np.array([0.1, -0.2])
    finite = np.array([True, False])

    dtheta = implicit_diff.implicit_tangents_from(
        jac_alpha=jac_alpha, dalpha=dalpha, dbeta=dbeta, finite=finite, xp=np
    )

    assert np.isfinite(dtheta).all()
    np.testing.assert_allclose(dtheta[0], (dalpha[0] + dbeta) / 0.5, rtol=1e-12)
    np.testing.assert_array_equal(dtheta[1], 0.0)


def test_implicit_tangents_near_critical_diverge_unclamped():
    # det(I - J) -> 0: the tangent must diverge with the true solve, never be clamped.
    eps = 1e-12
    jac_alpha = np.array([[[1.0 - eps, 0.0], [0.0, 0.0]]])
    dalpha = np.array([[1.0, 0.0]])
    dbeta = np.zeros(2)
    finite = np.array([True])

    dtheta = implicit_diff.implicit_tangents_from(
        jac_alpha=jac_alpha, dalpha=dalpha, dbeta=dbeta, finite=finite, xp=np
    )
    # rtol reflects float64 conditioning of the near-singular solve, not the rule
    np.testing.assert_allclose(dtheta[0, 0], 1.0 / eps, rtol=1e-4)


def test_isothermal_sph_on_axis_rule_is_analytic():
    """
    For a spherical isothermal lens the on-axis images of a source at radius b sit at
    theta = +(theta_E + b) and -(theta_E - b): d theta / d theta_E is exactly +1 and -1.
    The rule must reproduce this from the deflection field alone.
    """
    theta_e = 1.3
    b = 0.07
    mass = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=theta_e)

    theta = np.array([[0.0, theta_e + b], [0.0, -(theta_e - b)]])

    lens_calc = ag.LensCalc.from_mass_obj(mass_obj=mass)
    jacobian = lens_calc.jacobian_from(grid=theta)
    # jacobian_from returns A = I - d alpha / d theta in (x, y) ordering; convert to the
    # rule's d alpha / d theta in (y, x) ordering (the solved.py re-index precedent).
    a_xx, a_xy, a_yx, a_yy = (
        np.asarray(jacobian[i][j]) for i in (0, 1) for j in (0, 1)
    )
    jac_alpha = np.stack(
        [
            np.stack([1.0 - a_yy, -a_yx], axis=-1),
            np.stack([-a_xy, 1.0 - a_xx], axis=-1),
        ],
        axis=-2,
    )

    # isothermal: alpha = theta_E * unit(theta), so d alpha / d theta_E = alpha / theta_E
    alpha = np.asarray(mass.deflections_yx_2d_from(grid=ag.Grid2DIrregular(theta)))
    dalpha = alpha / theta_e

    dtheta = implicit_diff.implicit_tangents_from(
        jac_alpha=jac_alpha,
        dalpha=dalpha,
        dbeta=np.zeros(2),
        finite=np.array([True, True]),
        xp=np,
    )

    np.testing.assert_allclose(dtheta[0], [0.0, 1.0], atol=1e-4)
    np.testing.assert_allclose(dtheta[1], [0.0, -1.0], atol=1e-4)


def test_numpy_solve_path_never_touches_implicit_diff(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("implicit_diff must not be reached on the numpy path")

    monkeypatch.setattr(implicit_diff, "solve_padded_factory", _boom)

    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.1)
    solver = PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.01, magnification_threshold=0.1
    )
    tracer = Tracer(
        galaxies=[
            ag.Galaxy(
                redshift=0.5,
                mass=ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            ag.Galaxy(redshift=1.0),
        ]
    )
    result = solver.solve(tracer=tracer, source_plane_coordinate=(0.03, 0.03))
    assert len(np.asarray(result.array)) > 0

"""
Implicit differentiation of the ``PointSolver`` triangle-tiling solve.

The solver's refinement iteration is not differentiable: triangle containment is a hard
boolean test, so autodiff through the solve is identically zero (this was the correct
verdict of the 2026-07 gradient audit). The solved image positions themselves, however,
are smooth functions of the lens parameters between image-count events — each solved
image ``θᵢ`` satisfies the lens equation ``β = θᵢ − α(θᵢ, p)`` at fixed source-plane
position ``β``, so the inverse function theorem gives the exact tangent

    ``Aᵢ dθᵢ = dα|_{θᵢ} + dβ``   with   ``Aᵢ = ∂β/∂θ|_{θᵢ} = I − ∂α/∂θ|_{θᵢ}``

This is the same mechanism as gravity.jl (Lombardi 2024, arXiv:2406.15280, Eq. 30):
differentiate *at* the solution, never *through* the solver iteration. It is applied here
as a ``jax.custom_jvp`` whose rule is linear in the tangents, so JAX transposes it
automatically and reverse mode (``value_and_grad``) works.

Differentiability contract (mirrors the frozen-Delaunay-tables contract):

- **Between image-count events** the rule is exact up to the solver's residual: the
  forward positions are quantized at ``pixel_scale_precision``, so the computed
  likelihood is a staircase whose plateaus straddle the smooth exact-solve envelope.
  The implicit gradient is the derivative of that envelope — the quantity a gradient
  search wants. Finite differences below the stair width read exactly zero; certify
  against a fine-precision solver with a per-parameter FD step sweep
  (``autolens_workspace_test/scripts/point_source/jax_grad/gradient.py``).
- **At image-count events** (caustic crossings, magnification-threshold flips, triangle
  containment flips) the likelihood itself is discontinuous and no method has a
  gradient — the same measure-zero seam class as Delaunay re-wiring events; a sampler
  almost surely never lands on one.
- **Near critical curves** ``det(Aᵢ) → 0`` and the tangent legitimately diverges. No
  clamping or regularization is applied — the instability is surfaced (large or
  non-finite gradients), never silenced.

Padded rows (the ``inf`` sentinels of the fixed ``MAX_CONTAINING_SIZE`` output) are
constants of the output shape; their tangent is forced to zero so they cannot inject
NaNs into the batch.

Known limitation — free cosmology parameters: ``Tracer`` is registered with
``cosmology`` as ``no_flatten`` aux, so a cosmology carrying traced parameters (a free
``H0``) crosses the ``custom_jvp`` boundary as a stale tracer and raises
``UnexpectedTracerError``. Positions-only 2-plane fits lose nothing (image positions do
not depend on H0); multi-plane gradient fits with free cosmology require flattening the
cosmology into the Tracer pytree — recorded as a follow-up on PyAutoLens#657.

JAX is imported inside the factory, never at module level, per the workspace JAX rules.
"""

from typing import Tuple

import autoarray as aa


def implicit_tangents_from(jac_alpha, dalpha, dbeta, finite, xp):
    """
    The pure tangent linear algebra of the implicit rule, shared by the JAX JVP rule and
    the NumPy unit tests.

    Solves ``(I − ∂α/∂θ) dθ = dα + dβ`` row-by-row and zeroes the tangent of padded
    (non-finite) rows.

    Parameters
    ----------
    jac_alpha
        The per-image Jacobian ``∂α/∂θ`` of shape ``(n, 2, 2)`` evaluated at the solved
        positions.
    dalpha
        The tangent of the deflection field at the (fixed) solved positions, shape
        ``(n, 2)``.
    dbeta
        The tangent of the source-plane coordinate, shape ``(2,)``.
    finite
        Boolean mask of shape ``(n,)`` — True for real solved images, False for padded
        sentinel rows.
    xp
        The array module (``numpy`` or ``jax.numpy``).

    Returns
    -------
    The tangent of the solved positions, shape ``(n, 2)``; zero on padded rows. Near a
    critical curve ``det(I − ∂α/∂θ) → 0`` and the returned tangent diverges — this is
    deliberate (see the module docstring).
    """
    identity = xp.eye(2)
    a_mat = identity[None, :, :] - jac_alpha
    rhs = dalpha + dbeta[None, :]
    dtheta = xp.linalg.solve(a_mat, rhs[..., None])[..., 0]
    return xp.where(finite[:, None], dtheta, 0.0)


def tracer_is_jax_compatible(tracer) -> bool:
    """
    Whether ``tracer`` can cross a ``jax.custom_jvp`` boundary: it (and everything it
    holds) must flatten to genuine JAX-value leaves.

    True on the production model-fit path, where ``autofit.jax.register_model`` has
    registered the model classes and ``AnalysisPoint`` has registered ``Tracer`` — the
    leaves are arrays, scalars or JAX tracers. False for a hand-built ``Tracer`` whose
    ``Galaxy``/profile objects are unregistered (e.g. simulator scripts calling
    ``PointSolver.solve`` directly with ``use_jax=True``); the solve then falls back to
    the plain forward path, whose behaviour (including its zero gradient) is unchanged.
    """
    import numbers

    import numpy as np
    from jax import tree_util
    import jax

    for leaf in tree_util.tree_leaves(tracer):
        if isinstance(leaf, (jax.Array, jax.core.Tracer, np.ndarray, np.generic)):
            continue
        if isinstance(leaf, numbers.Number):
            continue
        return False
    return True


def solve_padded_factory(solver, plane_redshift, plane_index: int, xp):
    """
    Build the ``jax.custom_jvp``-wrapped padded solve for one ``(solver, plane)`` pair.

    The solver instance and plane identifiers are closed over rather than passed as
    arguments: ``PointSolver`` is not a registered pytree (the fit-class pytree
    registrations deliberately carry it as ``no_flatten`` aux data), so it cannot cross
    a ``custom_jvp`` boundary as a JAX value. The returned function takes only the two
    differentiable inputs.

    Parameters
    ----------
    solver
        The ``PointSolver`` whose forward solve is wrapped.
    plane_redshift
        Forwarded verbatim to the forward solve (``None`` for the source plane).
    plane_index
        The plane index the deflections in the tangent rule are computed between
        (``plane_i=0`` and this), derived from ``plane_redshift`` exactly as
        ``AbstractSolver._plane_grid`` derives it — ``-1`` when ``plane_redshift`` is
        ``None``.
    xp
        The JAX array module (``jax.numpy``); the factory is only reached on the JAX
        path.

    Returns
    -------
    A function ``(tracer, beta) -> (MAX_CONTAINING_SIZE, 2) array`` whose JVP applies
    the implicit fixed-point rule.
    """
    import jax

    def deflections_from(positions_array, tracer):
        grid = aa.Grid2DIrregular(values=positions_array, xp=xp)
        deflections = tracer.deflections_between_planes_from(
            grid=grid, plane_i=0, plane_j=plane_index, xp=xp
        )
        return deflections.array if hasattr(deflections, "array") else deflections

    @jax.custom_jvp
    def solve_padded(tracer, beta: Tuple[float, float]):
        return solver._solve_array(
            tracer=tracer,
            source_plane_coordinate=(beta[0], beta[1]),
            xp=xp,
            plane_redshift=plane_redshift,
            remove_infinities=False,
        )

    @solve_padded.defjvp
    def solve_padded_jvp(primals, tangents):
        tracer, beta = primals
        tracer_dot, beta_dot = tangents

        theta = solve_padded(tracer, beta)
        finite = xp.isfinite(theta).all(axis=1)
        theta_safe = xp.where(finite[:, None], theta, 0.0)

        def deflections_single(position, tracer_):
            return deflections_from(position[None, :], tracer_)[0]

        jac_alpha = jax.vmap(
            lambda position: jax.jacfwd(deflections_single, argnums=0)(position, tracer)
        )(theta_safe)

        _, dalpha = jax.jvp(
            lambda tracer_: deflections_from(theta_safe, tracer_),
            (tracer,),
            (tracer_dot,),
        )

        dtheta = implicit_tangents_from(
            jac_alpha=jac_alpha,
            dalpha=dalpha,
            dbeta=xp.asarray(beta_dot),
            finite=finite,
            xp=xp,
        )
        return theta, dtheta

    return solve_padded

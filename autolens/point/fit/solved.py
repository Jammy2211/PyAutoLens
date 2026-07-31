"""
Shared machinery for the analytically-solved point-source fit variants.

Rather than sampling the source-plane centre of a point source as a free model parameter
(a `centre` on `ag.ps.Point` / `ag.ps.PointFlux`), the `*Solved` fit classes throughout
`autolens.point.fit` solve for it analytically given the current tracer, using a
zero-parameter `ag.ps.PointSolved` profile. This module provides the shared linear algebra
and the `SolvedCentre` mixin that every solved-centre fit class uses.

**Source-plane solved likelihood** (`FitPositionsSourceSolved`, `positions/source/separations.py`)
follows Lombardi 2024 (arXiv:2406.15280) §5.1: the lens equation is Taylor-expanded around
each observed image position, giving a source-plane centre that is linear in the data and can
be solved in closed form, with the source-plane likelihood analytically marginalized over
that centre.

**Solved image-plane variants** (`FitPositionsImagePairAllSolved`, `FitPositionsImagePairRepeatSolved`)
reuse the same analytically-solved centre to drive the existing `PointSolver` forward solve,
but the image-plane pairing chi-squareds themselves are not part of the paper. This is an
extension in the spirit of glafic's source-position optimization, not a result from the paper,
and should not be attributed to it.

Convention
----------
`A ≡ ∂β/∂θ` is the standard lensing Jacobian of the lens equation (mapping image-plane
displacements to source-plane displacements), so `A⁻¹` is the magnification tensor and the
scalar magnification is `µ = 1/det(A)`. For an observed image position `θ̂ᵢ` with scalar
position noise `σᵢ` (image-plane precision `Θᵢ = σᵢ⁻² I₂`), the per-image precision on the
back-traced source-plane position is:

    `Wᵢ = Aᵢ⁻ᵀ Θᵢ Aᵢ⁻¹ = σᵢ⁻² Aᵢ⁻ᵀAᵢ⁻¹`

with `Aᵢ` evaluated at the observed position `θ̂ᵢ` (not the model/solved position). The
eigenvalues of `Wᵢ` are `λ²/σᵢ²` with `λ` the *linear* stretch along each eigendirection, so
the tensor is exact in every regime: isotropically each stretch is `√µ` (giving `µᵢ/σᵢ²`),
while near a critical curve the tangential stretch is `≈ µ` (giving `µᵢ²/σᵢ²` along the arc).

The `weighting = "magnification"` class-attribute option instead uses the scalar weighting
`Wᵢ = (µᵢ²/σᵢ²) I₂` already used by `FitPositionsSource.chi_squared_map` — the traditional
Lenstool-style approximation, which coincides with the tensor only in the near-critical
tangential limit (NOT isotropically) — retained for comparisons with that convention.

The solved source-plane centre is then the precision-weighted mean of the per-image
back-traced positions `β̂ᵢ`:

    `β* = (Σᵢ Wᵢ)⁻¹ Σᵢ Wᵢ β̂ᵢ`
"""
import numpy as np
from typing import Tuple

import autoarray as aa
import autogalaxy as ag

from autolens import exc


def _as_array(x):
    """
    Returns the underlying array of `x` if `x` is one of this project's array-structure wrappers (anything
    exposing `.array`, e.g. `aa.ArrayIrregular` / `aa.Grid2DIrregular`), otherwise returns `x` unchanged. Some
    fit classes are exercised in tests with plain `numpy.ndarray` `data` / `noise_map` inputs (never wrapped),
    which this module must tolerate.
    """
    return x.array if hasattr(x, "array") else x


def _lens_calc_for(fit) -> ag.LensCalc:
    """
    Returns the `ag.LensCalc` object used to compute the Hessian / Jacobian / magnification of `fit.tracer` at
    the plane containing the point source paired to `fit`, accounting for multi-plane ray-tracing.

    Mirrors `AbstractFitPoint.magnifications_at_positions`.
    """
    use_multi_plane = len(fit.tracer.planes) > 2
    plane_j = (
        fit.tracer.extract_plane_index_of_profile(profile_name=fit.name)
        if use_multi_plane
        else -1
    )
    return ag.LensCalc.from_tracer(
        tracer=fit.tracer,
        use_multi_plane=use_multi_plane,
        plane_j=plane_j,
    )


def precision_tensor_components_from(fit, weighting: str) -> Tuple:
    """
    Returns the (w11, w12, w21, w22) components of the per-image 2x2 precision matrix `Wᵢ`, evaluated at the
    fit's observed image-plane positions (`fit.positions`), for every image.

    Each component is an array of shape `(n_positions,)`. The matrix is symmetric by construction, so
    `w12 == w21`.

    Parameters
    ----------
    fit
        The point-source fit object (any `AbstractFitPositions` subclass) providing `positions`, `tracer`,
        `name`, `noise_map` and `_xp`.
    weighting
        `"jacobian"` — the tensor weighting `Wᵢ = Aᵢ⁻ᵀΘᵢAᵢ⁻¹`, with `Aᵢ` the lensing Jacobian at the observed
        position (see module docstring).
        `"magnification"` — the traditional scalar weighting `Wᵢ = (µᵢ²/σᵢ²) I₂`, matching
        `FitPositionsSource.chi_squared_map` (the near-critical tangential limit of the tensor).
    """
    xp = fit._xp
    precision_scalar = _as_array(fit.noise_map) ** -2.0  # Θ = σ⁻² I

    if weighting == "magnification":
        mu = _as_array(fit.magnifications_at_positions)
        w = (mu**2.0) * precision_scalar
        zero = xp.zeros_like(w)
        return w, zero, zero, w

    if weighting != "jacobian":
        raise exc.PointProfileMismatchException(
            f"Unsupported weighting '{weighting}' for the source-plane position residuals. "
            f"Valid options are 'jacobian' (tensor weighting, the default of the *Solved fit classes) or "
            f"'magnification' (scalar isotropic weighting, the default of `FitPositionsSource`)."
        )

    lens_calc = _lens_calc_for(fit)
    (a_xx, a_xy), (a_yx, a_yy) = lens_calc.jacobian_from(grid=fit.positions, xp=xp)

    # `jacobian_from` returns its 2x2 list in (x, y) row/col order (`a11`/`a22` there are the xx/yy
    # components — see its docstring: `A = [[1-hessian_xx, -hessian_xy], [-hessian_yx, 1-hessian_yy]]`),
    # whereas every position/grid array in this module is (y, x) (`positions[:, 0]` is y). Re-index into
    # (y, x) row/col order before using these as a 2x2 matrix alongside (y, x)-ordered residual vectors.
    a11, a12, a21, a22 = a_yy, a_yx, a_xy, a_xx

    det_a = a11 * a22 - a12 * a21

    ainv11 = a22 / det_a
    ainv12 = -a12 / det_a
    ainv21 = -a21 / det_a
    ainv22 = a11 / det_a

    # M = Aᵢ⁻ᵀAᵢ⁻¹, symmetric positive-(semi)definite by construction.
    m11 = ainv11 * ainv11 + ainv21 * ainv21
    m12 = ainv11 * ainv12 + ainv21 * ainv22
    m22 = ainv12 * ainv12 + ainv22 * ainv22

    w11 = m11 * precision_scalar
    w12 = m12 * precision_scalar
    w22 = m22 * precision_scalar

    return w11, w12, w12, w22


class SolvedCentre:
    """
    Mixin overriding `AbstractFitPoint.source_plane_coordinate` (the single funnel every position-based fit
    class reads the source-plane centre from, `fit/abstract.py:143-153`) to return the analytically-solved
    centre `β*`, rather than reading a `centre` model parameter.

    Must be listed before the concrete fit class in the MRO, e.g. `class FooSolved(SolvedCentre, Foo)`, so this
    override takes precedence over `AbstractFitPoint.source_plane_coordinate`.

    Requires the paired profile to have **no** `centre` attribute (e.g. `ag.ps.PointSolved`): if a
    `centre`-bearing profile (`ag.ps.Point` / `ag.ps.PointFlux`) is used with a `SolvedCentre` fit, its centre
    priors would be sampled by the non-linear search but silently ignored (the analytic solve does not read
    them) — a 2-parameter waste that is instead raised loudly.
    """

    weighting = "jacobian"

    #: Name of the corresponding free-centre fit class, used in the error message when a `centre`-bearing
    #: profile is paired to this fit. Overridden per concrete `*Solved` class.
    _non_solved_alternative_name = "the corresponding non-Solved fit class"

    @property
    def _beta_hat(self) -> aa.Grid2DIrregular:
        """
        The observed image-plane positions (`self.positions`) ray-traced to the source-plane, `β̂ᵢ`. Computed
        independently of `self.model_data` (which, for the image-plane pairing fit classes this mixin is also
        used with, means the forward-solved model image positions instead), via the same path as
        `FitPositionsSource.model_data`.
        """
        positions = self.positions
        if not hasattr(positions, "grid_2d_via_deflection_grid_from"):
            # Some fit classes are exercised in tests with a plain `numpy.ndarray` `data`
            # (never wrapped): promote to a `Grid2DIrregular` so the deflection / ray-tracing
            # calls below (which expect `.array`) have something to call it on.
            positions = aa.Grid2DIrregular(values=_as_array(positions), xp=self._xp)

        if len(self.tracer.planes) <= 2:
            deflections = self.tracer.deflections_yx_2d_from(
                grid=positions, xp=self._xp
            )
        else:
            deflections = self.tracer.deflections_between_planes_from(
                grid=positions, xp=self._xp, plane_i=0, plane_j=self.plane_index
            )

        return positions.grid_2d_via_deflection_grid_from(
            deflection_grid=deflections, xp=self._xp
        )

    @property
    def source_plane_coordinate(self) -> Tuple[float, float]:
        """
        Returns the analytically-solved source-plane centre:

            `β* = (Σᵢ Wᵢ)⁻¹ Σᵢ Wᵢ β̂ᵢ`

        with `Wᵢ` given by `precision_tensor_components_from` (using `self.weighting`) and `β̂ᵢ` given by
        `self._beta_hat`.
        """
        if hasattr(self.profile, "centre"):
            raise exc.PointProfileMismatchException(
                f"The point-source profile paired to dataset '{self.name}' "
                f"({self.profile.__class__.__name__}) has a `centre` attribute, so its free-centre priors "
                f"would be sampled by the non-linear search but silently ignored by "
                f"{self.__class__.__name__}, which solves for the source-plane centre analytically. Use a "
                f"parameter-free profile (e.g. `ag.ps.PointSolved`) with {self.__class__.__name__}, or use "
                f"{self._non_solved_alternative_name} with a `centre`-bearing profile such as `ag.ps.Point` "
                f"/ `ag.ps.PointFlux`."
            )

        xp = self._xp

        w11, w12, w21, w22 = precision_tensor_components_from(self, self.weighting)

        beta_hat = self._beta_hat.array
        beta_hat_y = beta_hat[:, 0]
        beta_hat_x = beta_hat[:, 1]

        sum_w11 = xp.sum(w11)
        sum_w12 = xp.sum(w12)
        sum_w21 = xp.sum(w21)
        sum_w22 = xp.sum(w22)

        rhs_y = xp.sum(w11 * beta_hat_y + w12 * beta_hat_x)
        rhs_x = xp.sum(w21 * beta_hat_y + w22 * beta_hat_x)

        det_sum_w = sum_w11 * sum_w22 - sum_w12 * sum_w21

        beta_star_y = (sum_w22 * rhs_y - sum_w12 * rhs_x) / det_sum_w
        beta_star_x = (-sum_w21 * rhs_y + sum_w11 * rhs_x) / det_sum_w

        return beta_star_y, beta_star_x

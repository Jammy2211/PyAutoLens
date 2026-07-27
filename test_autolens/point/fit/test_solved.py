"""
Tests for the analytically-solved point-source fit variants (`autolens.point.fit.solved` and the concrete
`*Solved` fit classes it underpins).

Numpy-only, per project convention (unit tests never import jax).
"""
import numpy as np
from scipy.optimize import minimize
import pytest

import autolens as al
from autolens.point.fit.solved import precision_tensor_components_from


def _isothermal_sph_tracer(einstein_radius=1.0, profile=None):
    profile = profile or al.ps.PointSolved()
    lens = al.Galaxy(
        redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=einstein_radius)
    )
    source = al.Galaxy(redshift=1.0, point_0=profile)
    return al.Tracer(galaxies=[lens, source])


def _elliptical_isothermal_tracer(axis_ratio, angle, einstein_radius=1.0, profile=None):
    profile = profile or al.ps.PointSolved()
    mass = al.mp.Isothermal(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle),
        einstein_radius=einstein_radius,
    )
    lens = al.Galaxy(redshift=0.5, mass=mass)
    source = al.Galaxy(redshift=1.0, point_0=profile)
    return al.Tracer(galaxies=[lens, source])


def _chi_squared_at(beta_hat, weighting_components, centre):
    w11, w12, w21, w22 = weighting_components
    dy = beta_hat[:, 0] - centre[0]
    dx = beta_hat[:, 1] - centre[1]
    return np.sum(dy * (w11 * dy + w12 * dx) + dx * (w21 * dy + w22 * dx))


class TestSourcePlaneSolvedCentre:
    def test__beta_star_equals_brute_force_minimizer_of_tensor_chi_squared(self):
        # Test 1: beta* (the analytic closed-form solve) must equal a brute-force numerical
        # minimization of the tensor-weighted source-plane chi-squared over the centre, on a
        # fixed mock tracer.
        tracer = _isothermal_sph_tracer(einstein_radius=1.0)

        positions = al.Grid2DIrregular(
            [(0.0, 1.5), (0.0, -1.3), (1.4, 0.05), (-0.9, -1.1)]
        )
        noise_map = al.ArrayIrregular([0.05, 0.05, 0.05, 0.05])

        fit = al.FitPositionsSourceSolved(
            name="point_0", data=positions, noise_map=noise_map, tracer=tracer, solver=None
        )

        beta_star = fit.source_plane_coordinate

        beta_hat = fit._beta_hat.array
        weighting_components = precision_tensor_components_from(fit, "jacobian")

        result = minimize(
            lambda centre: _chi_squared_at(beta_hat, weighting_components, centre),
            x0=np.zeros(2),
            method="Nelder-Mead",
            options={"xatol": 1.0e-10, "fatol": 1.0e-12, "maxiter": 20000},
        )

        assert beta_star[0] == pytest.approx(result.x[0], abs=1.0e-6)
        assert beta_star[1] == pytest.approx(result.x[1], abs=1.0e-6)

        # And the analytic chi-squared at beta* is (at least as good as, in practice equal to)
        # the brute-force minimum.
        assert fit.chi_squared == pytest.approx(result.fun, abs=1.0e-6)

    def test__beta_star_scalar_weighting_also_matches_brute_force_minimizer(self):
        tracer = _isothermal_sph_tracer(einstein_radius=1.2)

        positions = al.Grid2DIrregular([(0.0, 1.5), (0.0, -1.3), (1.4, 0.05)])
        noise_map = al.ArrayIrregular([0.05, 0.05, 0.05])

        fit = al.FitPositionsSourceSolved(
            name="point_0", data=positions, noise_map=noise_map, tracer=tracer, solver=None
        )
        fit.weighting = "magnification"

        beta_star = fit.source_plane_coordinate

        beta_hat = fit._beta_hat.array
        weighting_components = precision_tensor_components_from(fit, "magnification")

        result = minimize(
            lambda centre: _chi_squared_at(beta_hat, weighting_components, centre),
            x0=np.zeros(2),
            method="Nelder-Mead",
            options={"xatol": 1.0e-10, "fatol": 1.0e-12, "maxiter": 20000},
        )

        assert beta_star[0] == pytest.approx(result.x[0], abs=1.0e-6)
        assert beta_star[1] == pytest.approx(result.x[1], abs=1.0e-6)

    def test__solved_log_likelihood_matches_tensor_profiled_maximum_plus_marginalization(
        self,
    ):
        # Test 2 (source-plane, S1): the solved-centre log likelihood equals the log
        # likelihood of the *same* tensor-weighted chi-squared profiled (maximized) over a
        # free centre, plus the analytic marginalization term -- the marginalization term is
        # exactly the known correction between "point estimate at the MLE" and "analytically
        # integrated over a flat prior".
        tracer = _elliptical_isothermal_tracer(axis_ratio=0.7, angle=30.0)

        positions = al.Grid2DIrregular(
            [(0.0, 1.4), (0.0, -1.2), (1.3, 0.1), (-0.8, -1.0)]
        )
        noise_map = al.ArrayIrregular([0.03, 0.03, 0.03, 0.03])

        fit = al.FitPositionsSourceSolved(
            name="point_0", data=positions, noise_map=noise_map, tracer=tracer, solver=None
        )

        beta_hat = fit._beta_hat.array
        weighting_components = precision_tensor_components_from(fit, "jacobian")

        result = minimize(
            lambda centre: _chi_squared_at(beta_hat, weighting_components, centre),
            x0=np.zeros(2),
            method="Nelder-Mead",
            options={"xatol": 1.0e-10, "fatol": 1.0e-12, "maxiter": 20000},
        )

        profiled_max_log_likelihood = -0.5 * (result.fun + fit.noise_normalization)

        assert fit.log_likelihood == pytest.approx(
            profiled_max_log_likelihood + fit.marginalization_term, abs=1.0e-4
        )


class TestImagePlaneSolvedCentre:
    def _perfect_data_setup(self, axis_ratio=0.7, angle=30.0, true_centre=(0.05, 0.03)):
        lens = al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=(0.0, 0.0),
                ell_comps=al.convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle),
                einstein_radius=1.0,
            ),
        )
        tracer_free = al.Tracer(
            galaxies=[lens, al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=true_centre))]
        )
        tracer_solved = al.Tracer(
            galaxies=[lens, al.Galaxy(redshift=1.0, point_0=al.ps.PointSolved())]
        )

        grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)
        solver = al.PointSolver.for_grid(grid=grid, pixel_scale_precision=0.01)
        observed = solver.solve(tracer=tracer_free, source_plane_coordinate=true_centre)

        noise_map = al.ArrayIrregular([0.02] * len(observed))
        mock_solver = al.m.MockPointSolver(model_positions=observed)

        return tracer_free, tracer_solved, observed, noise_map, mock_solver, true_centre

    def test__pair_repeat_solved_log_likelihood_matches_free_centre_profiled_at_truth(
        self,
    ):
        # Test 2 (I2): with noiseless data generated at a known true centre, the solved
        # centre back-traces to (very close to) that true centre, so both the solved and a
        # free-centre fit *at* the true centre achieve the same (here: zero) chi-squared and
        # hence the same log likelihood -- the pairing chi-squared itself is untouched by the
        # mixin, so there is no marginalization offset to account for.
        (
            tracer_free,
            tracer_solved,
            observed,
            noise_map,
            mock_solver,
            true_centre,
        ) = self._perfect_data_setup()

        fit_solved = al.FitPositionsImagePairRepeatSolved(
            name="point_0",
            data=observed,
            noise_map=noise_map,
            tracer=tracer_solved,
            solver=mock_solver,
        )
        fit_free_at_truth = al.FitPositionsImagePairRepeat(
            name="point_0",
            data=observed,
            noise_map=noise_map,
            tracer=tracer_free,
            solver=mock_solver,
        )

        assert fit_solved.source_plane_coordinate[0] == pytest.approx(
            true_centre[0], abs=1.0e-3
        )
        assert fit_solved.source_plane_coordinate[1] == pytest.approx(
            true_centre[1], abs=1.0e-3
        )
        assert fit_solved.chi_squared == pytest.approx(0.0, abs=1.0e-2)
        assert fit_solved.log_likelihood == pytest.approx(
            fit_free_at_truth.log_likelihood, abs=1.0e-6
        )

    def test__pair_all_solved_log_likelihood_matches_free_centre_profiled_at_truth(self):
        # Test 2 (I1): as above, for the all-to-all pairing scheme.
        (
            tracer_free,
            tracer_solved,
            observed,
            noise_map,
            mock_solver,
            true_centre,
        ) = self._perfect_data_setup()

        fit_solved = al.FitPositionsImagePairAllSolved(
            name="point_0",
            data=observed,
            noise_map=noise_map,
            tracer=tracer_solved,
            solver=mock_solver,
        )
        fit_free_at_truth = al.FitPositionsImagePairAll(
            name="point_0",
            data=observed,
            noise_map=noise_map,
            tracer=tracer_free,
            solver=mock_solver,
        )

        assert fit_solved.log_likelihood == pytest.approx(
            fit_free_at_truth.log_likelihood, abs=1.0e-6
        )


class TestTensorVsScalarWeighting:
    def test__anisotropic_case__tensor_ordering_matches_image_plane__scalar_does_not(self):
        # Test 3: near a critical curve of an elliptical lens, the local precision tensor W
        # is strongly anisotropic (one eigen-direction is far more informative about the
        # source-plane centre than the other). Construct two candidate source-plane centres
        # A and B, both perturbations of the same back-traced position, chosen so that:
        #
        #   - candidate A is displaced along the *low*-precision eigen-direction (a large
        #     source-plane displacement barely moves the image),
        #   - candidate B is displaced by a *smaller* amount, but along the *high*-precision
        #     eigen-direction (a small source-plane displacement moves the image more).
        #
        # The scalar (magnification-squared) weighting is isotropic, so it ranks candidates
        # purely by source-plane Euclidean distance and prefers B (the smaller displacement).
        # The tensor weighting accounts for the anisotropy and prefers A. Ground truth is the
        # real (nonlinear) image-plane position each candidate centre ray-traces to: A must be
        # the genuinely better candidate (smaller true image-plane residual), matching the
        # tensor ordering and contradicting the scalar ordering.
        from scipy.optimize import root

        tracer = _elliptical_isothermal_tracer(axis_ratio=0.6, angle=45.0)

        observed_theta = (0.0, 0.85)
        positions = al.Grid2DIrregular([observed_theta])
        noise_map = al.ArrayIrregular([0.01])

        fit = al.FitPositionsSourceSolved(
            name="point_0", data=positions, noise_map=noise_map, tracer=tracer, solver=None
        )
        beta_hat = fit._beta_hat.array[0]

        w11, w12, w21, w22 = precision_tensor_components_from(fit, "jacobian")
        w = np.array([[w11[0], w12[0]], [w21[0], w22[0]]])
        eigvals, eigvecs = np.linalg.eigh(w)
        low_vec = eigvecs[:, 0]
        high_vec = eigvecs[:, 1]

        delta_a = 0.01 * low_vec
        delta_b = 0.001 * high_vec

        def chi_squared(delta, weighting):
            components = precision_tensor_components_from(fit, weighting)
            a, b, c, d = (float(x[0]) for x in components)
            dy, dx = delta
            return dy * (a * dy + b * dx) + dx * (c * dy + d * dx)

        tensor_a = chi_squared(delta_a, "jacobian")
        tensor_b = chi_squared(delta_b, "jacobian")
        scalar_a = chi_squared(delta_a, "magnification")
        scalar_b = chi_squared(delta_b, "magnification")

        # The two weightings disagree:
        assert tensor_a < tensor_b  # tensor prefers candidate A
        assert scalar_a > scalar_b  # scalar prefers candidate B

        # Ground truth: precisely ray-trace (via root-finding on the exact deflection field,
        # not the coarse triangulated PointSolver) the image-plane position each candidate
        # source centre corresponds to, near the observed image.
        def beta_of_theta(theta):
            grid = al.Grid2DIrregular([tuple(theta)])
            deflections = tracer.deflections_yx_2d_from(grid=grid)
            beta = grid.grid_2d_via_deflection_grid_from(deflection_grid=deflections)
            return beta.array[0]

        def theta_for_beta(target_beta):
            solution = root(
                lambda theta: beta_of_theta(theta) - target_beta,
                x0=np.array(observed_theta),
                method="hybr",
                tol=1.0e-14,
            )
            assert solution.success
            return solution.x

        theta_a = theta_for_beta(beta_hat + delta_a)
        theta_b = theta_for_beta(beta_hat + delta_b)

        image_distance_a = np.sqrt(np.sum((theta_a - np.array(observed_theta)) ** 2))
        image_distance_b = np.sqrt(np.sum((theta_b - np.array(observed_theta)) ** 2))

        # The tensor ordering (A better than B) matches the true image-plane ordering; the
        # scalar ordering (B better than A) does not.
        assert image_distance_a < image_distance_b


class TestLoudFailures:
    def test__solved_fit_given_centre_bearing_profile__raises_naming_alternative(self):
        tracer = _isothermal_sph_tracer(profile=al.ps.Point(centre=(0.0, 0.0)))
        positions = al.Grid2DIrregular([(0.0, 1.0), (0.0, 2.0)])
        noise_map = al.ArrayIrregular([1.0, 1.0])

        fit = al.FitPositionsSourceSolved(
            name="point_0", data=positions, noise_map=noise_map, tracer=tracer, solver=None
        )

        with pytest.raises(al.exc.PointExtractionException, match="FitPositionsSource"):
            fit.source_plane_coordinate

    def test__centre_requiring_fit_given_solved_profile__raises_naming_alternative(self):
        tracer = _isothermal_sph_tracer(profile=al.ps.PointSolved())
        positions = al.Grid2DIrregular([(0.0, 1.0), (0.0, 2.0)])
        noise_map = al.ArrayIrregular([1.0, 1.0])

        fit = al.FitPositionsSource(
            name="point_0", data=positions, noise_map=noise_map, tracer=tracer, solver=None
        )

        with pytest.raises(al.exc.PointExtractionException, match="PointSolved"):
            fit.source_plane_coordinate

    def test__pair_repeat_solved_given_centre_bearing_profile__raises(self):
        tracer = _isothermal_sph_tracer(profile=al.ps.Point(centre=(0.0, 0.0)))
        positions = al.Grid2DIrregular([(0.0, 1.0), (0.0, 2.0)])
        noise_map = al.ArrayIrregular([1.0, 1.0])
        solver = al.m.MockPointSolver(model_positions=positions)

        fit = al.FitPositionsImagePairRepeatSolved(
            name="point_0",
            data=positions,
            noise_map=noise_map,
            tracer=tracer,
            solver=solver,
        )

        with pytest.raises(
            al.exc.PointExtractionException, match="FitPositionsImagePairRepeat"
        ):
            fit.source_plane_coordinate

    def test__pair_all_solved_given_centre_bearing_profile__raises(self):
        tracer = _isothermal_sph_tracer(profile=al.ps.Point(centre=(0.0, 0.0)))
        positions = al.Grid2DIrregular([(0.0, 1.0), (0.0, 2.0)])
        noise_map = al.ArrayIrregular([1.0, 1.0])
        solver = al.m.MockPointSolver(model_positions=positions)

        fit = al.FitPositionsImagePairAllSolved(
            name="point_0",
            data=positions,
            noise_map=noise_map,
            tracer=tracer,
            solver=solver,
        )

        with pytest.raises(
            al.exc.PointExtractionException, match="FitPositionsImagePairAll"
        ):
            fit.source_plane_coordinate

    def test__point_flux_given_solved_fit__raises(self):
        # PointFlux also has a `centre`, so is just as invalid a pairing for a *Solved fit
        # as plain Point (the check is on `centre`, not on the specific profile subclass).
        tracer = _isothermal_sph_tracer(
            profile=al.ps.PointFlux(centre=(0.0, 0.0), flux=1.0)
        )
        positions = al.Grid2DIrregular([(0.0, 1.0), (0.0, 2.0)])
        noise_map = al.ArrayIrregular([1.0, 1.0])

        fit = al.FitPositionsSourceSolved(
            name="point_0", data=positions, noise_map=noise_map, tracer=tracer, solver=None
        )

        with pytest.raises(al.exc.PointExtractionException):
            fit.source_plane_coordinate

    def test__point_given_fit_fluxes__raises(self):
        # `Point` has no `flux` attribute, so the (pre-existing) free-flux `FitFluxes`
        # cannot fit it -- completing the matrix column for `FitFluxes`.
        galaxy_point_source = al.Galaxy(
            redshift=1.0, point_0=al.ps.Point(centre=(0.1, 0.1))
        )
        tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source])

        data = al.ArrayIrregular([1.0, 2.0])
        noise_map = al.ArrayIrregular([3.0, 1.0])
        positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

        with pytest.raises(al.exc.PointExtractionException):
            al.FitFluxes(
                name="point_0",
                data=data,
                noise_map=noise_map,
                positions=positions,
                tracer=tracer,
            )

    def test__point_solved_given_fit_fluxes__raises(self):
        # `PointSolved` also has no `flux` attribute, so the free-flux `FitFluxes` cannot
        # fit it either.
        galaxy_point_source = al.Galaxy(redshift=1.0, point_0=al.ps.PointSolved())
        tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source])

        data = al.ArrayIrregular([1.0, 2.0])
        noise_map = al.ArrayIrregular([3.0, 1.0])
        positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

        with pytest.raises(al.exc.PointExtractionException):
            al.FitFluxes(
                name="point_0",
                data=data,
                noise_map=noise_map,
                positions=positions,
                tracer=tracer,
            )

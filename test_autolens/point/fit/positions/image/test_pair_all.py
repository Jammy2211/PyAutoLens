import numpy as np
import pytest

import autolens as al


point = al.ps.Point(centre=(0.1, 0.1))
galaxy = al.Galaxy(redshift=1.0, point_0=point)
tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy])


@pytest.fixture
def data():
    return np.array([(0.0, 0.0), (1.0, 0.0)])


@pytest.fixture
def noise_map():
    return np.array([1.0, 1.0])


@pytest.fixture
def fit(data, noise_map):
    model_positions = al.Grid2DIrregular(
        [
            (-1.0749, -1.1),
            (1.19117, 1.175),
        ]
    )

    return al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(model_positions),
    )


def test__fit_positions_image_pair_all__two_model_positions__per_permutation_likelihoods_and_chi_squared_correct(
    fit,
):
    assert np.allclose(
        fit.all_permutations_log_likelihoods(),
        [
            -1.51114426,
            -1.50631469,
        ],
    )
    assert fit.chi_squared == -2.0 * -4.40375330990644


def test__fit_positions_image_pair_all__model_has_inf_position__inf_excluded_from_permutations(
    data,
    noise_map,
):
    model_positions = al.Grid2DIrregular(
        [
            (-1.0749, -1.1),
            (1.19117, 1.175),
            (np.inf, np.inf),
        ]
    )
    fit = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(model_positions),
    )

    assert np.allclose(
        fit.all_permutations_log_likelihoods(),
        [
            -1.51114426,
            -1.50631469,
        ],
    )
    assert fit.chi_squared == -2.0 * -4.40375330990644


def test__fit_positions_image_pair_all__model_has_duplicate_position__duplicate_permutations_handled(
    data,
    noise_map,
):
    model_positions = al.Grid2DIrregular(
        [
            (-1.0749, -1.1),
            (1.19117, 1.175),
            (1.19117, 1.175),
        ]
    )
    fit = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(model_positions),
    )

    assert np.allclose(
        fit.all_permutations_log_likelihoods(),
        [-1.14237812, -0.87193683],
    )
    assert fit.chi_squared == -2.0 * -4.211539531047171


def test__fit_positions_image_pair_all__penalty_regression__n_permutations_and_monotonic_likelihood(
    data, noise_map
):
    # Test 5: n_permutations must equal n_finite_model_positions ** n_observed, and adding a
    # spurious (but finite) extra model position must strictly lower the likelihood -- an
    # extra candidate explanation dilutes each permutation's probability without improving
    # the fit to any observed position.
    two_model_positions = al.Grid2DIrregular(
        [(-1.0749, -1.1), (1.19117, 1.175)]
    )
    fit_two = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(two_model_positions),
    )

    n_non_nan = np.count_nonzero(np.isfinite(two_model_positions.array).any(axis=1))
    assert n_non_nan == 2
    n_permutations = n_non_nan ** len(data)
    assert n_permutations == 2 ** len(data)

    three_model_positions = al.Grid2DIrregular(
        [(-1.0749, -1.1), (1.19117, 1.175), (5.0, 5.0)]
    )
    fit_three = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(three_model_positions),
    )

    n_non_nan_three = np.count_nonzero(
        np.isfinite(three_model_positions.array).any(axis=1)
    )
    assert n_non_nan_three == 3
    assert n_non_nan_three ** len(data) == 3 ** len(data)

    # log_likelihood = -0.5 * chi_squared, so a strictly lower likelihood is a strictly
    # higher chi_squared.
    assert fit_three.chi_squared > fit_two.chi_squared


def test__fit_positions_image_pair_all_solved__source_plane_coordinate_feeds_solver(
    data, noise_map
):
    galaxy_solved = al.Galaxy(redshift=1.0, point_0=al.ps.PointSolved())
    tracer_solved = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy_solved])

    model_positions = al.Grid2DIrregular([(-1.0749, -1.1), (1.19117, 1.175)])

    fit = al.FitPositionsImagePairAllSolved(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer_solved,
        solver=al.mock.MockPointSolver(model_positions),
    )

    assert np.isfinite(fit.source_plane_coordinate[0])
    assert np.isfinite(fit.source_plane_coordinate[1])
    # The pairing chi-squared itself is untouched (solver is mocked, so model positions are
    # fixed regardless of the solved centre): matches the plain FitPositionsImagePairAll
    # value from the fixture-equivalent test above.
    assert fit.chi_squared == -2.0 * -4.40375330990644


def test__no_model_positions__finite_no_image_floor_matching_siblings(data, noise_map):
    """
    Regression: with no model positions, `n_permutations` is 0, so `-log(0)` is `+inf` while the
    permutation sum is `-inf` and the two combined to NaN.

    `fitness.py` converts a NaN log-likelihood into `resample_figure_of_merit`, so the model was
    silently resampled rather than scored -- exactly what the `no_image_residual` floor exists to
    prevent. `FitPositionsImagePair` and `FitPositionsImagePairRepeat` already applied that floor;
    `FitPositionsImagePairAll` did not.

    Surfaced by @rhayes777's PyAutoLens#531 once `PointSolver.solve` began returning an empty grid
    instead of raising.
    """

    no_positions = al.Grid2DIrregular(np.zeros((0, 2)))

    kwargs = dict(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(no_positions),
    )

    fit_all = al.FitPositionsImagePairAll(**kwargs)

    # `log_likelihood` is not asserted here: the module's `noise_map` fixture is a raw ndarray,
    # which `fit_dataset.noise_normalization` cannot consume. That predates this fix and is why
    # every test in this file asserts `chi_squared` only. `log_likelihood` finiteness is covered
    # end-to-end against a real solver and an `ArrayIrregular` noise-map.
    assert np.isfinite(fit_all.chi_squared)

    # the floor is on the same scale as the siblings', not merely finite
    assert fit_all.chi_squared == pytest.approx(
        al.FitPositionsImagePair(**kwargs).chi_squared
    )
    assert fit_all.chi_squared == pytest.approx(
        al.FitPositionsImagePairRepeat(**kwargs).chi_squared
    )


def test__extreme_mismatch__log_sum_exp_stays_finite(data, noise_map):
    """
    Regression: the literal `log(sum(exp(log_p)))` underflows to `log(0) = -inf` once the best
    model/observed pairing is ~38 sigma or worse (`exp` underflows below the smallest float64),
    strangling gradient flow across the exact region gradient searches traverse to find the basin.
    The max-shifted log-sum-exp must stay finite at arbitrarily large mismatch and equal the
    directly-computed shifted reduction.
    """
    model_positions = al.Grid2DIrregular([(40.0, 40.0), (50.0, 50.0)])

    fit = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(model_positions),
    )

    log_likelihoods = fit.all_permutations_log_likelihoods()

    assert np.all(np.isfinite(log_likelihoods))
    assert np.isfinite(fit.chi_squared)

    # ~56 sigma worst pairing: every log_p is far below the ~-745 underflow threshold of exp().
    for data_position, sigma, log_likelihood in zip(data, noise_map, log_likelihoods):
        log_ps = np.array(
            [
                fit.log_p(data_position, model_position, sigma)
                for model_position in model_positions.array
            ]
        )
        assert np.all(log_ps < -745.0)
        expected = log_ps.max() + np.log(np.sum(np.exp(log_ps - log_ps.max())))
        assert log_likelihood == pytest.approx(expected, rel=1.0e-12)


def test__moderate_mismatch__log_sum_exp_matches_literal_form(data, noise_map):
    """Where the literal `log(sum(exp(...)))` is finite, the shifted form must equal it."""

    model_positions = al.Grid2DIrregular([(-1.0749, -1.1), (1.19117, 1.175)])

    fit = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(model_positions),
    )

    for data_position, sigma, log_likelihood in zip(
        data, noise_map, fit.all_permutations_log_likelihoods()
    ):
        literal = np.log(
            np.sum(
                np.exp(
                    np.array(
                        [
                            fit.log_p(data_position, model_position, sigma)
                            for model_position in model_positions.array
                        ]
                    )
                )
            )
        )
        assert log_likelihood == pytest.approx(literal, rel=1.0e-14)


def test__model_positions_present__chi_squared_unchanged(data, noise_map):
    """The no-image branch must not perturb the ordinary path."""

    fit = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(
            al.Grid2DIrregular([(-1.0749, -1.1), (1.19117, 1.175)])
        ),
    )

    assert fit.chi_squared == -2.0 * -4.40375330990644

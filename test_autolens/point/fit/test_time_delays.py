import numpy as np
from scipy.optimize import minimize_scalar
import pytest

import autolens as al


def test__fit_time_delays__all_residual_quantities_computed_correctly_with_mock_tracer():
    tracer = al.m.MockTracerPoint(
        profile=al.ps.Point(),
        time_delays=al.ArrayIrregular([2.0, 2.0]),
    )

    data = al.ArrayIrregular([1.0, 2.0])
    noise_map = al.ArrayIrregular([3.0, 1.0])
    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

    fit = al.FitTimeDelays(
        name="point_0",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    assert fit.data.in_list == [1.0, 2.0]
    assert fit.noise_map.in_list == [3.0, 1.0]
    assert fit.model_time_delays.in_list == [2.0, 2.0]
    assert fit.residual_map.in_list == [0.0, 1.0]
    assert fit.normalized_residual_map.in_list == [0.0, 1.0]
    assert fit.chi_squared_map.in_list == [0.0, 1.0]
    assert fit.chi_squared == pytest.approx(1.0, 1.0e-4)
    assert fit.noise_normalization == pytest.approx(5.87297, 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-3.43648935, 1.0e-4)


def test__fit_time_delays__model_time_delays_correct_with_real_isothermal_tracer(
    gal_x1_mp,
):
    point_source = al.ps.Point(centre=(0.1, 0.1))
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
    tracer = al.Tracer(galaxies=[gal_x1_mp, galaxy_point_source])

    data = al.ArrayIrregular([1.0, 2.0])
    noise_map = al.ArrayIrregular([3.0, 1.0])
    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

    fit = al.FitTimeDelays(
        name="point_0",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    assert fit.model_time_delays.in_list[1] == pytest.approx(-573.994580905, 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-22600.81488747, 1.0e-4)


def test__fit_time_delays_solved__solved_reference_time_equals_brute_force_scan():
    tracer = al.m.MockTracerPoint(
        profile=al.ps.PointSolved(),
        time_delays=al.ArrayIrregular([10.0, 12.0, 15.0]),
    )

    positions = al.Grid2DIrregular([(0.0, 1.5), (0.0, -1.3), (1.4, 0.05)])
    data = al.ArrayIrregular([1.0, 4.0, 6.0])
    noise_map = al.ArrayIrregular([0.5, 0.5, 1.0])

    fit = al.FitTimeDelaysSolved(
        name="point_0",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    model_delays = fit.model_data.array

    def chi_squared(reference_time):
        return np.sum(
            (data.array - (model_delays + reference_time)) ** 2 / noise_map.array**2
        )

    brute_force = minimize_scalar(chi_squared)

    assert fit.solved_reference_time == pytest.approx(brute_force.x, 1.0e-6)
    assert fit.chi_squared == pytest.approx(chi_squared(fit.solved_reference_time), 1.0e-8)


def test__fit_time_delays_solved__works_with_point_flux_profile_too():
    # Time-delay fitting does not read any attribute of the paired profile (the model
    # delays come from the tracer alone), so FitTimeDelaysSolved is not restricted to
    # PointSolved -- it works with any profile that can be name-paired.
    tracer = al.m.MockTracerPoint(
        profile=al.ps.PointFlux(flux=1.0),
        time_delays=al.ArrayIrregular([2.0, 2.0]),
    )

    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    data = al.ArrayIrregular([1.0, 2.0])
    noise_map = al.ArrayIrregular([3.0, 1.0])

    fit = al.FitTimeDelaysSolved(
        name="point_0",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    assert np.isfinite(float(fit.log_likelihood))

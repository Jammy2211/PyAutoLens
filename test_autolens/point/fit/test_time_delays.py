import numpy as np
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

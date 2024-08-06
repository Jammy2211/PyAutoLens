import numpy as np
import pytest

import autolens as al


def test__two_sets_of_positions__residuals_likelihood_correct():
    point = al.ps.Point(centre=(0.1, 0.1))
    galaxy = al.Galaxy(redshift=1.0, point_0=point)
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy])

    data = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])
    model_data = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])

    solver = al.m.MockPointSolver(model_positions=model_data)

    fit = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert fit.model_data.in_list == [(3.0, 1.0), (2.0, 3.0)]
    assert fit.noise_map.in_list == [0.5, 1.0]

    assert fit.residual_map.in_list == [np.sqrt(10.0), np.sqrt(2.0)]
    assert fit.normalized_residual_map.in_list == [
        np.sqrt(10.0) / 0.5,
        np.sqrt(2.0) / 1.0,
    ]
    assert fit.chi_squared_map.in_list == [
        (np.sqrt(10.0) / 0.5) ** 2,
        np.sqrt(2.0) ** 2.0,
    ]
    assert fit.chi_squared == pytest.approx(42.0, 1.0e-4)
    assert fit.noise_normalization == pytest.approx(2.28945, 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-22.14472, 1.0e-4)


def test__three_sets_of_positions__model_is_repeat_allocated():
    point = al.ps.Point(centre=(0.1, 0.1))
    galaxy = al.Galaxy(redshift=1.0, point_0=point)
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy])

    data = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])
    model_data = al.Grid2DIrregular([(3.0, 1.0), (3.0, 4.0)])

    solver = al.m.MockPointSolver(model_positions=model_data)

    fit = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert fit.model_data.in_list == [(3.0, 1.0), (3.0, 4.0)]
    assert fit.residual_map.in_list == [np.sqrt(10.0), 0.0, 0.0]

import numpy as np
import pytest

import autolens as al


def test__three_sets_of_positions__model_is_repeated__does_not_double_count():
    point = al.ps.Point(centre=(0.1, 0.1))
    galaxy = al.Galaxy(redshift=1.0, point_0=point)
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy])

    data = al.Grid2DIrregular([(2.0, 0.0), (1.0, 0.0), (0.0, 0.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0, 2.0])
    model_data = al.Grid2DIrregular([(4.0, 0.0), (3.0, 0.0), (0.0, 0.0)])

    solver = al.m.MockPointSolver(model_positions=model_data)

    fit = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    print(fit.residual_map)

    assert fit.model_data.in_list == [(4.0, 0.0), (3.0, 0.0), (0.0, 0.0)]
    assert fit.residual_map.in_list == [2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]

    print(fit.noise_map)
    print(fit.normalized_residual_map)

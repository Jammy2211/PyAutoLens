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

    # Uses FitPositionsImagePairRepeat to produce residual_map but test is focused on all attributes

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


def test__multi_plane_position_solving():
    grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

    g0 = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.1, 0.1)))
    g2 = al.Galaxy(redshift=2.0, point_1=al.ps.Point(centre=(0.1, 0.1)))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    dataset = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])

    solver = al.PointSolver.for_grid(grid=grid, pixel_scale_precision=0.01)

    fit_0 = al.AbstractFitPositionsImagePair(
        name="point_0",
        data=dataset,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    fit_1 = al.AbstractFitPositionsImagePair(
        name="point_1",
        data=dataset,
        noise_map=noise_map,  #
        tracer=tracer,
        solver=solver,
    )

    scaling_factor = tracer.cosmology.scaling_factor_between_redshifts_from(
        redshift_0=0.5, redshift_1=1.0, redshift_final=2.0
    )

    print(fit_0.model_data)
    print(fit_1.model_data.array)

    assert fit_0.model_data[0, :] == pytest.approx(
        scaling_factor * fit_1.model_data.array[0, :], 1.0e-1
    )
    assert fit_0.model_data[0, :] == pytest.approx(
        scaling_factor * fit_1.model_data.array[0, :], 1.0e-1
    )

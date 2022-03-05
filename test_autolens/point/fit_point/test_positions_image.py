import numpy as np
import pytest

import autolens as al


def test__two_sets_of_positions__residuals_likelihood_correct():

    point_source = al.ps.Point(centre=(0.1, 0.1))
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
    tracer = al.Tracer.from_galaxies(
        galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source]
    )

    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ValuesIrregular([0.5, 1.0])
    model_positions = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])

    point_solver = al.m.MockPointSolver(model_positions=model_positions)

    fit = al.FitPositionsImage(
        name="point_0",
        positions=positions,
        noise_map=noise_map,
        tracer=tracer,
        point_solver=point_solver,
    )

    assert fit.model_positions.in_list == [(3.0, 1.0), (2.0, 3.0)]

    assert fit.model_positions.in_list == [(3.0, 1.0), (2.0, 3.0)]

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


def test__more_model_positions_than_data_positions__pairs_closest_positions():

    g0 = al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.1, 0.1)))

    tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ValuesIrregular([0.5, 1.0])
    model_positions = al.Grid2DIrregular(
        [(3.0, 1.0), (2.0, 3.0), (1.0, 0.0), (0.0, 1.0)]
    )

    point_solver = al.m.MockPointSolver(model_positions=model_positions)

    fit = al.FitPositionsImage(
        name="point_0",
        positions=positions,
        noise_map=noise_map,
        tracer=tracer,
        point_solver=point_solver,
    )

    assert fit.model_positions.in_list == [(1.0, 0.0), (2.0, 3.0)]
    assert fit.noise_map.in_list == [0.5, 1.0]
    assert fit.residual_map.in_list == [1.0, np.sqrt(2.0)]
    assert fit.normalized_residual_map.in_list == [2.0, np.sqrt(2.0) / 1.0]
    assert fit.chi_squared_map.in_list == [4.0, np.sqrt(2.0) ** 2.0]
    assert fit.chi_squared == pytest.approx(6.0, 1.0e-4)
    assert fit.noise_normalization == pytest.approx(2.289459, 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-4.144729, 1.0e-4)


def test__multi_plane_position_solving():

    grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05, sub_size=1)

    g0 = al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.1, 0.1)))
    g2 = al.Galaxy(redshift=2.0, point_1=al.ps.Point(centre=(0.1, 0.1)))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ValuesIrregular([0.5, 1.0])

    point_solver = al.PointSolver(grid=grid, pixel_scale_precision=0.01)

    fit_0 = al.FitPositionsImage(
        name="point_0",
        positions=positions,
        noise_map=noise_map,
        tracer=tracer,
        point_solver=point_solver,
    )

    fit_1 = al.FitPositionsImage(
        name="point_1",
        positions=positions,
        noise_map=noise_map,
        tracer=tracer,
        point_solver=point_solver,
    )

    scaling_factor = al.util.cosmology.scaling_factor_between_redshifts_from(
        redshift_0=0.5, redshift_1=1.0, redshift_final=2.0, cosmology=tracer.cosmology
    )

    assert fit_0.model_positions[0, 0] == pytest.approx(
        scaling_factor * fit_1.model_positions[0, 0], 1.0e-1
    )
    assert fit_0.model_positions[0, 1] == pytest.approx(
        scaling_factor * fit_1.model_positions[0, 1], 1.0e-1
    )

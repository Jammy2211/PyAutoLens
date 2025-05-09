import pytest

import autolens as al


def test__two_sets_of_positions__residuals_likelihood_correct():
    point_source = al.ps.Point(centre=(0.0, 0.0))
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source])

    positions = al.Grid2DIrregular([(0.0, 1.0), (0.0, 2.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])

    fit = al.FitPositionsSource(
        name="point_0", data=positions, noise_map=noise_map, tracer=tracer, solver=None
    )

    assert fit.model_data.in_list == [(0.0, 1.0), (0.0, 2.0)]
    assert fit.noise_map.in_list == [0.5, 1.0]
    assert fit.residual_map.in_list == [1.0, 2.0]
    assert fit.normalized_residual_map.in_list == [1.0 / 0.5, 2.0 / 1.0]
    assert fit.chi_squared_map.in_list == [(1.0 / 0.5) ** 2.0, 2.0**2.0]
    assert fit.chi_squared == pytest.approx(8.0, 1.0e-4)
    assert fit.noise_normalization == pytest.approx(2.28945, 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-5.14472988, 1.0e-4)

    # Inclusion of mass model below means there are nonzero magnifications at each position, which get factored into
    # chi-squared calculation.

    galaxy_mass = al.Galaxy(
        redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.1)
    )

    tracer = al.Tracer(galaxies=[galaxy_mass, galaxy_point_source])

    fit = al.FitPositionsSource(
        name="point_0", data=positions, noise_map=noise_map, tracer=tracer, solver=None
    )

    assert fit.magnifications_at_positions.in_list == pytest.approx(
        [1.1111049387688177, 1.0526308864400329], 1.0e-4
    )
    assert fit.model_data.in_list == [(0.0, 0.9), (0.0, 1.9)]

    assert fit.chi_squared_map.in_list == pytest.approx(
        [3.9999555592589244, 3.9999947369459807], 1.0e-4
    )
    assert fit.log_likelihood == pytest.approx(-4.98805743691215, 1.0e-4)


def test__multi_plane_position_solving():
    g0 = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.1, 0.1)))
    g2 = al.Galaxy(redshift=2.0, point_1=al.ps.Point(centre=(0.1, 0.1)))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    positions = al.Grid2DIrregular([(0.0, 1.0), (0.0, 2.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])

    traced_grids = tracer.traced_grid_2d_list_from(grid=positions)

    fit_0 = al.FitPositionsSource(
        name="point_0", data=positions, noise_map=noise_map, tracer=tracer, solver=None
    )

    assert fit_0.model_data[0, 1] == pytest.approx(0.326054, 1.0e-1)
    assert fit_0.model_data[1, 1] == pytest.approx(1.326054, 1.0e-1)

    assert (fit_0.model_data == traced_grids[1]).all()

    fit_1 = al.FitPositionsSource(
        name="point_1", data=positions, noise_map=noise_map, tracer=tracer, solver=None
    )

    assert (fit_1.model_data == traced_grids[2]).all()

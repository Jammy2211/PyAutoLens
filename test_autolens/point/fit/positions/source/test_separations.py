import numpy as np
import pytest

import autolens as al


def test__fit_positions_source__all_residual_quantities_correct_without_mass_profile():
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
    assert fit.chi_squared_map.in_list == [(1.0 / 0.5) ** 2.0, 2.0 ** 2.0]
    assert fit.chi_squared == pytest.approx(8.0, 1.0e-4)
    assert fit.noise_normalization == pytest.approx(2.28945, 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-5.14472988, 1.0e-4)


def test__fit_positions_source__with_isothermal_mass_profile__magnification_reduces_source_positions():
    # Inclusion of mass model means nonzero magnifications at each position, which
    # reduce the reconstructed source-plane separation.
    point_source = al.ps.Point(centre=(0.0, 0.0))
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
    galaxy_mass = al.Galaxy(
        redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.1)
    )

    tracer = al.Tracer(galaxies=[galaxy_mass, galaxy_point_source])

    positions = al.Grid2DIrregular([(0.0, 1.0), (0.0, 2.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])

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


def test__fit_positions_source__multi_plane_tracer__model_data_traces_to_correct_source_plane():
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


class FitPositionsSourceJacobian(al.FitPositionsSource):
    weighting = "jacobian"


def test__fit_positions_source__default_weighting_is_magnification():
    assert al.FitPositionsSource.weighting == "magnification"


def test__fit_positions_source__jacobian_weighting__matches_solved_at_solved_centre():
    """
    The free-centre tensor fit evaluated with its `centre` fixed at the solved centre `β*` must
    reproduce `FitPositionsSourceSolved`'s chi-squared and noise normalization exactly — the two
    likelihoods differ only by the solved class's analytic-marginalization term.
    """
    galaxy_mass = al.Galaxy(
        redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.1)
    )
    positions = al.Grid2DIrregular([(0.0, 1.0), (0.0, 2.0), (1.0, 0.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0, 0.8])

    tracer_solved = al.Tracer(
        galaxies=[
            galaxy_mass,
            al.Galaxy(redshift=1.0, point_0=al.ps.PointSolved()),
        ]
    )
    fit_solved = al.FitPositionsSourceSolved(
        name="point_0",
        data=positions,
        noise_map=noise_map,
        tracer=tracer_solved,
        solver=None,
    )

    beta_star = fit_solved.source_plane_coordinate

    tracer_free = al.Tracer(
        galaxies=[
            galaxy_mass,
            al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=beta_star)),
        ]
    )
    fit_free = FitPositionsSourceJacobian(
        name="point_0",
        data=positions,
        noise_map=noise_map,
        tracer=tracer_free,
        solver=None,
    )

    assert fit_free.chi_squared_map.in_list == pytest.approx(
        fit_solved.chi_squared_map.in_list, rel=1.0e-8
    )
    assert fit_free.chi_squared == pytest.approx(fit_solved.chi_squared, rel=1.0e-8)
    assert fit_free.noise_normalization == pytest.approx(
        fit_solved.noise_normalization, rel=1.0e-8
    )
    assert fit_free.log_likelihood == pytest.approx(
        fit_solved.log_likelihood - fit_solved.marginalization_term, rel=1.0e-8
    )


def test__fit_positions_source__jacobian_weighting__observed_plane_noise_normalization():
    galaxy_mass = al.Galaxy(
        redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.1)
    )
    positions = al.Grid2DIrregular([(0.0, 1.0), (0.0, 2.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])

    fit = FitPositionsSourceJacobian(
        name="point_0",
        data=positions,
        noise_map=noise_map,
        tracer=al.Tracer(
            galaxies=[
                galaxy_mass,
                al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.0, 0.0))),
            ]
        ),
        solver=None,
    )

    sigma_sq = noise_map.array**2.0
    assert fit.noise_normalization == pytest.approx(
        np.sum(np.log((2.0 * np.pi) ** 2.0 * sigma_sq**2.0)), rel=1.0e-12
    )


def test__fit_positions_source__unknown_weighting_raises():
    class FitPositionsSourceTypo(al.FitPositionsSource):
        weighting = "magnificaton"

    fit = FitPositionsSourceTypo(
        name="point_0",
        data=al.Grid2DIrregular([(0.0, 1.0)]),
        noise_map=al.ArrayIrregular([0.5]),
        tracer=al.Tracer(
            galaxies=[
                al.Galaxy(redshift=0.5),
                al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.0, 0.0))),
            ]
        ),
        solver=None,
    )

    with pytest.raises(al.exc.PointProfileMismatchException):
        fit.chi_squared_map


def test__fit_positions_source_solved__source_plane_centre_matches_no_free_centre_prior():
    point_source = al.ps.PointSolved()
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
    galaxy_mass = al.Galaxy(
        redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.1)
    )
    tracer = al.Tracer(galaxies=[galaxy_mass, galaxy_point_source])

    positions = al.Grid2DIrregular([(0.0, 1.0), (0.0, 2.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])

    fit = al.FitPositionsSourceSolved(
        name="point_0", data=positions, noise_map=noise_map, tracer=tracer, solver=None
    )

    beta_star = fit.source_plane_coordinate

    assert np.isfinite(beta_star[0])
    assert np.isfinite(beta_star[1])
    assert np.isfinite(fit.chi_squared)
    assert np.isfinite(fit.noise_normalization)
    assert np.isfinite(fit.marginalization_term)
    assert np.isfinite(float(fit.log_likelihood))

    # Default weighting is the tensor ("jacobian") weighting.
    assert fit.weighting == "jacobian"

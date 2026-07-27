import numpy as np
from scipy.optimize import minimize_scalar
import pytest

import autolens as al


def test__fit_fluxes__all_residual_quantities_computed_correctly_with_mock_tracer():
    tracer = al.m.MockTracerPoint(profile=al.ps.PointFlux(flux=2.0))

    data = al.ArrayIrregular([1.0, 2.0])
    noise_map = al.ArrayIrregular([3.0, 1.0])
    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

    fit = al.FitFluxes(
        name="point_0",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    assert fit.data.in_list == [1.0, 2.0]
    assert fit.noise_map.in_list == [3.0, 1.0]
    assert fit.model_fluxes.in_list == [2.0, 2.0]
    assert fit.residual_map.in_list == [-1.0, -0.0]
    assert fit.normalized_residual_map.in_list == [-1.0 / 3.0, -0.0]
    assert fit.chi_squared_map.in_list == [1.0 / 9.0, 0.0]
    assert fit.chi_squared == pytest.approx(1.0 / 9.0, 1.0e-4)
    assert fit.noise_normalization == pytest.approx(5.87297, 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-2.992044910633, 1.0e-4)


def test__fit_fluxes__model_flux_magnified_correctly_with_real_isothermal_tracer(
    gal_x1_mp,
):
    point_source = al.ps.PointFlux(centre=(0.1, 0.1), flux=2.0)
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
    tracer = al.Tracer(galaxies=[gal_x1_mp, galaxy_point_source])

    data = al.ArrayIrregular([1.0, 2.0])
    noise_map = al.ArrayIrregular([3.0, 1.0])
    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

    fit = al.FitFluxes(
        name="point_0",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    assert fit.model_fluxes.in_list[1] == pytest.approx(2.5, 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-3.11702, 1.0e-4)


def test__fit_fluxes_solved__solved_flux_equals_brute_force_scan():
    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0))
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=al.ps.PointSolved())
    tracer = al.Tracer(galaxies=[lens, galaxy_point_source])

    positions = al.Grid2DIrregular([(0.0, 1.5), (0.0, -1.3), (1.4, 0.05)])
    data = al.ArrayIrregular([5.0, 3.2, 4.1])
    noise_map = al.ArrayIrregular([0.3, 0.4, 0.2])

    fit = al.FitFluxesSolved(
        name="point_0",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    mu = fit.magnifications_at_positions.array

    def chi_squared(flux):
        return np.sum((data.array - mu * flux) ** 2 / noise_map.array**2)

    brute_force = minimize_scalar(chi_squared)

    assert fit.solved_flux == pytest.approx(brute_force.x, 1.0e-6)
    assert fit.model_data.in_list == pytest.approx((mu * fit.solved_flux).tolist(), 1.0e-8)


def test__fit_fluxes_solved__profile_with_flux_attribute__raises_naming_alternative():
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=al.ps.PointFlux(flux=2.0))
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source])

    data = al.ArrayIrregular([1.0, 2.0])
    noise_map = al.ArrayIrregular([3.0, 1.0])
    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

    with pytest.raises(al.exc.PointExtractionException, match="FitFluxes"):
        al.FitFluxesSolved(
            name="point_0",
            data=data,
            noise_map=noise_map,
            positions=positions,
            tracer=tracer,
        )


def test__fit_fluxes_solved__works_with_plain_point_profile_no_flux_attribute():
    galaxy_point_source = al.Galaxy(
        redshift=1.0, point_0=al.ps.Point(centre=(0.1, 0.1))
    )
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source])

    data = al.ArrayIrregular([1.0, 2.0])
    noise_map = al.ArrayIrregular([3.0, 1.0])
    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

    fit = al.FitFluxesSolved(
        name="point_0",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    assert np.isfinite(fit.solved_flux)
    assert np.isfinite(float(fit.log_likelihood))

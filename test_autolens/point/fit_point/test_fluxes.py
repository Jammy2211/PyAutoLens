from functools import partial
import pytest

import autolens as al


def test__one_set_of_fluxes__residuals_likelihood_correct():

    tracer = al.m.MockTracerPoint(
        profile=al.ps.PointFlux(flux=2.0), magnification=al.ValuesIrregular([2.0, 2.0])
    )

    fluxes = al.ValuesIrregular([1.0, 2.0])
    noise_map = al.ValuesIrregular([3.0, 1.0])
    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

    fit = al.FitFluxes(
        name="point_0",
        fluxes=fluxes,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    assert fit.fluxes.in_list == [1.0, 2.0]
    assert fit.noise_map.in_list == [3.0, 1.0]
    assert fit.model_fluxes.in_list == [4.0, 4.0]
    assert fit.residual_map.in_list == [-3.0, -2.0]
    assert fit.normalized_residual_map.in_list == [-1.0, -2.0]
    assert fit.chi_squared_map.in_list == [1.0, 4.0]
    assert fit.chi_squared == pytest.approx(5.0, 1.0e-4)
    assert fit.noise_normalization == pytest.approx(5.87297, 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-5.43648, 1.0e-4)


def test__use_real_tracer(gal_x1_mp):

    point_source = al.ps.PointFlux(centre=(0.1, 0.1), flux=2.0)
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
    tracer = al.Tracer.from_galaxies(galaxies=[gal_x1_mp, galaxy_point_source])

    fluxes = al.ValuesIrregular([1.0, 2.0])
    noise_map = al.ValuesIrregular([3.0, 1.0])
    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

    fit = al.FitFluxes(
        name="point_0",
        fluxes=fluxes,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    assert fit.model_fluxes.in_list[1] == pytest.approx(2.5, 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-3.11702, 1.0e-4)


def test__multi_plane_calculation(gal_x1_mp):

    g0 = al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=1.0, point_0=al.ps.PointFlux(flux=1.0))
    g2 = al.Galaxy(redshift=2.0, point_1=al.ps.PointFlux(flux=2.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fluxes = al.ValuesIrregular([1.0])
    noise_map = al.ValuesIrregular([3.0])
    positions = al.Grid2DIrregular([(2.0, 0.0)])

    fit_0 = al.FitFluxes(
        name="point_0",
        fluxes=fluxes,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    deflections_func = partial(
        tracer.deflections_between_planes_from, plane_i=0, plane_j=1
    )

    magnification_0 = tracer.magnification_2d_via_hessian_from(
        grid=positions, deflections_func=deflections_func
    )

    assert fit_0.magnifications[0] == magnification_0

    fit_1 = al.FitFluxes(
        name="point_1",
        fluxes=fluxes,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    deflections_func = partial(
        tracer.deflections_between_planes_from, plane_i=0, plane_j=2
    )

    magnification_1 = tracer.magnification_2d_via_hessian_from(
        grid=positions, deflections_func=deflections_func
    )

    assert fit_1.magnifications[0] == magnification_1

    assert fit_0.magnifications[0] != pytest.approx(fit_1.magnifications[0], 1.0e-1)

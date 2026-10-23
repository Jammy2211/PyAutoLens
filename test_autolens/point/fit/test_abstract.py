import numpy as np
import pytest

import autogalaxy as ag
import autolens as al


def test__magnifications_at_positions__multi_plane_calculation(gal_x1_mp):
    g0 = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=1.0, point_0=al.ps.PointFlux(flux=1.0))
    g2 = al.Galaxy(redshift=2.0, point_1=al.ps.PointFlux(flux=2.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    data = al.ArrayIrregular([1.0])
    noise_map = al.ArrayIrregular([3.0])
    positions = al.Grid2DIrregular([(2.0, 0.0)])

    fit_0 = al.FitFluxes(
        name="point_0",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    od_0 = ag.LensCalc.from_tracer(tracer, use_multi_plane=True, plane_j=1)
    magnification_0 = abs(od_0.magnification_2d_via_hessian_from(grid=positions))

    assert fit_0.magnifications_at_positions[0] == magnification_0

    fit_1 = al.FitFluxes(
        name="point_1",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    od_1 = ag.LensCalc.from_tracer(tracer, use_multi_plane=True, plane_j=2)
    magnification_1 = abs(od_1.magnification_2d_via_hessian_from(grid=positions))

    assert fit_1.magnifications_at_positions[0] == magnification_1

    assert fit_0.magnifications_at_positions[0] != pytest.approx(
        fit_1.magnifications_at_positions.array[0], 1.0e-1
    )


def _sie_tracer_and_positions():
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), einstein_radius=1.0, ell_comps=(0.0, 0.1)
        ),
    )
    source = al.Galaxy(
        redshift=1.0, point_0=al.ps.PointFlux(centre=(0.0, 0.0), flux=1.0)
    )

    tracer = al.Tracer(galaxies=[lens, source])

    # Hand-supplied image positions: outside the critical curve (positive parity, minimum) and inside it
    # (negative parity, saddle point).
    positions = al.Grid2DIrregular([(0.0, 1.8), (0.0, -0.6)])

    return tracer, positions


def _assert_mixed_parity(fit):
    signed = fit.signed_magnifications_at_positions
    unsigned = fit.magnifications_at_positions

    assert isinstance(signed, al.ArrayIrregular)
    assert isinstance(unsigned, al.ArrayIrregular)

    signed = np.asarray(signed.array)
    unsigned = np.asarray(unsigned.array)

    assert signed[0] > 0.0
    assert signed[1] < 0.0
    assert signed == pytest.approx(np.array([2.571428571, -1.2]), 1.0e-4)
    assert (unsigned == np.abs(signed)).all()


def test__signed_magnifications_at_positions__mixed_parity__numpy():
    tracer, positions = _sie_tracer_and_positions()

    fit = al.FitFluxes(
        name="point_0",
        data=al.ArrayIrregular([1.0, 1.0]),
        noise_map=al.ArrayIrregular([1.0, 1.0]),
        positions=positions,
        tracer=tracer,
    )

    _assert_mixed_parity(fit)


def test__signed_magnifications_at_positions__mixed_parity__jax():
    jnp = pytest.importorskip("jax.numpy")

    tracer, positions = _sie_tracer_and_positions()

    fit = al.FitFluxes(
        name="point_0",
        data=al.ArrayIrregular([1.0, 1.0]),
        noise_map=al.ArrayIrregular([1.0, 1.0]),
        positions=positions,
        tracer=tracer,
        xp=jnp,
    )

    assert fit.use_jax is True

    _assert_mixed_parity(fit)

    fit_np = al.FitFluxes(
        name="point_0",
        data=al.ArrayIrregular([1.0, 1.0]),
        noise_map=al.ArrayIrregular([1.0, 1.0]),
        positions=positions,
        tracer=tracer,
    )

    assert np.asarray(fit.signed_magnifications_at_positions.array) == pytest.approx(
        fit_np.signed_magnifications_at_positions.array, 1.0e-8
    )

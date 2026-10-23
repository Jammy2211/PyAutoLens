import numpy as np
import pytest

import autogalaxy as ag
import autolens as al


def test__signed_magnifications_at_positions__multi_plane__uses_point_source_plane_not_final_plane():
    """
    A three-plane tracer whose point source sits in the middle plane: the fit's magnification must be the
    multi-plane magnification up to that plane (`plane_j`), and must differ from the final-plane value, so the
    test fails if `plane_j` silently resolves to the final plane.
    """
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), einstein_radius=1.0, ell_comps=(0.0, 0.1)
        ),
    )
    middle = al.Galaxy(
        redshift=1.0,
        mass=al.mp.IsothermalSph(centre=(0.1, 0.1), einstein_radius=0.3),
        point_0=al.ps.PointFlux(centre=(0.0, 0.0), flux=1.0),
    )
    final = al.Galaxy(redshift=2.0, mass=al.mp.IsothermalSph(einstein_radius=0.1))

    tracer = al.Tracer(galaxies=[lens, middle, final])

    assert len(tracer.planes) == 3

    plane_j = tracer.extract_plane_index_of_profile(profile_name="point_0")

    assert plane_j == 1

    positions = al.Grid2DIrregular([(0.0, 1.8), (0.0, -0.6), (1.5, 0.0)])

    fit = al.FitFluxes(
        name="point_0",
        data=al.ArrayIrregular([1.0, 1.0, 1.0]),
        noise_map=al.ArrayIrregular([1.0, 1.0, 1.0]),
        positions=positions,
        tracer=tracer,
    )

    assert fit.plane_index == plane_j

    expected = ag.LensCalc.from_tracer(
        tracer, use_multi_plane=True, plane_j=plane_j
    ).magnification_2d_via_hessian_from(grid=positions)

    final_plane = ag.LensCalc.from_tracer(
        tracer, use_multi_plane=True, plane_j=-1
    ).magnification_2d_via_hessian_from(grid=positions)

    signed = np.asarray(fit.signed_magnifications_at_positions.array)

    assert signed == pytest.approx(np.asarray(expected.array), 1.0e-10)
    assert np.asarray(fit.magnifications_at_positions.array) == pytest.approx(
        np.abs(np.asarray(expected.array)), 1.0e-10
    )

    assert not np.allclose(signed, np.asarray(final_plane.array), rtol=1.0e-3)

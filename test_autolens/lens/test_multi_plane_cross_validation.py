r"""
Cross-validation of multi-plane ray tracing against independent oracles.

Single-plane lensing quantities in this project are already cross-validated two ways (e.g.
``test_autogalaxy/profiles/mass/total/test_isothermal.py::test__shear_yx_2d_from__matches_via_hessian``
compares an analytic closed form against Richardson-extrapolated derivatives of the deflections).
Multi-plane ray tracing had no such second opinion: ``tracer_util.traced_grid_2d_list_from`` and
its scaling factors were the only implementation of the recursion, and every downstream quantity
was checked against itself or against literals produced by the same code. PyAutoLens#480 survived
four months in exactly that machinery.

This module builds oracles that share no code with ``tracer_util``:

1. **Cosmology arm.** The scaling factors are recomputed from ``astropy.cosmology.Planck15``
   angular diameter distances, rather than from the hand-rolled Simpson integration in
   ``autogalaxy/cosmology/model.py``.
2. **Paper arm.** The multi-plane lens equation is written out here directly from the published
   formalism (below), using each galaxy's *single-plane* ``deflections_yx_2d_from`` and the
   astropy scaling factors.
3. **Numerical Jacobian arm.** Central differences of the map ``theta -> beta_j`` with
   ``mu = 1 / det(J)``; this is the oracle that settled PyAutoLens#480, and it is asserted to be
   stable across step sizes ``h`` from 1e-4 to 1e-7.
4. **Jacobian recursion arm.** The McCully et al. recursion for the Jacobian, propagated
   plane by plane, with the per-plane deflection gradients from central differences of the
   *profile* deflections.
5. **Degenerate reductions and closed forms.** All-mass-in-one-plane reduces to the single-plane
   analytic result; two aligned isothermal spheres on the axis have a closed-form solution and
   two Einstein rings.

The formalism
-------------

**Multi-plane lens equation** (Schneider, Ehlers & Falco, *Gravitational Lenses*, 1992, §9.1,
equations 9.6 and 9.7b). Writing ``theta_1`` for the observed image-plane position and
``alpha_i`` for the deflection of plane ``i`` evaluated at that plane's own traced position,

    theta_j = theta_1 - sum_{i<j} beta_ij alpha_i(theta_i)

where the distance ratio is

    beta_ij = (D_ij D_s) / (D_j D_is)

with ``D_ab`` the angular diameter distance from redshift ``a`` to redshift ``b``, and ``D_s``
the distance to the **final** plane of the system. Two consequences of this normalisation are
used repeatedly below:

* ``beta_ij = 1`` when ``z_j = z_final`` (then ``D_j = D_s`` and ``D_ij = D_is``), and
* ``beta_ij = 0`` when ``z_i = z_j`` (then ``D_ij = 0``): a deflector sitting *at* plane ``j``
  cannot affect the mapping *to* plane ``j``.

**Convergence, shear and magnification** (Narayan & Bartelmann 1996,
https://inspirehep.net/literature/419263, equations 55 and 60). With ``A = d beta / d theta``
the Jacobian of the lens mapping,

    A = I - H,   H_ab = d alpha_a / d theta_b   (equation 55, the Hessian of the potential)
    kappa = tr(H) / 2 = 1 - tr(A) / 2
    gamma_1 = (H_xx - H_yy) / 2,   gamma_2 = H_xy
    mu = 1 / det(A)                                                        (equation 60)

**Jacobian recursion** (McCully, Keeton, Wong & Zabludoff 2014, arXiv:1401.0197). Differentiating
the lens equation above with respect to ``theta_1`` gives, with ``U_i = d alpha_i / d theta``
evaluated at ``theta_i``,

    A_1 = I
    A_j = I - sum_{i<j} beta_ij U_i A_i

This is a genuinely different numerical route to ``mu`` than central-differencing the traced
position, because the chain rule is applied analytically between planes and only the per-plane
``U_i`` are differenced.

Two convention traps
--------------------

**(a) ``deflections_between_planes_from`` is a difference of traced grids, not a deflection.**
``Tracer.deflections_between_planes_from(plane_i=i, plane_j=j)`` returns
``traced_grids[i] - traced_grids[j]``. Because every traced grid is built with the final-plane
normalisation above, this quantity is the *final-plane-scaled* difference between two planes'
positions. It is **not** the physical deflection at plane ``j`` in the convention where
``alpha_j`` would be scaled by ``D_js / D_s``. For ``plane_i=0`` it is exactly
``theta - theta_j``, which is what the oracle below asserts.

**(b) Truncating a tracer to planes <= j is not an oracle for plane j.** Dropping the planes
above ``j`` changes ``redshift_final``, and therefore changes *every* ``beta_ij`` in the
remaining recursion, because ``D_s`` and ``D_is`` in the formula above refer to the final plane
of whatever system is being traced. During PyAutoLens#480 this "obvious" cross-check gave a
magnification of 1.86 where the correct value was 27.9 — a factor of 15, from a convention
mismatch rather than a bug. The correct way to ask for a quantity at an intermediate plane is
to keep the full tracer and select the plane index (``plane_j=j``), which is what
``ag.LensCalc.from_tracer(tracer, plane_j=j)`` and the oracles here do.
"""

import numpy as np
import pytest

import astropy.units as u
from astropy.cosmology import Planck15 as astropy_planck15

import autogalaxy as ag
import autolens as al


# ----------------------------------------------------------------------------------------------
# Independent oracles. None of these call ``tracer_util`` or ``LensCalc``; the only shared code is
# each mass profile's own single-plane ``deflections_yx_2d_from``, which is itself cross-validated
# against analytic closed forms elsewhere.
# ----------------------------------------------------------------------------------------------


def _angular_diameter_distance(redshift_0, redshift_1):
    """
    Angular diameter distance between two redshifts, in Mpc, from astropy's ``Planck15``.

    Astropy 7 renamed the two-redshift form from ``angular_diameter_distance_z1z2`` to a
    two-argument ``angular_diameter_distance``; both are tried so the oracle is independent of
    the installed astropy version.
    """
    try:
        distance = astropy_planck15.angular_diameter_distance(redshift_0, redshift_1)
    except TypeError:
        distance = astropy_planck15.angular_diameter_distance_z1z2(
            redshift_0, redshift_1
        )
    return distance.to(u.Mpc).value


def _beta_astropy(redshift_i, redshift_j, redshift_final):
    """
    The distance ratio ``beta_ij = (D_ij D_s) / (D_j D_is)`` of SEF 1992 eq. 9.7b, computed from
    astropy angular diameter distances rather than from the project's own cosmology integration.
    """
    D_ij = _angular_diameter_distance(redshift_i, redshift_j)
    D_s = _angular_diameter_distance(0.0, redshift_final)
    D_j = _angular_diameter_distance(0.0, redshift_j)
    D_is = _angular_diameter_distance(redshift_i, redshift_final)

    return (D_ij * D_s) / (D_j * D_is)


def _beta_project(redshift_i, redshift_j, redshift_final):
    """
    The same distance ratio, from the project's own cosmology. Used to isolate the *recursion*
    from the *distance integration*: the two beta routines agree only to ~2e-7 (they integrate
    ``1 / E(z)`` differently, and that agreement is what
    ``test__scaling_factor__matches_astropy_planck15`` pins at 1e-6), so an oracle built on
    astropy betas can never test the ray-tracing algebra below that floor. Feeding the project's
    own betas into the paper's recursion removes the cosmology from the comparison entirely and
    lets the recursion itself be asserted to 1e-10.
    """
    return float(
        ag.cosmo.Planck15().scaling_factor_between_redshifts_from(
            redshift_0=redshift_i,
            redshift_1=redshift_j,
            redshift_final=redshift_final,
        )
    )


def _deflections(galaxies, positions):
    """
    Summed single-plane deflection angles of a list of galaxies at the given ``(N, 2)`` positions.
    """
    grid = al.Grid2DIrregular(values=np.asarray(positions, dtype=float))

    total = np.zeros(np.asarray(positions, dtype=float).shape)

    for galaxy in galaxies:
        total = total + np.asarray(galaxy.deflections_yx_2d_from(grid=grid))

    return total


def _trace_from_paper(planes, positions, redshift_final, beta_from=_beta_astropy):
    """
    The multi-plane lens equation of SEF 1992 eq. 9.6, written out here from the paper:

        theta_j = theta_1 - sum_{i<j} beta_ij alpha_i(theta_i)

    with ``beta_ij`` supplied by ``beta_from`` (astropy by default). Returns one ``(N, 2)``
    array of positions per plane.
    """
    redshifts = [galaxies[0].redshift for galaxies in planes]

    theta_list = []
    alpha_list = []

    for plane_index, galaxies in enumerate(planes):

        theta = np.asarray(positions, dtype=float).copy()

        for previous_index in range(plane_index):
            beta = beta_from(
                redshifts[previous_index], redshifts[plane_index], redshift_final
            )
            theta = theta - beta * alpha_list[previous_index]

        theta_list.append(theta)
        alpha_list.append(_deflections(galaxies, theta))

    return theta_list


def _jacobian_via_central_difference(mapping, positions, h=1e-6):
    """
    The ``(N, 2, 2)`` Jacobian ``d beta / d theta`` of an arbitrary ``(y, x) -> (beta_y, beta_x)``
    mapping, by 2-point central differences.
    """
    jacobians = []

    for y, x in np.asarray(positions, dtype=float):

        d_y = (np.asarray(mapping(y + h, x)) - np.asarray(mapping(y - h, x))) / (2.0 * h)
        d_x = (np.asarray(mapping(y, x + h)) - np.asarray(mapping(y, x - h))) / (2.0 * h)

        jacobians.append(np.array([[d_y[0], d_x[0]], [d_y[1], d_x[1]]]))

    return np.array(jacobians)


def _magnification_via_jacobian(mapping, positions, h=1e-6):
    """
    ``mu = 1 / det(A)`` (Narayan & Bartelmann eq. 60) from a numerically differenced Jacobian.
    """
    jacobians = _jacobian_via_central_difference(mapping, positions, h=h)

    return np.array([1.0 / np.linalg.det(jacobian) for jacobian in jacobians])


def _traced_position_mapping(tracer, plane_index):
    """
    The map ``theta -> beta_j`` through the implementation under test, as a scalar-in/array-out
    callable suitable for central differencing.
    """

    def mapping(y, x):
        grid = al.Grid2DIrregular(values=[(y, x)])
        return np.asarray(tracer.traced_grid_2d_list_from(grid=grid)[plane_index])[0]

    return mapping


def _deflection_gradient(galaxies, position, h=1e-6):
    """
    ``U_ab = d alpha_a / d theta_b`` for one plane's galaxies, at a single position, by central
    differences of the *profile* deflections (no ray tracing involved).
    """
    y, x = float(position[0]), float(position[1])

    d_y = (
        _deflections(galaxies, [(y + h, x)])[0] - _deflections(galaxies, [(y - h, x)])[0]
    ) / (2.0 * h)
    d_x = (
        _deflections(galaxies, [(y, x + h)])[0] - _deflections(galaxies, [(y, x - h)])[0]
    ) / (2.0 * h)

    return np.array([[d_y[0], d_x[0]], [d_y[1], d_x[1]]])


def _jacobian_via_recursion(planes, positions, redshift_final, h=1e-6):
    """
    The McCully et al. (2014) Jacobian recursion

        A_1 = I,   A_j = I - sum_{i<j} beta_ij U_i A_i

    Returns a list with one ``(N, 2, 2)`` array of Jacobians per plane. The chain rule is applied
    analytically between planes; only the per-plane ``U_i`` are differenced numerically.
    """
    redshifts = [galaxies[0].redshift for galaxies in planes]
    positions = np.asarray(positions, dtype=float)

    jacobian_list = [
        np.zeros((positions.shape[0], 2, 2)) for _ in range(len(planes))
    ]

    for position_index, position in enumerate(positions):

        theta_list = []
        A_list = []
        U_list = []

        for plane_index, galaxies in enumerate(planes):

            theta = position.copy()
            A = np.eye(2)

            for previous_index in range(plane_index):
                beta = _beta_astropy(
                    redshifts[previous_index], redshifts[plane_index], redshift_final
                )
                theta = theta - beta * _deflections(
                    planes[previous_index], [theta_list[previous_index]]
                )[0]
                A = A - beta * U_list[previous_index] @ A_list[previous_index]

            theta_list.append(theta)
            A_list.append(A)
            U_list.append(_deflection_gradient(galaxies, theta, h=h))

            jacobian_list[plane_index][position_index] = A

    return jacobian_list


def _convergence_from_jacobian(jacobians):
    """
    ``kappa = 1 - tr(A) / 2`` (Narayan & Bartelmann eq. 55 rearranged for ``A = I - H``).
    """
    return np.array([1.0 - np.trace(jacobian) / 2.0 for jacobian in jacobians])


def _shear_from_jacobian(jacobians):
    """
    The shear in the project's ``[gamma_2, gamma_1]`` column convention (see
    ``LensCalc.shear_yx_2d_via_hessian_from``), derived from ``H = I - A`` with
    ``H_ab = d alpha_a / d theta_b`` indexed ``(y, x)``:

        gamma_1 = (H_xx - H_yy) / 2 = (A_yy - A_xx) / 2
        gamma_2 = H_xy             = -A_xy_row       (i.e. ``-A[1, 0]``)

    Note the FIRST returned column is ``gamma_2`` and the second is ``gamma_1``.
    """
    shear = []

    for jacobian in jacobians:
        gamma_1 = 0.5 * (jacobian[0, 0] - jacobian[1, 1])
        gamma_2 = -jacobian[1, 0]
        shear.append([gamma_2, gamma_1])

    return np.array(shear)


# ----------------------------------------------------------------------------------------------
# Shared configurations.
# ----------------------------------------------------------------------------------------------


def _general_galaxies():
    """
    A four-plane system with three distinct elliptical deflectors and a massless source plane.
    """
    g_0 = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.1, -0.05), einstein_radius=1.2
        ),
    )
    g_1 = al.Galaxy(
        redshift=1.0,
        mass=al.mp.PowerLaw(
            centre=(0.15, -0.1), ell_comps=(0.05, 0.1), einstein_radius=0.4, slope=2.2
        ),
    )
    g_2 = al.Galaxy(
        redshift=1.5,
        mass=al.mp.NFW(
            centre=(-0.2, 0.25), ell_comps=(0.03, 0.02), kappa_s=0.08, scale_radius=5.0
        ),
    )
    g_3 = al.Galaxy(redshift=2.0)

    return [g_0, g_1, g_2, g_3]


GENERAL_POSITIONS = [
    (1.7, 0.9),
    (-1.4, 1.1),
    (0.8, -1.6),
    (-1.9, -0.7),
    (2.1, 0.3),
    (0.4, 2.0),
]


@pytest.fixture(name="general_tracer")
def make_general_tracer():
    return al.Tracer(galaxies=_general_galaxies(), cosmology=al.cosmo.Planck15())


@pytest.fixture(name="general_planes")
def make_general_planes():
    return [[galaxy] for galaxy in _general_galaxies()]


# ----------------------------------------------------------------------------------------------
# 1. The cosmology arm.
# ----------------------------------------------------------------------------------------------


def test__planck15_parameters__match_astropy():
    """
    The astropy comparison below is only meaningful if both cosmologies describe the same
    universe. ``ag.cosmo.Planck15`` is a hand-rolled ``FlatLambdaCDM`` with its own Simpson
    integration, so its parameters are checked against astropy's ``Planck15`` explicitly before
    any distance is compared.
    """
    cosmology = ag.cosmo.Planck15()

    assert cosmology.H0 == pytest.approx(astropy_planck15.H0.value, rel=1e-12)
    assert cosmology.Om0 == pytest.approx(astropy_planck15.Om0, rel=1e-12)
    assert cosmology.Tcmb0 == pytest.approx(astropy_planck15.Tcmb0.value, rel=1e-12)
    assert cosmology.Neff == pytest.approx(astropy_planck15.Neff, rel=1e-12)
    assert cosmology.Ob0 == pytest.approx(astropy_planck15.Ob0, rel=1e-12)

    assert np.sum(cosmology.m_nu) == pytest.approx(
        np.sum(astropy_planck15.m_nu.to(u.eV).value), rel=1e-12
    )


def test__scaling_factor__matches_astropy_planck15():
    """
    ``scaling_factor_between_redshifts_from`` must reproduce ``beta_ij = (D_ij D_s) / (D_j D_is)``
    (SEF 1992 eq. 9.7b) computed from astropy angular diameter distances. The project computes its
    distances by a hand-rolled Simpson integration of ``1 / E(z)``; astropy uses its own quadrature
    and a full massive-neutrino treatment, so agreement here is evidence rather than tautology.

    The ``(0.1, 1.0, 3.0)`` triple is the one behind the pinned ``beta_01 = 0.9348`` literal in
    ``test_tracer_util.py``, which until now was the only check that literal had.
    """
    cosmology = ag.cosmo.Planck15()

    redshift_triples = [
        (0.1, 1.0, 3.0),
        (0.1, 2.0, 3.0),
        (1.0, 2.0, 3.0),
        (0.5, 1.0, 2.0),
        (0.5, 1.5, 2.0),
        (0.2, 0.75, 1.5),
        (1.5, 2.5, 4.0),
    ]

    for redshift_i, redshift_j, redshift_final in redshift_triples:

        scaling_factor = float(
            cosmology.scaling_factor_between_redshifts_from(
                redshift_0=redshift_i,
                redshift_1=redshift_j,
                redshift_final=redshift_final,
            )
        )

        np.testing.assert_allclose(
            scaling_factor,
            _beta_astropy(redshift_i, redshift_j, redshift_final),
            rtol=1e-6,
            err_msg=f"beta mismatch for (z_i, z_j, z_final) = "
            f"({redshift_i}, {redshift_j}, {redshift_final})",
        )


def test__scaling_factor__limits_are_one_and_zero():
    """
    The two limits that make the final-plane normalisation what it is: ``beta_ij = 1`` when the
    target plane IS the final plane, and ``beta_ij = 0`` when the deflector sits at the target
    plane. Both are asserted against astropy as well as the implementation, since a formalism with
    a different normalisation (e.g. physical deflections scaled by ``D_js / D_s``) would fail here.
    """
    cosmology = ag.cosmo.Planck15()

    assert float(
        cosmology.scaling_factor_between_redshifts_from(
            redshift_0=0.5, redshift_1=2.0, redshift_final=2.0
        )
    ) == pytest.approx(1.0, abs=1e-12)
    assert _beta_astropy(0.5, 2.0, 2.0) == pytest.approx(1.0, abs=1e-12)

    assert float(
        cosmology.scaling_factor_between_redshifts_from(
            redshift_0=0.5, redshift_1=0.5, redshift_final=2.0
        )
    ) == pytest.approx(0.0, abs=1e-12)
    assert _beta_astropy(0.5, 0.5, 2.0) == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------------------------------
# 2. Degenerate reductions.
# ----------------------------------------------------------------------------------------------


def test__all_mass_in_one_plane__reduces_to_single_plane_analytic():
    """
    The sharpest degenerate case. A four-plane tracer whose mass all sits in the first plane must
    reduce to the single-plane lens equation at the final plane, where ``beta = 1``:

        theta_final = theta - alpha(theta)

    and every derived quantity must equal the profile's own analytic closed form: the isothermal
    sphere magnification ``mu = r / (r - theta_E)``, its analytic convergence, and its analytic
    shear. Nothing here goes through the multi-plane recursion, so any scaling-factor error at the
    final plane shows up as a mismatch against a closed form.
    """
    mass = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0)

    galaxies = [
        al.Galaxy(redshift=0.5, mass=mass),
        al.Galaxy(redshift=1.0),
        al.Galaxy(redshift=1.5),
        al.Galaxy(redshift=2.0),
    ]

    tracer = al.Tracer(galaxies=galaxies, cosmology=al.cosmo.Planck15())

    positions = [(1.7, 0.9), (-1.4, 1.1), (0.8, -1.6), (2.1, 0.3)]
    grid = al.Grid2DIrregular(values=positions)

    traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

    deflections = np.asarray(mass.deflections_yx_2d_from(grid=grid))

    np.testing.assert_allclose(
        np.asarray(traced_grid_list[-1]),
        np.asarray(positions) - deflections,
        rtol=1e-12,
        atol=1e-12,
    )

    radii = np.sqrt(np.sum(np.asarray(positions) ** 2.0, axis=1))
    magnification_analytic = radii / (radii - 1.0)

    magnification = np.asarray(
        ag.LensCalc.from_tracer(
            tracer, use_multi_plane=True, plane_i=0, plane_j=3
        ).magnification_2d_via_hessian_from(grid=grid)
    )

    np.testing.assert_allclose(
        magnification, magnification_analytic, rtol=1e-3, atol=1e-6
    )

    lens_calc = ag.LensCalc.from_tracer(
        tracer, use_multi_plane=True, plane_i=0, plane_j=3
    )

    np.testing.assert_allclose(
        np.asarray(lens_calc.convergence_2d_via_hessian_from(grid=grid)),
        np.asarray(mass.convergence_2d_from(grid=grid)),
        rtol=1e-3,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        np.asarray(lens_calc.shear_yx_2d_via_hessian_from(grid=grid)),
        np.asarray(mass.shear_yx_2d_from(grid=grid)),
        rtol=1e-3,
        atol=1e-6,
    )


def test__all_mass_in_one_plane__intermediate_plane_scales_by_beta():
    """
    At an intermediate plane the same tracer must give ``theta_j = theta - beta_0j alpha(theta)``,
    so the convergence measured at plane ``j`` is exactly ``beta_0j`` times the profile's analytic
    convergence. This is the test that a scaling factor computed with the wrong ``redshift_final``
    fails: truncating the tracer to planes <= j would give ``beta = 1`` here instead of 0.674.
    """
    mass = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0)

    galaxies = [
        al.Galaxy(redshift=0.5, mass=mass),
        al.Galaxy(redshift=1.0),
        al.Galaxy(redshift=2.0),
    ]

    tracer = al.Tracer(galaxies=galaxies, cosmology=al.cosmo.Planck15())

    positions = [(1.7, 0.9), (-1.4, 1.1), (0.8, -1.6), (2.1, 0.3)]
    grid = al.Grid2DIrregular(values=positions)

    beta = _beta_astropy(0.5, 1.0, 2.0)

    assert beta == pytest.approx(0.6739452, rel=1e-6)

    traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

    np.testing.assert_allclose(
        np.asarray(traced_grid_list[1]),
        np.asarray(positions) - beta * np.asarray(mass.deflections_yx_2d_from(grid=grid)),
        rtol=1e-6,
        atol=1e-10,
    )

    convergence = np.asarray(
        ag.LensCalc.from_tracer(
            tracer, use_multi_plane=True, plane_i=0, plane_j=1
        ).convergence_2d_via_hessian_from(grid=grid)
    )

    np.testing.assert_allclose(
        convergence,
        beta * np.asarray(mass.convergence_2d_from(grid=grid)),
        rtol=1e-3,
        atol=1e-6,
    )


def test__deflector_at_plane_j__does_not_change_mapping_to_plane_j():
    """
    ``beta_ij = 0`` when ``z_i = z_j``, so mass sitting *at* plane ``j`` cannot bend light that has
    only reached plane ``j``. Adding a deflector to the intermediate plane must therefore leave
    both the traced positions at that plane and the magnification there bit-for-bit unchanged,
    while changing the final plane. This was observed incidentally during PyAutoLens#480 and is
    pinned here.
    """
    lens = al.Galaxy(
        redshift=0.5, mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.6)
    )

    tracer_without = al.Tracer(
        galaxies=[lens, al.Galaxy(redshift=1.0), al.Galaxy(redshift=2.0)],
        cosmology=al.cosmo.Planck15(),
    )
    tracer_with = al.Tracer(
        galaxies=[
            lens,
            al.Galaxy(
                redshift=1.0,
                mass=al.mp.IsothermalSph(centre=(0.1, 0.1), einstein_radius=0.2),
            ),
            al.Galaxy(redshift=2.0),
        ],
        cosmology=al.cosmo.Planck15(),
    )

    positions = [(1.7, 0.9), (-1.4, 1.1), (0.8, -1.6), (2.1, 0.3)]
    grid = al.Grid2DIrregular(values=positions)

    traced_without = tracer_without.traced_grid_2d_list_from(grid=grid)
    traced_with = tracer_with.traced_grid_2d_list_from(grid=grid)

    np.testing.assert_allclose(
        np.asarray(traced_with[1]), np.asarray(traced_without[1]), rtol=1e-14, atol=1e-14
    )

    magnification_without = np.asarray(
        ag.LensCalc.from_tracer(
            tracer_without, use_multi_plane=True, plane_i=0, plane_j=1
        ).magnification_2d_via_hessian_from(grid=grid)
    )
    magnification_with = np.asarray(
        ag.LensCalc.from_tracer(
            tracer_with, use_multi_plane=True, plane_i=0, plane_j=1
        ).magnification_2d_via_hessian_from(grid=grid)
    )

    np.testing.assert_allclose(
        magnification_with, magnification_without, rtol=1e-12, atol=1e-12
    )

    assert not np.allclose(
        np.asarray(traced_with[2]), np.asarray(traced_without[2]), rtol=1e-6
    )


def test__three_plane__matches_explicit_longhand_formula():
    """
    The recursion written out longhand for two deflector planes, with no loop and no helper:

        theta_2 = theta_1 - beta_01 alpha_0(theta_1)
        theta_3 = theta_1 - beta_02 alpha_0(theta_1) - beta_12 alpha_1(theta_2)

    with ``beta_02 = beta_12 = 1`` because plane 2 is the final plane. A single misplaced scaling
    factor in the implementation's inner loop would break this.
    """
    g_0 = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=1.2),
    )
    g_1 = al.Galaxy(
        redshift=1.0,
        mass=al.mp.IsothermalSph(centre=(0.2, -0.1), einstein_radius=0.3),
    )
    g_2 = al.Galaxy(redshift=2.0)

    tracer = al.Tracer(galaxies=[g_0, g_1, g_2], cosmology=al.cosmo.Planck15())

    positions = np.array([(1.7, 0.9), (-1.4, 1.1), (0.8, -1.6), (2.1, 0.3)])
    grid = al.Grid2DIrregular(values=positions)

    beta_01 = _beta_astropy(0.5, 1.0, 2.0)

    assert _beta_astropy(0.5, 2.0, 2.0) == pytest.approx(1.0, abs=1e-12)
    assert _beta_astropy(1.0, 2.0, 2.0) == pytest.approx(1.0, abs=1e-12)

    alpha_0 = np.asarray(g_0.deflections_yx_2d_from(grid=grid))

    theta_2 = positions - beta_01 * alpha_0

    alpha_1 = np.asarray(
        g_1.deflections_yx_2d_from(grid=al.Grid2DIrregular(values=theta_2))
    )

    theta_3 = positions - alpha_0 - alpha_1

    traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

    np.testing.assert_allclose(np.asarray(traced_grid_list[0]), positions, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(traced_grid_list[1]), theta_2, rtol=1e-6, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(traced_grid_list[2]), theta_3, rtol=1e-6, atol=1e-10
    )


# ----------------------------------------------------------------------------------------------
# 3. The analytic two-aligned-isothermal-sphere arm.
# ----------------------------------------------------------------------------------------------

# Two aligned singular isothermal spheres, both centred on the origin. On the y-axis the deflection
# of an SIS is exactly ``theta_E`` directed radially outward, so the whole system collapses to a
# scalar problem with a closed-form solution and two Einstein rings (see below).
SIS_REDSHIFTS = (0.5, 1.0, 2.0)
SIS_EINSTEIN_RADII = (1.0, 0.5)


def _sis_tracer():
    return al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=SIS_REDSHIFTS[0],
                mass=al.mp.IsothermalSph(
                    centre=(0.0, 0.0), einstein_radius=SIS_EINSTEIN_RADII[0]
                ),
            ),
            al.Galaxy(
                redshift=SIS_REDSHIFTS[1],
                mass=al.mp.IsothermalSph(
                    centre=(0.0, 0.0), einstein_radius=SIS_EINSTEIN_RADII[1]
                ),
            ),
            al.Galaxy(redshift=SIS_REDSHIFTS[2]),
        ],
        cosmology=al.cosmo.Planck15(),
    )


def test__two_aligned_sis__matches_closed_form_on_the_axis():
    r"""
    Two aligned singular isothermal spheres, on the axis, have a closed-form multi-plane solution.
    An SIS deflects by exactly ``theta_E`` radially, so for an image at ``theta_1`` on the y-axis:

        theta_2 = theta_1 - beta_01 theta_E1 sign(theta_1)
        theta_3 = theta_1 - theta_E1 sign(theta_1) - theta_E2 sign(theta_2)

    using ``beta_02 = beta_12 = 1`` at the final plane. The sign of ``theta_2`` matters: images
    inside ``beta_01 theta_E1`` cross the axis before reaching the second deflector and are
    deflected the other way by it. This is a closed form derived from the paper's equation, not
    from the code. It is asserted twice: with astropy betas to 1e-6 (the floor set by the two
    cosmologies' differing quadrature) and with the project's own betas to 1e-10, which tests the
    tracing algebra with the cosmology removed. ``theta_3`` is beta-free and holds at 1e-10 either
    way.
    """
    tracer = _sis_tracer()

    positions = np.array([(2.0, 0.0), (1.5, 0.0), (0.5, 0.0), (0.2, 0.0), (-1.3, 0.0)])
    grid = al.Grid2DIrregular(values=positions)

    theta_1 = positions[:, 0]

    traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

    for beta_from, rtol in [(_beta_astropy, 1e-6), (_beta_project, 1e-10)]:

        beta_01 = beta_from(*SIS_REDSHIFTS)

        theta_2 = theta_1 - beta_01 * SIS_EINSTEIN_RADII[0] * np.sign(theta_1)
        theta_3 = (
            theta_1
            - SIS_EINSTEIN_RADII[0] * np.sign(theta_1)
            - SIS_EINSTEIN_RADII[1] * np.sign(theta_2)
        )

        # Both sign branches of theta_2 are exercised by the chosen positions.
        assert np.any(theta_2 > 0.0) and np.any(theta_2 < 0.0)

        np.testing.assert_allclose(
            np.asarray(traced_grid_list[1])[:, 0], theta_2, rtol=rtol, atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(traced_grid_list[2])[:, 0], theta_3, rtol=1e-10, atol=1e-10
        )

    np.testing.assert_allclose(
        np.asarray(traced_grid_list[2])[:, 1], 0.0, atol=1e-10
    )


def test__two_aligned_sis__double_einstein_ring_radii():
    r"""
    The strongest arm available: a system with a genuine closed-form multi-plane observable. For a
    source on the axis, ``theta_3 = 0`` has two positive roots when
    ``theta_E2 > (1 - beta_01) theta_E1``:

        outer ring (theta_2 > 0):  theta_1 = theta_E1 + theta_E2
        inner ring (theta_2 < 0):  theta_1 = theta_E1 - theta_E2

    i.e. the double Einstein ring. With ``theta_E1 = 1.0``, ``theta_E2 = 0.5`` and
    ``beta_01 = 0.674`` the radii are exactly 1.5 and 0.5 arcsec, independent of cosmology for the
    outer ring and of the second deflector's position for the inner one. Both must ray-trace to
    the origin of the final plane.
    """
    tracer = _sis_tracer()

    beta_01 = _beta_astropy(*SIS_REDSHIFTS)

    # The condition for two rings to exist, checked rather than assumed.
    assert SIS_EINSTEIN_RADII[1] > (1.0 - beta_01) * SIS_EINSTEIN_RADII[0]

    radius_outer = SIS_EINSTEIN_RADII[0] + SIS_EINSTEIN_RADII[1]
    radius_inner = SIS_EINSTEIN_RADII[0] - SIS_EINSTEIN_RADII[1]

    assert radius_inner < beta_01 * SIS_EINSTEIN_RADII[0] < radius_outer

    grid = al.Grid2DIrregular(
        values=[
            (radius_outer, 0.0),
            (radius_inner, 0.0),
            (-radius_outer, 0.0),
            (-radius_inner, 0.0),
        ]
    )

    traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

    np.testing.assert_allclose(
        np.asarray(traced_grid_list[2]), np.zeros((4, 2)), atol=1e-10
    )

    assert radius_outer != pytest.approx(radius_inner, abs=1e-3)


# ----------------------------------------------------------------------------------------------
# 4. The general elliptical multi-plane case, against every arm.
# ----------------------------------------------------------------------------------------------


def test__traced_positions__match_paper_recursion(general_tracer, general_planes):
    """
    Every plane of a four-plane elliptical system, against the SEF 1992 lens equation written from
    the paper. Run twice: with astropy scaling factors at 1e-6, and with the project's own scaling
    factors at 1e-10. The first arm tests the whole chain including the cosmology and is limited by
    the two cosmologies' quadrature (~2e-7); the second removes the cosmology and holds the
    recursion itself, plane ordering included, to 1e-10.
    """
    grid = al.Grid2DIrregular(values=GENERAL_POSITIONS)

    traced_grid_list = general_tracer.traced_grid_2d_list_from(grid=grid)

    for beta_from, rtol in [(_beta_astropy, 1e-6), (_beta_project, 1e-10)]:

        theta_list = _trace_from_paper(
            general_planes,
            GENERAL_POSITIONS,
            redshift_final=2.0,
            beta_from=beta_from,
        )

        assert len(traced_grid_list) == len(theta_list) == 4

        for plane_index in range(4):
            np.testing.assert_allclose(
                np.asarray(traced_grid_list[plane_index]),
                theta_list[plane_index],
                rtol=rtol,
                atol=1e-10,
                err_msg=f"traced positions disagree at plane {plane_index} "
                f"({beta_from.__name__})",
            )


def test__deflections_between_planes__is_the_traced_grid_difference(
    general_tracer, general_planes
):
    """
    Convention trap (a): ``deflections_between_planes_from(plane_i=0, plane_j=j)`` is
    ``theta - theta_j``, the final-plane-normalised *difference*, and NOT the physical deflection
    at plane ``j``. Asserted against the paper-recursion positions, and asserted to differ from the
    summed single-plane deflections of the planes below ``j`` (which is what a reader assuming the
    other convention would expect).
    """
    grid = al.Grid2DIrregular(values=GENERAL_POSITIONS)

    theta_list = _trace_from_paper(
        general_planes,
        GENERAL_POSITIONS,
        redshift_final=2.0,
        beta_from=_beta_project,
    )

    for plane_index in range(1, 4):

        deflections = np.asarray(
            general_tracer.deflections_between_planes_from(
                grid=grid, plane_i=0, plane_j=plane_index
            )
        )

        np.testing.assert_allclose(
            deflections,
            np.asarray(GENERAL_POSITIONS) - theta_list[plane_index],
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"deflections_between_planes disagree at plane {plane_index}",
        )

    # At an intermediate plane the scaled difference is NOT the unscaled sum of the deflections of
    # the planes in front of it: the beta factors are strictly less than one there.
    deflections_plane_1 = np.asarray(
        general_tracer.deflections_between_planes_from(grid=grid, plane_i=0, plane_j=1)
    )
    unscaled = _deflections(general_planes[0], GENERAL_POSITIONS)

    assert not np.allclose(deflections_plane_1, unscaled, rtol=1e-3)


def test__magnification__matches_both_jacobian_oracles(general_tracer, general_planes):
    """
    Three routes to the magnification at every plane of the four-plane system:

    * ``LensCalc.magnification_2d_via_hessian_from`` — Richardson-extrapolated derivatives of
      ``deflections_between_planes_from``;
    * central differences of ``theta -> beta_j`` through ``traced_grid_2d_list_from``, the oracle
      that settled PyAutoLens#480;
    * the McCully et al. Jacobian recursion, which never ray-traces a perturbed position at all.

    The last two share no code with the first, and the third shares no code with the second beyond
    the mass profiles themselves.
    """
    grid = al.Grid2DIrregular(values=GENERAL_POSITIONS)

    jacobian_list = _jacobian_via_recursion(
        general_planes, GENERAL_POSITIONS, redshift_final=2.0
    )

    for plane_index in range(1, 4):

        magnification_hessian = np.asarray(
            ag.LensCalc.from_tracer(
                general_tracer, use_multi_plane=True, plane_i=0, plane_j=plane_index
            ).magnification_2d_via_hessian_from(grid=grid)
        )

        magnification_traced = _magnification_via_jacobian(
            _traced_position_mapping(general_tracer, plane_index), GENERAL_POSITIONS
        )

        magnification_recursion = np.array(
            [1.0 / np.linalg.det(jacobian) for jacobian in jacobian_list[plane_index]]
        )

        np.testing.assert_allclose(
            magnification_traced,
            magnification_recursion,
            rtol=1e-3,
            err_msg=f"ray-traced and recursion magnifications disagree at plane {plane_index}",
        )

        np.testing.assert_allclose(
            magnification_hessian,
            magnification_traced,
            rtol=1e-3,
            err_msg=f"Hessian and ray-traced magnifications disagree at plane {plane_index}",
        )


def test__magnification__stable_across_finite_difference_step(general_tracer):
    """
    The ray-traced Jacobian is only an oracle if it is not itself measuring step-size noise. The
    magnification at the final plane is computed at ``h`` from 1e-4 down to 1e-7 and required to
    agree with itself to 1e-6 relative, which is what makes a disagreement with the Hessian path a
    finding about the Hessian path (as it was for the bug in
    ``LensCalc._hessian_via_richardson``) rather than about the oracle.
    """
    mapping = _traced_position_mapping(general_tracer, 3)

    reference = _magnification_via_jacobian(mapping, GENERAL_POSITIONS, h=1e-6)

    for h in [1e-4, 1e-5, 1e-7]:
        np.testing.assert_allclose(
            _magnification_via_jacobian(mapping, GENERAL_POSITIONS, h=h),
            reference,
            rtol=1e-6,
            err_msg=f"ray-traced magnification is not stable at h={h}",
        )


def test__convergence_and_shear__match_recursion_jacobian(
    general_tracer, general_planes
):
    """
    Convergence and shear at every plane, from the McCully recursion Jacobian via
    ``kappa = 1 - tr(A) / 2`` and the ``[gamma_2, gamma_1]`` column convention of
    ``LensCalc.shear_yx_2d_via_hessian_from``. A column swap or a sign flip in either path breaks
    this, which is exactly the failure mode the single-plane isothermal cross-check guards against
    and which multi-plane had no guard for.
    """
    grid = al.Grid2DIrregular(values=GENERAL_POSITIONS)

    jacobian_list = _jacobian_via_recursion(
        general_planes, GENERAL_POSITIONS, redshift_final=2.0
    )

    for plane_index in range(1, 4):

        lens_calc = ag.LensCalc.from_tracer(
            general_tracer, use_multi_plane=True, plane_i=0, plane_j=plane_index
        )

        np.testing.assert_allclose(
            np.asarray(lens_calc.convergence_2d_via_hessian_from(grid=grid)),
            _convergence_from_jacobian(jacobian_list[plane_index]),
            rtol=1e-3,
            atol=1e-6,
            err_msg=f"convergence disagrees at plane {plane_index}",
        )

        np.testing.assert_allclose(
            np.asarray(lens_calc.shear_yx_2d_via_hessian_from(grid=grid)),
            _shear_from_jacobian(jacobian_list[plane_index]),
            rtol=1e-3,
            atol=1e-6,
            err_msg=f"shear disagrees at plane {plane_index}",
        )


# ----------------------------------------------------------------------------------------------
# 5. The known NumPy Richardson-step defect, pinned as a strict xfail.
# ----------------------------------------------------------------------------------------------

# The PyAutoLens#480 configuration (the fixture in ``test_autolens/point/triangles/
# test_solver_multi_plane.py``): a smooth main lens, and an intermediate source at z=1.0 which is
# itself a compact deflector centred ON the point source it hosts. Rays that image that source
# therefore pass within ~4e-4 arcsec of a singular isothermal centre on their way to z=2.0, which
# is the regime where a fixed 0.01 arcsec finite-difference step cannot work.
BUG_480_POSITIONS = [
    (0.89140625, 0.63102580),
    (-0.67578125, -0.76634227),
    (1.07109375, -0.24131437),
    (-0.36875000, 1.04644736),
]

# mu at the LAST plane (z=2.0) at those positions, from JAX exact autodiff in float64, as recorded
# in ``PyAutoMind/complete/2026/08/point-solver-magnification-plane-redshift.md``. These are the
# reference values this module's own ray-traced oracle is checked against: an exact automatic
# derivative, computed by a different tool at a different time, is as independent as it gets.
BUG_480_MAGNIFICATION_LAST_PLANE = [0.04508, 0.01099, -0.08602, -0.01118]


def _bug_480_tracer():
    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        ),
    )
    intermediate_source = al.Galaxy(
        redshift=1.0,
        mass=al.mp.Isothermal(
            centre=(0.02, 0.03),
            einstein_radius=0.2,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        ),
    )
    far_source = al.Galaxy(redshift=2.0)

    return al.Tracer(galaxies=[lens_galaxy, intermediate_source, far_source])


def test__bug_480_configuration__ray_traced_jacobian_matches_recorded_jax_autodiff():
    """
    The oracle's own credentials, on the configuration that exposed the Richardson-step defect.

    The ray-traced Jacobian must reproduce the JAX float64 autodiff magnifications recorded when
    PyAutoLens#480 was settled. It does, to 1e-3, at every one of the four image positions. This is
    what licenses the xfail below to be read as a statement about ``_hessian_via_richardson`` and
    not about the oracle.

    The first three positions are additionally required to be stable between h=1e-6 and h=1e-7. The
    fourth is not: it sits closest to the singular centre and moves from -0.01101 to -0.01118 over
    that range (the ray-traced value recorded in #480 for it, -0.01101, likewise differs from the
    JAX value -0.01118 in the third decimal place). It is excluded from the stability assertion
    rather than absorbed by a looser tolerance.
    """
    tracer = _bug_480_tracer()

    mapping = _traced_position_mapping(tracer, 2)

    magnification = _magnification_via_jacobian(mapping, BUG_480_POSITIONS, h=1e-7)

    np.testing.assert_allclose(
        magnification, BUG_480_MAGNIFICATION_LAST_PLANE, rtol=1e-3
    )

    np.testing.assert_allclose(
        _magnification_via_jacobian(mapping, BUG_480_POSITIONS[:3], h=1e-6),
        _magnification_via_jacobian(mapping, BUG_480_POSITIONS[:3], h=1e-7),
        rtol=1e-3,
    )


def test__bug_480_configuration__richardson_hessian_agrees_at_the_intermediate_plane():
    """
    The control that makes the defect below specific rather than general. The map to z=1.0 involves
    only the smooth main lens, and there ``LensCalc``'s Richardson-extrapolated Hessian agrees with
    the ray-traced Jacobian to 1e-6 on the very same tracer and the very same positions. The
    failure at the last plane is therefore a property of the step size against the deflection
    field's scale, not a broken Hessian.
    """
    tracer = _bug_480_tracer()

    grid = al.Grid2DIrregular(values=BUG_480_POSITIONS)

    magnification_hessian = np.asarray(
        ag.LensCalc.from_tracer(
            tracer, use_multi_plane=True, plane_i=0, plane_j=1
        ).magnification_2d_via_hessian_from(grid=grid)
    )

    magnification_traced = _magnification_via_jacobian(
        _traced_position_mapping(tracer, 1), BUG_480_POSITIONS
    )

    np.testing.assert_allclose(magnification_hessian, magnification_traced, rtol=1e-6)


def test__bug_480_configuration__numpy_richardson_hessian_agrees_at_the_last_plane():
    """
    Regression test for PyAutoGalaxy#591.

    ``LensCalc.magnification_2d_via_hessian_from`` at the last plane must agree with the ray-traced
    Jacobian, which the test above shows reproduces exact autodiff on this configuration. It did
    not while ``LensCalc._hessian_via_richardson`` used a hardcoded ``buffer=0.01`` arcsec step:
    where a ray passes ~4e-4 arcsec from the compact z=1.0 deflector's centre that step straddles
    the whole deflector, and the NumPy path returned
    ``[-0.00694, -0.00221, 0.00139, 0.00246]`` against the ``[0.04508, 0.01099, -0.08602,
    -0.01118]`` that exact JAX autodiff and the ray-traced Jacobian agree on -- wrong by
    ~100-120% with a flipped sign on all four points. PyAutoGalaxy#591 made the Richardson step
    adapt per point, and this test now asserts the agreement rather than pinning the defect.

    The tolerance is the same 1e-3 used everywhere else in this module: it was not widened when the
    test recorded a disagreement, and it is not tightened now that it records an agreement.
    """
    tracer = _bug_480_tracer()

    grid = al.Grid2DIrregular(values=BUG_480_POSITIONS)

    magnification_hessian = np.asarray(
        ag.LensCalc.from_tracer(
            tracer, use_multi_plane=True, plane_i=0, plane_j=2
        ).magnification_2d_via_hessian_from(grid=grid)
    )

    np.testing.assert_allclose(
        magnification_hessian, BUG_480_MAGNIFICATION_LAST_PLANE, rtol=1e-3
    )

import math

import numpy as np
import pytest

import autofit as af
from autolens import dimensions as dim
from autolens.data.array import grids
from autolens.model.profiles import mass_profiles as mp

from test.unit.mock.model import mock_cosmology

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


@pytest.fixture(autouse=True)
def reset_config():
    """
    Use configuration from the default path. You may want to change this to set a specific path.
    """
    af.conf.instance = af.conf.default


class TestEinsteinRadiusMass(object):
    def test__radius_of_critical_curve_and_einstein_radius__radius_unit_conversions(
        self
    ):

        cosmology = mock_cosmology.MockCosmology(kpc_per_arcsec=2.0)

        sis_arcsec = mp.SphericalIsothermal(
            centre=(dim.Length(0.0, "arcsec"), dim.Length(0.0, "arcsec")),
            einstein_radius=dim.Length(2.0, "arcsec"),
        )

        radius = sis_arcsec.average_convergence_of_1_radius_in_units(
            unit_length="arcsec", cosmology=cosmology
        )
        assert radius == pytest.approx(2.0, 1e-4)

        radius = sis_arcsec.einstein_radius_in_units(unit_length="arcsec")
        assert radius == pytest.approx(2.0, 1e-4)

        radius = sis_arcsec.einstein_radius_in_units(
            unit_length="kpc", redshift_profile=0.5, cosmology=cosmology
        )
        assert radius == pytest.approx(4.0, 1e-4)

        sis_kpc = mp.SphericalIsothermal(
            centre=(dim.Length(0.0, "kpc"), dim.Length(0.0, "kpc")),
            einstein_radius=dim.Length(2.0, "kpc"),
        )

        radius = sis_kpc.average_convergence_of_1_radius_in_units(unit_length="kpc")
        assert radius == pytest.approx(2.0, 1e-4)

        radius = sis_kpc.einstein_radius_in_units(unit_length="kpc")
        assert radius == pytest.approx(2.0, 1e-4)

        radius = sis_kpc.einstein_radius_in_units(
            unit_length="arcsec", redshift_profile=0.5, cosmology=cosmology
        )
        assert radius == pytest.approx(1.0, 1e-4)

        nfw_arcsec = mp.SphericalNFW(
            centre=(dim.Length(0.0, "arcsec"), dim.Length(0.0, "arcsec")),
            kappa_s=0.5,
            scale_radius=dim.Length(5.0, "arcsec"),
        )

        radius = nfw_arcsec.average_convergence_of_1_radius_in_units(
            unit_length="arcsec"
        )
        assert radius == pytest.approx(2.76386, 1e-4)

        radius = nfw_arcsec.einstein_radius_in_units(
            unit_length="arcsec", cosmology=cosmology
        )
        assert radius == pytest.approx(2.76386, 1e-4)

        radius = nfw_arcsec.einstein_radius_in_units(
            unit_length="kpc", redshift_profile=0.5, cosmology=cosmology
        )
        assert radius == pytest.approx(2.0 * 2.76386, 1e-4)

        nfw_kpc = mp.SphericalNFW(
            centre=(dim.Length(0.0, "kpc"), dim.Length(0.0, "kpc")),
            kappa_s=0.5,
            scale_radius=dim.Length(5.0, "kpc"),
        )

        radius = nfw_kpc.average_convergence_of_1_radius_in_units(
            unit_length="kpc", redshift_profile=0.5, cosmology=cosmology
        )
        assert radius == pytest.approx(2.76386, 1e-4)

        radius = nfw_kpc.einstein_radius_in_units(unit_length="kpc")
        assert radius == pytest.approx(2.76386, 1e-4)

        radius = nfw_kpc.einstein_radius_in_units(
            unit_length="arcsec", redshift_profile=0.5, cosmology=cosmology
        )
        assert radius == pytest.approx(0.5 * 2.76386, 1e-4)

    def test__einstein_mass__radius_unit_conversions(self):

        cosmology = mock_cosmology.MockCosmology(
            kpc_per_arcsec=2.0, critical_surface_density=2.0
        )

        sis_arcsec = mp.SphericalIsothermal(
            centre=(dim.Length(0.0, "arcsec"), dim.Length(0.0, "arcsec")),
            einstein_radius=dim.Length(1.0, "arcsec"),
        )

        mass = sis_arcsec.einstein_mass_in_units(
            unit_mass="angular", cosmology=cosmology
        )
        assert mass == pytest.approx(np.pi, 1e-4)

        mass = sis_arcsec.einstein_mass_in_units(
            unit_mass="solMass",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )
        assert mass == pytest.approx(2.0 * np.pi, 1e-4)

        sis_kpc = mp.SphericalIsothermal(
            centre=(dim.Length(0.0, "kpc"), dim.Length(0.0, "kpc")),
            einstein_radius=dim.Length(2.0, "kpc"),
        )

        mass = sis_kpc.einstein_mass_in_units(
            unit_mass="angular", redshift_profile=0.5, cosmology=cosmology
        )
        assert mass == pytest.approx(4.0 * np.pi, 1e-4)

        mass = sis_kpc.einstein_mass_in_units(
            unit_mass="solMass",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )
        assert mass == pytest.approx(2.0 * np.pi, 1e-4)

        nfw_arcsec = mp.SphericalNFW(
            centre=(dim.Length(0.0, "arcsec"), dim.Length(0.0, "arcsec")),
            kappa_s=0.5,
            scale_radius=dim.Length(5.0, "arcsec"),
        )

        mass = nfw_arcsec.einstein_mass_in_units(
            unit_mass="angular", cosmology=cosmology
        )
        assert mass == pytest.approx(np.pi * 2.76386 ** 2.0, 1e-4)

        mass = nfw_arcsec.einstein_mass_in_units(
            unit_mass="solMass",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )
        assert mass == pytest.approx(2.0 * np.pi * 2.76386 ** 2.0, 1e-4)

        nfw_kpc = mp.SphericalNFW(
            centre=(dim.Length(0.0, "kpc"), dim.Length(0.0, "kpc")),
            kappa_s=0.5,
            scale_radius=dim.Length(5.0, "kpc"),
        )

        mass = nfw_kpc.einstein_mass_in_units(
            unit_mass="angular", redshift_profile=0.5, cosmology=cosmology
        )
        assert mass == pytest.approx(np.pi * 2.76386 ** 2.0, 1e-4)

        mass = nfw_kpc.einstein_mass_in_units(
            unit_mass="solMass",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )
        assert mass == pytest.approx(0.5 * np.pi * 2.76386 ** 2.0, 1e-4)


def mass_within_radius_of_profile_from_grid_calculation(radius, profile):

    mass_total = 0.0

    xs = np.linspace(-radius * 1.5, radius * 1.5, 40)
    ys = np.linspace(-radius * 1.5, radius * 1.5, 40)

    edge = xs[1] - xs[0]
    area = edge ** 2

    for x in xs:
        for y in ys:

            eta = profile.grid_to_elliptical_radii(np.array([[x, y]]))

            if eta < radius:
                mass_total += profile.convergence_func(eta) * area

    return mass_total


class TestMassWithinCircle(object):
    def test__mass_in_angular_units__singular_isothermal_sphere__compare_to_analytic(
        self
    ):

        sis = mp.SphericalIsothermal(einstein_radius=2.0)

        radius = dim.Length(2.0, "arcsec")

        mass = sis.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )
        assert math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)

        sis = mp.SphericalIsothermal(einstein_radius=4.0)

        radius = dim.Length(4.0, "arcsec")

        mass = sis.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )
        assert math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)

    def test__mass_in_angular_units__singular_isothermal__compare_to_grid(self):

        sis = mp.SphericalIsothermal(einstein_radius=2.0)

        radius = dim.Length(1.0, "arcsec")

        mass_grid = mass_within_radius_of_profile_from_grid_calculation(
            radius=radius, profile=sis
        )

        mass = sis.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )

        assert mass_grid == pytest.approx(mass, 0.02)

    def test__radius_units_conversions__mass_profile_updates_units_and_computes_correct_mass(
        self
    ):

        cosmology = mock_cosmology.MockCosmology(kpc_per_arcsec=2.0)

        # arcsec -> arcsec

        sis_arcsec = mp.SphericalIsothermal(
            centre=(dim.Length(0.0, "arcsec"), dim.Length(0.0, "arcsec")),
            einstein_radius=dim.Length(2.0, "arcsec"),
        )

        radius = dim.Length(2.0, "arcsec")
        mass = sis_arcsec.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )
        assert math.pi * sis_arcsec.einstein_radius * radius == pytest.approx(
            mass, 1e-3
        )

        # arcsec -> kpc

        radius = dim.Length(2.0, "kpc")
        mass = sis_arcsec.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
            cosmology=cosmology,
        )
        assert math.pi * sis_arcsec.einstein_radius * 1.0 == pytest.approx(mass, 1e-3)

        # 2.0 arcsec = 4.0 kpc, same masses.

        radius = dim.Length(2.0, "arcsec")
        mass_arcsec = sis_arcsec.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
            cosmology=cosmology,
        )
        radius = dim.Length(4.0, "kpc")
        mass_kpc = sis_arcsec.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
            cosmology=cosmology,
        )
        assert mass_arcsec == mass_kpc

        # kpc -> kpc

        sis_kpc = mp.SphericalIsothermal(
            centre=(dim.Length(0.0, "kpc"), dim.Length(0.0, "kpc")),
            einstein_radius=dim.Length(2.0, "kpc"),
        )

        radius = dim.Length(2.0, "kpc")
        mass = sis_kpc.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
            cosmology=cosmology,
        )
        assert math.pi * sis_kpc.einstein_radius * radius == pytest.approx(mass, 1e-3)

        # kpc -> arcsec

        radius = dim.Length(2.0, "arcsec")
        mass = sis_kpc.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
            cosmology=cosmology,
        )
        assert 2.0 * math.pi * sis_kpc.einstein_radius * radius == pytest.approx(
            mass, 1e-3
        )

        # 2.0 arcsec = 4.0 kpc, same masses.

        radius = dim.Length(2.0, "arcsec")
        mass_arcsec = sis_kpc.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
            cosmology=cosmology,
        )
        radius = dim.Length(4.0, "kpc")
        mass_kpc = sis_kpc.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )
        assert mass_arcsec == mass_kpc

    def test__mass_units_conversions__multiplies_by_critical_surface_density_factor(
        self
    ):

        cosmology = mock_cosmology.MockCosmology(critical_surface_density=2.0)

        sis = mp.SphericalIsothermal(einstein_radius=2.0)
        radius = dim.Length(2.0, "arcsec")

        mass = sis.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
            cosmology=cosmology,
        )
        assert math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)

        mass = sis.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="solMass",
            cosmology=cosmology,
        )
        assert 2.0 * math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)

        mass = sis.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="solMass",
            cosmology=cosmology,
        )
        assert 2.0 * math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)


class TestMassWithinEllipse(object):
    def test__mass_in_angular_units__singular_isothermal_sphere__compare_circle_and_ellipse(
        self
    ):

        sis = mp.SphericalIsothermal(einstein_radius=2.0)

        radius = dim.Length(2.0)
        mass_circle = sis.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )
        mass_ellipse = sis.mass_within_ellipse_in_units(
            major_axis=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )
        assert mass_circle == mass_ellipse

        sie = mp.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.5, phi=0.0)
        radius = dim.Length(2.0)
        mass_circle = sie.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )
        mass_ellipse = sie.mass_within_ellipse_in_units(
            major_axis=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )
        assert mass_circle == mass_ellipse * 2.0

    def test__mass_in_angular_units__singular_isothermal_ellipsoid__compare_to_grid(
        self
    ):

        sie = mp.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.5, phi=0.0)

        radius = dim.Length(0.5)

        mass_grid = mass_within_radius_of_profile_from_grid_calculation(
            radius=radius, profile=sie
        )

        mass = sie.mass_within_ellipse_in_units(
            major_axis=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )

        # Large errors required due to cusp at center of SIE - can get to errors of 0.01 for a 400 x 400 grid.
        assert mass_grid == pytest.approx(mass, 0.1)

    def test__radius_units_conversions__mass_profile_updates_units_and_computes_correct_mass(
        self
    ):

        cosmology = mock_cosmology.MockCosmology(kpc_per_arcsec=2.0)

        # arcsec -> arcsec

        sie_arcsec = mp.SphericalIsothermal(
            centre=(dim.Length(0.0, "arcsec"), dim.Length(0.0, "arcsec")),
            einstein_radius=dim.Length(2.0, "arcsec"),
        )

        major_axis = dim.Length(0.5, "arcsec")

        mass_grid = mass_within_radius_of_profile_from_grid_calculation(
            radius=major_axis, profile=sie_arcsec
        )

        mass = sie_arcsec.mass_within_ellipse_in_units(
            major_axis=major_axis,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )
        assert mass_grid == pytest.approx(mass, 0.1)

        # arcsec -> kpc

        major_axis = dim.Length(0.5, "kpc")
        mass = sie_arcsec.mass_within_ellipse_in_units(
            major_axis=major_axis,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
            cosmology=cosmology,
        )
        assert 0.5 * mass_grid == pytest.approx(mass, 0.1)

        # 2.0 arcsec = 4.0 kpc, same masses.

        radius = dim.Length(2.0, "arcsec")
        mass_arcsec = sie_arcsec.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
            cosmology=cosmology,
        )
        radius = dim.Length(4.0, "kpc")
        mass_kpc = sie_arcsec.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
            cosmology=cosmology,
        )
        assert mass_arcsec == mass_kpc

        # kpc -> kpc

        sie_kpc = mp.SphericalIsothermal(
            centre=(dim.Length(0.0, "kpc"), dim.Length(0.0, "kpc")),
            einstein_radius=dim.Length(2.0, "kpc"),
        )

        major_axis = dim.Length(0.5, "kpc")
        mass = sie_kpc.mass_within_ellipse_in_units(
            major_axis=major_axis,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )
        assert mass_grid == pytest.approx(mass, 0.1)

        # kpc -> arcsec

        major_axis = dim.Length(0.5, "arcsec")
        mass = sie_kpc.mass_within_ellipse_in_units(
            major_axis=major_axis,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
            cosmology=cosmology,
        )
        assert 2.0 * mass_grid == pytest.approx(mass, 0.1)

        # 2.0 arcsec = 4.0 kpc, same masses.

        radius = dim.Length(2.0, "arcsec")
        mass_arcsec = sie_kpc.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
            cosmology=cosmology,
        )
        radius = dim.Length(4.0, "kpc")
        mass_kpc = sie_kpc.mass_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )
        assert mass_arcsec == mass_kpc

    def test__mass_unit_conversions__compare_to_grid__mutliplies_by_critical_surface_density(
        self
    ):

        cosmology = mock_cosmology.MockCosmology(critical_surface_density=2.0)

        sie = mp.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.5, phi=0.0)

        radius = dim.Length(2.0, "arcsec")

        mass_grid = mass_within_radius_of_profile_from_grid_calculation(
            radius=radius, profile=sie
        )
        mass = sie.mass_within_ellipse_in_units(
            major_axis=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
            cosmology=cosmology,
        )

        # Large errors required due to cusp at center of SIE - can get to errors of 0.01 for a 400 x 400 grid.
        assert mass_grid == pytest.approx(radius * sie.axis_ratio * mass, 0.1)

        critical_surface_density = dim.MassOverLength2(2.0, "arcsec", "solMass")
        mass = sie.mass_within_ellipse_in_units(
            major_axis=radius,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="solMass",
            cosmology=cosmology,
        )

        # Large errors required due to cusp at center of SIE - can get to errors of 0.01 for a 400 x 400 grid.
        assert mass_grid == pytest.approx(0.5 * radius * sie.axis_ratio * mass, 0.1)


class TestDensityBetweenAnnuli(object):
    def test__circular_annuli__sis__analyic_density_agrees(self):

        cosmology = mock_cosmology.MockCosmology(
            kpc_per_arcsec=2.0, critical_surface_density=2.0
        )

        einstein_radius = 1.0
        sis_arcsec = mp.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=einstein_radius
        )

        inner_annuli_radius = dim.Length(2.0, "arcsec")
        outer_annuli_radius = dim.Length(3.0, "arcsec")

        inner_mass = math.pi * einstein_radius * inner_annuli_radius
        outer_mass = math.pi * einstein_radius * outer_annuli_radius

        density_between_annuli = sis_arcsec.density_between_circular_annuli_in_angular_units(
            inner_annuli_radius=inner_annuli_radius,
            outer_annuli_radius=outer_annuli_radius,
            unit_length="arcsec",
            unit_mass="angular",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )

        annuli_area = (np.pi * outer_annuli_radius ** 2.0) - (
            np.pi * inner_annuli_radius ** 2.0
        )

        assert (outer_mass - inner_mass) / annuli_area == pytest.approx(
            density_between_annuli, 1e-4
        )

        density_between_annuli = sis_arcsec.density_between_circular_annuli_in_angular_units(
            inner_annuli_radius=inner_annuli_radius,
            outer_annuli_radius=outer_annuli_radius,
            unit_length="arcsec",
            unit_mass="solMass",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )

        annuli_area = (np.pi * outer_annuli_radius ** 2.0) - (
            np.pi * inner_annuli_radius ** 2.0
        )

        assert (2.0 * outer_mass - 2.0 * inner_mass) / annuli_area == pytest.approx(
            density_between_annuli, 1e-4
        )

        density_between_annuli = sis_arcsec.density_between_circular_annuli_in_angular_units(
            inner_annuli_radius=inner_annuli_radius,
            outer_annuli_radius=outer_annuli_radius,
            unit_length="kpc",
            unit_mass="angular",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )

        inner_mass = math.pi * 2.0 * einstein_radius * inner_annuli_radius
        outer_mass = math.pi * 2.0 * einstein_radius * outer_annuli_radius

        annuli_area = (np.pi * 2.0 * outer_annuli_radius ** 2.0) - (
            np.pi * 2.0 * inner_annuli_radius ** 2.0
        )

        assert (outer_mass - inner_mass) / annuli_area == pytest.approx(
            density_between_annuli, 1e-4
        )

    def test__circular_annuli__nfw_profile__compare_to_manual_mass(self):

        cosmology = mock_cosmology.MockCosmology(
            kpc_per_arcsec=2.0, critical_surface_density=2.0
        )

        nfw = mp.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, kappa_s=1.0)

        inner_mass = nfw.mass_within_circle_in_units(
            radius=dim.Length(1.0),
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )

        outer_mass = nfw.mass_within_circle_in_units(
            radius=dim.Length(2.0),
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="angular",
        )

        density_between_annuli = nfw.density_between_circular_annuli_in_angular_units(
            inner_annuli_radius=dim.Length(1.0),
            outer_annuli_radius=dim.Length(2.0),
            unit_length="arcsec",
            unit_mass="angular",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )

        annuli_area = (np.pi * 2.0 ** 2.0) - (np.pi * 1.0 ** 2.0)

        assert (outer_mass - inner_mass) / annuli_area == pytest.approx(
            density_between_annuli, 1e-4
        )


class TestDeflectionsViaPotential(object):
    def test__compare_sis_deflections_via_potential_and_calculation__reg_grid(self):

        sis = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(10, 10), pixel_scale=0.05
        )

        deflections_via_calculation = sis.deflections_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        deflections_via_potential = sis.deflections_via_potential_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        mean_error = np.mean(deflections_via_potential - deflections_via_calculation)

        assert mean_error < 1e-4

    def test__compare_sie_at_phi_45__deflections_via_potential_and_calculation__reg_grid(
        self
    ):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=45.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(10, 10), pixel_scale=0.05
        )

        deflections_via_calculation = sie.deflections_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        deflections_via_potential = sie.deflections_via_potential_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        mean_error = np.mean(deflections_via_potential - deflections_via_calculation)

        assert mean_error < 1e-4

    def test__compare_sie_at_phi_0__deflections_via_potential_and_calculation_reg_grid(
        self
    ):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(10, 10), pixel_scale=0.05
        )

        deflections_via_calculation = sie.deflections_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        deflections_via_potential = sie.deflections_via_potential_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        mean_error = np.mean(deflections_via_potential - deflections_via_calculation)

        assert mean_error < 1e-4

    def test__sub_grid_binning(self):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(10, 10), pixel_scale=0.05, sub_grid_size=2
        )

        deflections_binned_reg_grid = sie.deflections_via_potential_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        deflections_sub_grid = sie.deflections_via_potential_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        pixel_1_reg_grid = deflections_binned_reg_grid[0]

        pixel_1_from_av_sub_grid = (
            deflections_sub_grid[0]
            + deflections_sub_grid[1]
            + deflections_sub_grid[2]
            + deflections_sub_grid[3]
        ) / 4

        assert pixel_1_reg_grid == pytest.approx(pixel_1_from_av_sub_grid, 1e-4)

        pixel_10000_reg_grid = deflections_binned_reg_grid[99]

        pixel_10000_from_av_sub_grid = (
            deflections_sub_grid[399]
            + deflections_sub_grid[398]
            + deflections_sub_grid[397]
            + deflections_sub_grid[396]
        ) / 4

        assert pixel_10000_reg_grid == pytest.approx(pixel_10000_from_av_sub_grid, 1e-4)


class TestConvergenceViajacobian(object):
    def test__compare_sis_convergence_via_jacobian_and_calculation__reg_grid(self):

        sis = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(20, 20), pixel_scale=0.05
        )

        convergence_via_calculation = sis.convergence_from_grid(
            grid=grid, return_in_2d=True, return_binned=True
        )

        convergence_via_jacobian = sis.convergence_from_jacobian(
            grid=grid, return_in_2d=True, return_binned=True
        )

        mean_error = np.mean(convergence_via_jacobian - convergence_via_calculation)

        assert convergence_via_jacobian.shape == (20, 20)
        assert mean_error < 1e-1

        convergence_via_calculation = sis.convergence_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        convergence_via_jacobian = sis.convergence_from_jacobian(
            grid=grid, return_in_2d=False, return_binned=True
        )

        mean_error = np.mean(convergence_via_jacobian - convergence_via_calculation)

        assert convergence_via_jacobian.shape == (400,)
        assert mean_error < 1e-1

    def test__compare_sie_at_phi_45__convergence_via_jacobian_and_calculation__reg_grid(
        self
    ):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=45.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(20, 20), pixel_scale=0.05
        )

        convergence_via_calculation = sie.convergence_from_grid(
            grid=grid, return_in_2d=True, return_binned=True
        )

        convergence_via_jacobian = sie.convergence_from_jacobian(
            grid=grid, return_in_2d=True, return_binned=True
        )

        mean_error = np.mean(convergence_via_jacobian - convergence_via_calculation)

        assert mean_error < 1e-1

    def test__convergence_sub_grid_binning(self):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(20, 20), pixel_scale=0.05, sub_grid_size=2
        )

        convergence_binned_reg_grid = sie.convergence_from_jacobian(
            grid=grid, return_in_2d=False, return_binned=True
        )

        convergence_sub_grid = sie.convergence_from_jacobian(
            grid=grid, return_in_2d=False, return_binned=False
        )

        pixel_1_reg_grid = convergence_binned_reg_grid[0]
        pixel_1_from_av_sub_grid = (
            convergence_sub_grid[0]
            + convergence_sub_grid[1]
            + convergence_sub_grid[2]
            + convergence_sub_grid[3]
        ) / 4

        assert pixel_1_reg_grid == pytest.approx(pixel_1_from_av_sub_grid, 1e-4)

        pixel_10000_reg_grid = convergence_binned_reg_grid[99]

        pixel_10000_from_av_sub_grid = (
            convergence_sub_grid[399]
            + convergence_sub_grid[398]
            + convergence_sub_grid[397]
            + convergence_sub_grid[396]
        ) / 4

        assert pixel_10000_reg_grid == pytest.approx(pixel_10000_from_av_sub_grid, 1e-4)

        convergence_via_calculation = sie.convergence_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        convergence_via_jacobian = sie.convergence_from_jacobian(
            grid=grid, return_in_2d=False, return_binned=True
        )

        mean_error = np.mean(convergence_via_jacobian - convergence_via_calculation)

        assert convergence_via_jacobian.shape == (400,)
        assert mean_error < 1e-1


class TestjacobianandMagnification(object):
    def test__jacobian_components__reg_grid(self):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(100, 100), pixel_scale=0.05
        )

        jacobian = sie.lensing_jacobian_from_grid(grid=grid, return_in_2d=False)

        A_12 = jacobian[0, 1]
        A_21 = jacobian[1, 0]

        mean_error = np.mean(A_12 - A_21)

        assert mean_error < 1e-4

    def test__jacobian_components__sub_grid(self):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(100, 100), pixel_scale=0.05, sub_grid_size=2
        )

        jacobian = sie.lensing_jacobian_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        A_12 = jacobian[0, 1]
        A_21 = jacobian[1, 0]

        mean_error = np.mean(A_12 - A_21)

        assert mean_error < 1e-4

    def test__compare_magnification_from_eigen_values_and_from_determinant__reg_grid(
        self
    ):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(100, 100), pixel_scale=0.05
        )

        magnification_via_determinant = sie.magnification_from_grid(
            grid=grid, return_in_2d=True
        )

        tangential_eigen_value = sie.tangential_eigen_value_from_shear_and_convergence(
            grid=grid, return_in_2d=True
        )

        radal_eigen_value = sie.radial_eigen_value_from_shear_and_convergence(
            grid=grid, return_in_2d=True
        )

        magnification_via_eigen_values = 1 / (
            tangential_eigen_value * radal_eigen_value
        )

        mean_error = np.mean(
            magnification_via_determinant - magnification_via_eigen_values
        )

        assert mean_error < 1e-4

    def test__compare_magnification_from_determinant_and_from_convergence_and_shear__reg_grid(
        self
    ):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(100, 100), pixel_scale=0.05
        )

        magnification_via_determinant = sie.magnification_from_grid(
            grid=grid, return_in_2d=True
        )

        convergence = sie.convergence_from_jacobian(grid=grid, return_in_2d=True)

        shear = sie.shear_from_jacobian(grid=grid, return_in_2d=True)

        magnification_via_convergence_and_shear = 1 / (
            (1 - convergence) ** 2 - shear ** 2
        )

        mean_error = np.mean(
            magnification_via_determinant - magnification_via_convergence_and_shear
        )

        assert mean_error < 1e-4

    def test__compare_magnification_from_eigen_values_and_from_determinant__sub_grid(
        self
    ):

        # TODO : Why does this test fail when we return the binned sub grid = True?

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(100, 100), pixel_scale=0.05, sub_grid_size=2
        )

        magnification_via_determinant = sie.magnification_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        tangential_eigen_value = sie.tangential_eigen_value_from_shear_and_convergence(
            grid=grid, return_in_2d=True, return_binned=False
        )

        radal_eigen_value = sie.radial_eigen_value_from_shear_and_convergence(
            grid=grid, return_in_2d=True, return_binned=False
        )

        magnification_via_eigen_values = 1 / (
            tangential_eigen_value * radal_eigen_value
        )

        mean_error = np.mean(
            magnification_via_determinant - magnification_via_eigen_values
        )

        assert mean_error < 1e-4

    def test__compare_magnification_from_determinant_and_from_convergence_and_shear__sub_grid(
        self
    ):

        # TODO : Why does this test fail when we return the binned sub grid = True?

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(100, 100), pixel_scale=0.05, sub_grid_size=2
        )

        magnification_via_determinant = sie.magnification_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        convergence = sie.convergence_from_jacobian(
            grid=grid, return_in_2d=True, return_binned=False
        )

        shear = sie.shear_from_jacobian(
            grid=grid, return_in_2d=True, return_binned=False
        )

        magnification_via_convergence_and_shear = 1 / (
            (1 - convergence) ** 2 - shear ** 2
        )

        mean_error = np.mean(
            magnification_via_determinant - magnification_via_convergence_and_shear
        )

        assert mean_error < 1e-4

    def test__jacobian_sub_grid_binning(self):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(10, 10), pixel_scale=0.05, sub_grid_size=2
        )

        jacobian_binned_reg_grid = sie.lensing_jacobian_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )
        a11_binned_reg_grid = jacobian_binned_reg_grid[0, 0]

        jacobian_sub_grid = sie.lensing_jacobian_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )
        a11_sub_grid = jacobian_sub_grid[0, 0]

        pixel_1_reg_grid = a11_binned_reg_grid[0]
        pixel_1_from_av_sub_grid = (
            a11_sub_grid[0] + a11_sub_grid[1] + a11_sub_grid[2] + a11_sub_grid[3]
        ) / 4

        assert jacobian_binned_reg_grid.shape == (2, 2, 100)
        assert jacobian_sub_grid.shape == (2, 2, 400)
        assert pixel_1_reg_grid == pytest.approx(pixel_1_from_av_sub_grid, 1e-4)

        pixel_10000_reg_grid = a11_binned_reg_grid[99]

        pixel_10000_from_av_sub_grid = (
            a11_sub_grid[399]
            + a11_sub_grid[398]
            + a11_sub_grid[397]
            + a11_sub_grid[396]
        ) / 4

        assert pixel_10000_reg_grid == pytest.approx(pixel_10000_from_av_sub_grid, 1e-4)

    def test_shear_sub_grid_binning(self):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(10, 10), pixel_scale=0.05, sub_grid_size=2
        )

        shear_binned_reg_grid = sie.shear_from_jacobian(
            grid=grid, return_in_2d=False, return_binned=True
        )

        shear_sub_grid = sie.shear_from_jacobian(
            grid=grid, return_in_2d=False, return_binned=False
        )

        pixel_1_reg_grid = shear_binned_reg_grid[0]
        pixel_1_from_av_sub_grid = (
            shear_sub_grid[0]
            + shear_sub_grid[1]
            + shear_sub_grid[2]
            + shear_sub_grid[3]
        ) / 4

        assert pixel_1_reg_grid == pytest.approx(pixel_1_from_av_sub_grid, 1e-4)

        pixel_10000_reg_grid = shear_binned_reg_grid[99]

        pixel_10000_from_av_sub_grid = (
            shear_sub_grid[399]
            + shear_sub_grid[398]
            + shear_sub_grid[397]
            + shear_sub_grid[396]
        ) / 4

        assert pixel_10000_reg_grid == pytest.approx(pixel_10000_from_av_sub_grid, 1e-4)

    def test_lambda_t_sub_grid_binning(self):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(10, 10), pixel_scale=0.05, sub_grid_size=2
        )

        lambda_t_binned_reg_grid = sie.tangential_eigen_value_from_shear_and_convergence(
            grid=grid, return_in_2d=False, return_binned=True
        )

        lambda_t_sub_grid = sie.tangential_eigen_value_from_shear_and_convergence(
            grid=grid, return_in_2d=False, return_binned=False
        )

        pixel_1_reg_grid = lambda_t_binned_reg_grid[0]
        pixel_1_from_av_sub_grid = (
            lambda_t_sub_grid[0]
            + lambda_t_sub_grid[1]
            + lambda_t_sub_grid[2]
            + lambda_t_sub_grid[3]
        ) / 4

        assert pixel_1_reg_grid == pytest.approx(pixel_1_from_av_sub_grid, 1e-4)

        pixel_10000_reg_grid = lambda_t_binned_reg_grid[99]

        pixel_10000_from_av_sub_grid = (
            lambda_t_sub_grid[399]
            + lambda_t_sub_grid[398]
            + lambda_t_sub_grid[397]
            + lambda_t_sub_grid[396]
        ) / 4

        assert pixel_10000_reg_grid == pytest.approx(pixel_10000_from_av_sub_grid, 1e-4)

    def test_lambda_r_sub_grid_binning(self):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(100, 100), pixel_scale=0.05, sub_grid_size=2
        )

        lambda_r_binned_reg_grid = sie.radial_eigen_value_from_shear_and_convergence(
            grid=grid, return_in_2d=False, return_binned=True
        )

        lambda_r_sub_grid = sie.radial_eigen_value_from_shear_and_convergence(
            grid=grid, return_in_2d=False, return_binned=False
        )

        pixel_1_reg_grid = lambda_r_binned_reg_grid[0]
        pixel_1_from_av_sub_grid = (
            lambda_r_sub_grid[0]
            + lambda_r_sub_grid[1]
            + lambda_r_sub_grid[2]
            + lambda_r_sub_grid[3]
        ) / 4

        assert pixel_1_reg_grid == pytest.approx(pixel_1_from_av_sub_grid, 1e-4)

        pixel_10000_reg_grid = lambda_r_binned_reg_grid[99]

        pixel_10000_from_av_sub_grid = (
            lambda_r_sub_grid[399]
            + lambda_r_sub_grid[398]
            + lambda_r_sub_grid[397]
            + lambda_r_sub_grid[396]
        ) / 4

        assert pixel_10000_reg_grid == pytest.approx(pixel_10000_from_av_sub_grid, 1e-4)


class TestCriticalCurvesandCaustics(object):

    # def test_compare_magnification_from_determinant_and_from_convergence_and_shear(self):
    #
    #     sie = mp.EllipticalIsothermal(centre=(0.0, 0.0), phi=0.0, axis_ratio=0.8, einstein_radius=2.0)
    #
    #     grid = grids.SubGrid.from_shape_pixel_scale_and_sub_grid_size(
    #         shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)
    #
    #     magnification_via_determinant = sie.magnification_from_grid(grid=grid)
    #
    #     convergence = sie.convergence_from_jacobian(grid=grid)
    #
    #     shear = sie.shear_from_jacobian(grid=grid)
    #
    #     magnification_via_convergence_and_shear = 1 / ((1 - convergence)**2 - shear**2)
    #
    #     mean_error = np.mean(magnification_via_determinant-magnification_via_convergence_and_shear)
    #
    #     assert mean_error < 1e-2

    def test__critical_curves_spherical_isothermal__reg_grid(self):

        sis = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(20, 20), pixel_scale=0.25, sub_grid_size=1
        )

        critical_curves = sis.critical_curves_from_grid(grid=grid)

        critical_curve_tangential = critical_curves[0]

        x_critical_tangential, y_critical_tangential = (
            critical_curve_tangential[:, 1],
            critical_curve_tangential[:, 0],
        )

        assert x_critical_tangential ** 2 + y_critical_tangential ** 2 == pytest.approx(
            sis.einstein_radius ** 2, 5e-1
        )

    def test__critical_curves_spherical_isothermal__sub_grid(self):

        sis = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(10, 10), pixel_scale=0.5, sub_grid_size=4
        )

        critical_curves = sis.critical_curves_from_grid(grid=grid)

        critical_curve_tangential = critical_curves[0]

        x_critical_tangential, y_critical_tangential = (
            critical_curve_tangential[:, 1],
            critical_curve_tangential[:, 0],
        )

        assert x_critical_tangential ** 2 + y_critical_tangential ** 2 == pytest.approx(
            sis.einstein_radius ** 2, 5e-1
        )

    def test__caustics_spherical_isothermal__sub_grid(self):

        ## testing caustics on regular grid does not pass as can be seen when visualised
        ## for an SIS the caustic does not correspond to the circle at the Einstein radius as it should
        ## this is because the radial ciritical curve should actually be a point

        sis = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(10, 10), pixel_scale=0.5, sub_grid_size=4
        )

        caustics = sis.caustics_from_grid(grid=grid)

        caustic_tangential = caustics[1]

        x_caustic_tangential, y_caustic_tangential = (
            caustic_tangential[:, 1],
            caustic_tangential[:, 0],
        )

        assert x_caustic_tangential ** 2 + y_caustic_tangential ** 2 == pytest.approx(
            sis.einstein_radius ** 2, 5e-1
        )

    def test__compare_tangential_critical_curves_from_magnification_and_lamda_t__reg_grid(
        self
    ):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2, axis_ratio=0.8, phi=40
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(20, 20), pixel_scale=0.25
        )

        critical_curves_from_magnification = sie.critical_curves_from_grid(grid=grid)
        critical_curve_tangential_from_magnification = critical_curves_from_magnification[
            0
        ]
        critical_curve_tangential_from_lambda_t = sie.tangential_critical_curve_from_grid(
            grid=grid
        )

        assert critical_curve_tangential_from_lambda_t == pytest.approx(
            critical_curve_tangential_from_magnification, 5e-1
        )

    def test__compare_tangential_critical_curves_from_magnification_and_lambda_t__sub_grid(
        self
    ):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2, axis_ratio=0.8, phi=40
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(10, 10), pixel_scale=0.5, sub_grid_size=2
        )

        critical_curves_from_magnification = sie.critical_curves_from_grid(grid=grid)
        critical_curve_tangential_from_magnification = critical_curves_from_magnification[
            0
        ]
        critical_curve_tangential_from_lambda_t = sie.tangential_critical_curve_from_grid(
            grid=grid
        )

        assert critical_curve_tangential_from_lambda_t == pytest.approx(
            critical_curve_tangential_from_magnification, 5e-1
        )

    # def test_compare_radial_critical_curves_from_magnification_and_lamda_r__reg_grid(self):

    #    sie = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=2, axis_ratio=0.8, phi=40)

    #    grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(shape=(100, 100), pixel_scale=0.05)

    #    critical_curves_from_mag = sie.critical_curves_from_grid(grid=grid)
    #    critical_curve_rad_from_mag = set(map(tuple, critical_curves_from_mag[1]))
    #    critical_curve_rad_from_lambda_t = set(map(tuple, sie.radial_critical_curve_from_grid(grid=grid)))

    #    assert critical_curve__rad_from_mag == pytest.approx(critical_curve_rad_from_lambda_t, 1e-4)

    ## trying to compare sets so that the order of x, y coordinates doesn't matter
    ## not passing because approx can't compare non numeric values???
    ## same for sub grid below

    # def test_compare_radial_critical_curves_from_magnification_and_lamda_r__sub_grid(self):

    #    sie = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=2, axis_ratio=0.8, phi=40)

    #   grid = grids.SubGrid.from_shape_pixel_scale_and_sub_grid_size(shape=(100, 100), pixel_scale=0.05,
    #                                                                  sub_grid_size=2)
    #   critical_curves_from_mag = sie.critical_curves_from_grid(grid=grid)
    #   critical_curve_rad_from_mag = set(map(tuple, critical_curves_from_mag[1]))
    #   critical_curve_rad_from_lambda_t = set(map(tuple, sie.radial_critical_curve_from_grid(grid=grid)))

    #   assert critical_curve__rad_from_mag == pytest.approx(critical_curve_rad_from_lambda_t, 1e-4)

    def test__compare_tangentialgential_caustic_from_magnification_and_lambda_t__reg_grid(
        self
    ):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2, axis_ratio=0.8, phi=40
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(20, 20), pixel_scale=0.25
        )

        caustics_from_magnification = sie.caustics_from_grid(grid=grid)
        caustic_tangential_from_magnification = caustics_from_magnification[0]
        caustic_tangential_from_lambda_t = sie.tangential_caustic_from_grid(grid=grid)

        assert caustic_tangential_from_lambda_t == pytest.approx(
            caustic_tangential_from_magnification, 5e-1
        )

    def test__compare_tangential_caustic_from_magnification_and_lambda_t__sub_grid(
        self
    ):

        sie = mp.EllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2, axis_ratio=0.8, phi=40
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(10, 10), pixel_scale=0.5, sub_grid_size=2
        )

        caustics_from_magnification = sie.caustics_from_grid(grid=grid)
        caustic_tangential_from_magnification = caustics_from_magnification[0]
        caustic_tangential_from_lambda_t = sie.tangential_caustic_from_grid(grid=grid)

        assert caustic_tangential_from_lambda_t == pytest.approx(
            caustic_tangential_from_magnification, 5e-1
        )


#  def test_compare_radial_caustic_from_magnification_and_lambda_r__reg_grid(self):

#    sie = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=2, axis_ratio=0.8, phi=40)

#    grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(shape=(100, 100), pixel_scale=0.05)

#     caustics_from_magnification = sie.caustics_from_grid(grid=grid)
#    caustic_rad_from_mag = set(map(tuple, caustics_from_magnification[1]))
#   caustic_rad_from_lambda_r = set(map( tuple, sie.radial_caustic_from_grid(grid=grid)))

#     assert caustic_rad_from_lambda_r == pytest.approx(caustic_rad_from_mag, 1e-1)

#  def test_compare_radial_caustic_from_magnification_and_lambda_r__sub_grid(self):

#     sie = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=2, axis_ratio=0.8, phi=40)

#    grid = grids.SubGrid.from_shape_pixel_scale_and_sub_grid_size(shape=(100, 100), pixel_scale=0.05,
#                                                                sub_grid_size=2)

#   caustics_from_magnification = sie.caustics_from_grid(grid=grid)

# caustic_rad_from_mag = set(map(tuple, caustics_from_magnification[1]))

# caustic_rad_from_lambda_r = set(map(tuple, sie.radial_caustic_from_grid(grid=grid)))

#     assert caustic_rad_from_lambda_r == pytest.approx(caustic_rad_from_mag, 1e-1)

import numpy as np
import pytest

from autolens import exc
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_and_mass_profiles as lmp, light_profiles as lp, mass_profiles as mp


@pytest.fixture(name="sersic_0")
def make_sersic_0():
    return lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0)


@pytest.fixture(name="sersic_1")
def make_sersic_1():
    return lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=2.0, effective_radius=0.4, sersic_index=2.0)


@pytest.fixture(name="gal_sersic_x1")
def make_gal_sersic_x1(sersic_0):
    return g.Galaxy(redshift=0.5, lp0=sersic_0)


@pytest.fixture(name="gal_sersic_x2")
def make_gal_sersic_x2(sersic_0, sersic_1):
    return g.Galaxy(redshift=0.5, lp0=sersic_0, lp1=sersic_1)


class TestLightProfiles(object):

    class TestIntensity:

        def test__one_profile_gal__intensity_is_same_individual_profile(self, sersic_0, gal_sersic_x1):
            sersic_intensity = sersic_0.intensities_from_grid(grid=np.array([[1.05, -0.55]]))

            gal_sersic_intensity = gal_sersic_x1.intensities_from_grid(np.array([[1.05, -0.55]]))

            assert sersic_intensity == gal_sersic_intensity

        def test__two_profile_gal__intensity_is_sum_of_individual_profiles(self, sersic_0, sersic_1, gal_sersic_x2):
            intensity = sersic_0.intensities_from_grid(np.array([[1.05, -0.55]]))
            intensity += sersic_1.intensities_from_grid(np.array([[1.05, -0.55]]))

            gal_intensity = gal_sersic_x2.intensities_from_grid(np.array([[1.05, -0.55]]))

            assert intensity == gal_intensity

    class TestLuminosityWithin:

        def test__circle__one_profile_gal__integral_is_same_as_individual_profile(self):

            sersic = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                         effective_radius=2.0,
                                         sersic_index=1.0)

            integral_radius = 5.5

            intensity_integral = sersic.luminosity_within_circle(radius=integral_radius)

            gal_sersic = g.Galaxy(redshift=0.5,
                                  light_profile_1=lp.EllipticalSersic(
                                      axis_ratio=1.0,
                                      phi=0.0,
                                      intensity=3.0,
                                      effective_radius=2.0,
                                      sersic_index=1.0))

            gal_intensity_integral = gal_sersic.luminosity_within_circle(radius=integral_radius)

            assert intensity_integral == gal_intensity_integral

        def test__circle__two_profile_gal__integral_is_sum_of_individual_profiles(self):
            sersic_1 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                           effective_radius=2.0,
                                           sersic_index=1.0)

            sersic_2 = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                           effective_radius=3.0,
                                           sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral = sersic_1.luminosity_within_circle(radius=integral_radius)
            intensity_integral += sersic_2.luminosity_within_circle(radius=integral_radius)

            gal_sersic = g.Galaxy(redshift=0.5,
                                  light_profile_1=lp.EllipticalSersic(
                                      axis_ratio=1.0,
                                      phi=0.0,
                                      intensity=3.0,
                                      effective_radius=2.0,
                                      sersic_index=1.0),
                                  light_profile_2=lp.EllipticalSersic(
                                      axis_ratio=0.5,
                                      phi=0.0,
                                      intensity=7.0,
                                      effective_radius=3.0,
                                      sersic_index=2.0))

            gal_intensity_integral = gal_sersic.luminosity_within_circle(radius=integral_radius)

            assert intensity_integral == gal_intensity_integral

        def test__same_as_above__include_critical_surface_mass_density(self):
            sersic_1 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                           effective_radius=2.0,
                                           sersic_index=1.0)

            sersic_2 = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                           effective_radius=3.0,
                                           sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral = sersic_1.luminosity_within_circle(radius=integral_radius, conversion_factor=2.0)
            intensity_integral += sersic_2.luminosity_within_circle(radius=integral_radius, conversion_factor=2.0)

            gal_sersic = g.Galaxy(redshift=0.5,
                                  light_profile_1=lp.EllipticalSersic(
                                      axis_ratio=1.0,
                                      phi=0.0,
                                      intensity=3.0,
                                      effective_radius=2.0,
                                      sersic_index=1.0),
                                  light_profile_2=lp.EllipticalSersic(
                                      axis_ratio=0.5,
                                      phi=0.0,
                                      intensity=7.0,
                                      effective_radius=3.0,
                                      sersic_index=2.0))

            gal_intensity_integral = gal_sersic.luminosity_within_circle(radius=integral_radius)
            assert intensity_integral == 2.0*gal_intensity_integral
            gal_intensity_integral = gal_sersic.luminosity_within_circle(radius=integral_radius, conversion_factor=2.0)
            assert intensity_integral == gal_intensity_integral

        def test__ellipse__one_profile_gal__integral_is_same_as_individual_profile(self):
            sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                         effective_radius=2.0,
                                         sersic_index=1.0)

            integral_radius = 0.5

            intensity_integral = sersic.luminosity_within_ellipse(major_axis=integral_radius)

            gal_sersic = g.Galaxy(redshift=0.5,
                                  light_profile_1=lp.EllipticalSersic(
                                      axis_ratio=0.5,
                                      phi=0.0,
                                      intensity=3.0,
                                      effective_radius=2.0,
                                      sersic_index=1.0))

            gal_intensity_integral = gal_sersic.luminosity_within_ellipse(major_axis=integral_radius)

            assert intensity_integral == gal_intensity_integral

        def test__ellipse__two_profile_gal__integral_is_sum_of_individual_profiles(self):
            sersic_1 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                           effective_radius=2.0,
                                           sersic_index=1.0)

            sersic_2 = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                           effective_radius=3.0,
                                           sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral = sersic_1.luminosity_within_ellipse(major_axis=integral_radius)
            intensity_integral += sersic_2.luminosity_within_ellipse(major_axis=integral_radius)

            gal_sersic = g.Galaxy(redshift=0.5,
                                  light_profile_1=lp.EllipticalSersic(
                                      axis_ratio=1.0,
                                      phi=0.0,
                                      intensity=3.0,
                                      effective_radius=2.0,
                                      sersic_index=1.0),
                                  light_profile_2=lp.EllipticalSersic(
                                      axis_ratio=0.5,
                                      phi=0.0,
                                      intensity=7.0,
                                      effective_radius=3.0,
                                      sersic_index=2.0))

            gal_intensity_integral = gal_sersic.luminosity_within_ellipse(major_axis=integral_radius)

            assert intensity_integral == gal_intensity_integral

        def test__ellipse__same_as_above__include_critical_surface_mass_density(self):
            sersic_1 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                           effective_radius=2.0,
                                           sersic_index=1.0)

            sersic_2 = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                           effective_radius=3.0,
                                           sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral = sersic_1.luminosity_within_ellipse(major_axis=integral_radius, conversion_factor=2.0)
            intensity_integral += sersic_2.luminosity_within_ellipse(major_axis=integral_radius, conversion_factor=2.0)

            gal_sersic = g.Galaxy(redshift=0.5,
                                  light_profile_1=lp.EllipticalSersic(
                                      axis_ratio=1.0,
                                      phi=0.0,
                                      intensity=3.0,
                                      effective_radius=2.0,
                                      sersic_index=1.0),
                                  light_profile_2=lp.EllipticalSersic(
                                      axis_ratio=0.5,
                                      phi=0.0,
                                      intensity=7.0,
                                      effective_radius=3.0,
                                      sersic_index=2.0))

            gal_intensity_integral = gal_sersic.luminosity_within_ellipse(major_axis=integral_radius)
            assert intensity_integral == 2.0*gal_intensity_integral
            gal_intensity_integral = gal_sersic.luminosity_within_ellipse(major_axis=integral_radius,
                                                                          conversion_factor=2.0)
            assert intensity_integral == gal_intensity_integral

        def test__no_light_profile__returns_none(self):

            gal = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal())

            assert gal.luminosity_within_circle(radius=1.0) == None
            assert gal.luminosity_within_ellipse(major_axis=1.0) == None

    class TestSymmetricProfiles(object):

        def test_1d_symmetry(self):
            sersic_1 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                           effective_radius=0.6,
                                           sersic_index=4.0)

            sersic_2 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                           effective_radius=0.6,
                                           sersic_index=4.0, centre=(100, 0))

            gal_sersic = g.Galaxy(redshift=0.5, light_profile_1=sersic_1, light_profile_2=sersic_2)

            assert gal_sersic.intensities_from_grid(
                np.array([[0.0, 0.0]])) == gal_sersic.intensities_from_grid(np.array([[100.0, 0.0]]))
            assert gal_sersic.intensities_from_grid(
                np.array([[49.0, 0.0]])) == gal_sersic.intensities_from_grid(np.array([[51.0, 0.0]]))

        def test_2d_symmetry(self):
            sersic_1 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                           effective_radius=0.6,
                                           sersic_index=4.0)

            sersic_2 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                           effective_radius=0.6,
                                           sersic_index=4.0, centre=(100, 0))

            sersic_3 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                           effective_radius=0.6,
                                           sersic_index=4.0, centre=(0, 100))

            sersic_4 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                           effective_radius=0.6,
                                           sersic_index=4.0, centre=(100, 100))

            gal_sersic = g.Galaxy(redshift=0.5, light_profile_1=sersic_1, light_profile_2=sersic_2,
                                  light_profile_3=sersic_3, light_profile_4=sersic_4)

            assert gal_sersic.intensities_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                gal_sersic.intensities_from_grid(np.array([[51.0, 0.0]])), 1e-5)

            assert gal_sersic.intensities_from_grid(np.array([[0.0, 49.0]])) == pytest.approx(
                gal_sersic.intensities_from_grid(np.array([[0.0, 51.0]])), 1e-5)

            assert gal_sersic.intensities_from_grid(np.array([[100.0, 49.0]])) == pytest.approx(
                gal_sersic.intensities_from_grid(np.array([[100.0, 51.0]])), 1e-5)

            assert gal_sersic.intensities_from_grid(np.array([[49.0, 49.0]])) == pytest.approx(
                gal_sersic.intensities_from_grid(np.array([[51.0, 51.0]])), 1e-5)


@pytest.fixture(name="sie_0")
def make_sie_0():
    return mp.EllipticalIsothermal(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)


@pytest.fixture(name="sie_1")
def make_sie_1():
    return mp.EllipticalIsothermal(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)


@pytest.fixture(name="gal_sie_x1")
def make_gal_sie_x1(sie_0):
    return g.Galaxy(redshift=0.5, mass_profile_1=sie_0)


@pytest.fixture(name="gal_sie_x2")
def make_gal_sie_x2(sie_0, sie_1):
    return g.Galaxy(redshift=0.5, mass_profile_1=sie_0, mass_profile_2=sie_1)


class TestMassProfiles(object):

    class TestConvergence:

        def test__one_profile_gal__convergence_is_same_individual_profile(self, sie_0, gal_sie_x1):
            sie_convergence = sie_0.convergence_from_grid(np.array([[1.05, -0.55]]))

            gal_sie_convergence = gal_sie_x1.convergence_from_grid(np.array([[1.05, -0.55]]))

            assert sie_convergence == gal_sie_convergence

        def test__two_profile_gal__convergence_is_sum_of_individual_profiles(self, sie_0, sie_1, gal_sie_x2):
            convergence = sie_0.convergence_from_grid(np.array([[1.05, -0.55]]))
            convergence += sie_1.convergence_from_grid(np.array([[1.05, -0.55]]))

            gal_convergence = gal_sie_x2.convergence_from_grid(np.array([[1.05, -0.55]]))

            assert convergence == gal_convergence

    class TestPotential:

        def test__one_profile_gal__potential_is_same_individual_profile(self, sie_0, gal_sie_x1):
            sie_potential = sie_0.potential_from_grid(np.array([[1.05, -0.55]]))

            gal_sie_potential = gal_sie_x1.potential_from_grid(np.array([[1.05, -0.55]]))

            assert sie_potential == gal_sie_potential

        def test__two_profile_gal__potential_is_sum_of_individual_profiles(self, sie_0, sie_1, gal_sie_x2):
            potential = sie_0.potential_from_grid(np.array([[1.05, -0.55]]))
            potential += sie_1.potential_from_grid(np.array([[1.05, -0.55]]))

            gal_potential = gal_sie_x2.potential_from_grid(np.array([[1.05, -0.55]]))

            assert potential == gal_potential

    class TestDeflectionAngles:

        def test__one_profile_gal__deflection_angles_is_same_individual_profile(self, sie_0, gal_sie_x1):
            sie_deflection_angles = sie_0.deflections_from_grid(np.array([[1.05, -0.55]]))

            gal_sie_deflection_angles = gal_sie_x1.deflections_from_grid(np.array([[1.05, -0.55]]))

            assert sie_deflection_angles[0, 0] == gal_sie_deflection_angles[0, 0]
            assert sie_deflection_angles[0, 1] == gal_sie_deflection_angles[0, 1]

        def test__two_profile_gal__deflection_angles_is_sum_of_individual_profiles(self, sie_0, sie_1, gal_sie_x2):
            deflection_angles_0 = sie_0.deflections_from_grid(np.array([[1.05, -0.55]]))
            deflection_angles_1 = sie_1.deflections_from_grid(np.array([[1.05, -0.55]]))

            deflection_angles = deflection_angles_0 + deflection_angles_1

            gal_deflection_angles = gal_sie_x2.deflections_from_grid(np.array([[1.05, -0.55]]))

            assert deflection_angles[0, 0] == gal_deflection_angles[0, 0]
            assert deflection_angles[0, 1] == gal_deflection_angles[0, 1]

    class TestMassWithin:

        def test__within_circle__no_critical_surface_mass_density__one_profile_gal__integral_is_same_as_individual_profile(self, sie_0, gal_sie_x1):
            integral_radius = 5.5

            mass_integral = sie_0.mass_within_circle_in_angular_units(radius=integral_radius)

            gal_mass_integral = gal_sie_x1.mass_within_circle_in_angular_units(radius=integral_radius)

            assert mass_integral == gal_mass_integral

        def test__within_circle__no_critical_surface_mass_density__two_profile_gal__integral_is_sum_of_individual_profiles(self):

            sie_0 = mp.EllipticalIsothermal(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_1 = mp.EllipticalIsothermal(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)

            integral_radius = 5.5

            mass_integral = sie_0.mass_within_circle_in_angular_units(radius=integral_radius)
            mass_integral += sie_1.mass_within_circle_in_angular_units(radius=integral_radius)

            gal_sie = g.Galaxy(redshift=0.5,
                               mass_profile_1=mp.EllipticalIsothermal(axis_ratio=0.8, phi=10.0,
                                                                      einstein_radius=1.0),
                               mass_profile_2=mp.EllipticalIsothermal(axis_ratio=0.6, phi=30.0,
                                                                      einstein_radius=1.2))

            gal_mass_integral = gal_sie.mass_within_circle_in_angular_units(radius=integral_radius)

            assert mass_integral == gal_mass_integral

        def test__same_as_above_but_physical_mass__uses_critical_surface_mass_density(self, sie_0, gal_sie_x1):

            integral_radius = 5.5

            mass_integral = sie_0.mass_within_circle_in_mass_units(radius=integral_radius, critical_surface_mass_density=2.0)

            gal_mass_integral = gal_sie_x1.mass_within_circle_in_mass_units(radius=integral_radius, critical_surface_mass_density=2.0)

            assert mass_integral == gal_mass_integral

        def test__within_ellipse__no_critical_surface_mass_density__one_profile_gal__integral_is_same_as_individual_profile(self, sie_0, gal_sie_x1):
            integral_radius = 0.5

            dimensionless_mass_integral = sie_0.mass_within_ellipse_in_angular_units(major_axis=integral_radius)

            gal_dimensionless_mass_integral = gal_sie_x1.mass_within_ellipse_in_angular_units(
                major_axis=integral_radius)

            assert dimensionless_mass_integral == gal_dimensionless_mass_integral

        def test__within_eliipse__no_critical_surface_mass_density__two_profile_gal__integral_is_sum_of_individual_profiles(self, sie_0, sie_1,
                                                                                          gal_sie_x2):
            integral_radius = 5.5

            dimensionless_mass_integral = sie_0.mass_within_ellipse_in_angular_units(major_axis=integral_radius)
            dimensionless_mass_integral += sie_1.mass_within_ellipse_in_angular_units(major_axis=integral_radius)

            gal_dimensionless_mass_integral = gal_sie_x2.mass_within_ellipse_in_angular_units(
                major_axis=integral_radius)

            assert dimensionless_mass_integral == gal_dimensionless_mass_integral

        def test__same_as_above_ellipse_but_physical_mass__uses_critical_surface_mass_density(self, sie_0, gal_sie_x1):
            integral_radius = 0.5

            dimensionless_mass_integral = sie_0.mass_within_ellipse_in_mass_units(major_axis=integral_radius,
                                                                                  critical_surface_mass_density=2.0)

            gal_dimensionless_mass_integral = gal_sie_x1.mass_within_ellipse_in_mass_units(major_axis=integral_radius,
                                                                                           critical_surface_mass_density=2.0)

            assert dimensionless_mass_integral == gal_dimensionless_mass_integral

        def test__no_mass_profile__returns_none(self):

            gal = g.Galaxy(redshift=0.5, light=lp.SphericalSersic())

            assert gal.mass_within_circle_in_mass_units(radius=1.0, critical_surface_mass_density=1.0) == None
            assert gal.mass_within_ellipse_in_mass_units(major_axis=1.0, critical_surface_mass_density=1.0) == None

    class TestSymmetricProfiles:

        def test_1d_symmetry(self):
            isothermal_1 = mp.EllipticalIsothermal(axis_ratio=0.5, phi=45.0,
                                                   einstein_radius=1.0)

            isothermal_2 = mp.EllipticalIsothermal(centre=(100, 0), axis_ratio=0.5, phi=45.0,
                                                   einstein_radius=1.0)

            gal_isothermal = g.Galaxy(redshift=0.5, mass_profile_1=isothermal_1, mass_profile_2=isothermal_2)

            assert gal_isothermal.convergence_from_grid(
                np.array([[1.0, 0.0]])) == gal_isothermal.convergence_from_grid(np.array([[99.0, 0.0]]))

            assert gal_isothermal.convergence_from_grid(
                np.array([[49.0, 0.0]])) == gal_isothermal.convergence_from_grid(np.array([[51.0, 0.0]]))

            assert gal_isothermal.potential_from_grid(np.array([[1.0, 0.0]])) == pytest.approx(
                gal_isothermal.potential_from_grid(np.array([[99.0, 0.0]])), 1e-6)

            assert gal_isothermal.potential_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                gal_isothermal.potential_from_grid(np.array([[51.0, 0.0]])), 1e-6)

            assert gal_isothermal.deflections_from_grid(np.array([[1.0, 0.0]])) == pytest.approx(
                gal_isothermal.deflections_from_grid(np.array([[99.0, 0.0]])), 1e-6)

            assert gal_isothermal.deflections_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                gal_isothermal.deflections_from_grid(np.array([[51.0, 0.0]])), 1e-6)

        def test_2d_symmetry(self):
            isothermal_1 = mp.SphericalIsothermal(einstein_radius=1.0)

            isothermal_2 = mp.SphericalIsothermal(centre=(100, 0), einstein_radius=1.0)

            isothermal_3 = mp.SphericalIsothermal(centre=(0, 100), einstein_radius=1.0)

            isothermal_4 = mp.SphericalIsothermal(centre=(100, 100), einstein_radius=1.0)

            gal_isothermal = g.Galaxy(redshift=0.5,
                                      mass_profile_1=isothermal_1, mass_profile_2=isothermal_2,
                                      mass_profile_3=isothermal_3, mass_profile_4=isothermal_4)

            assert gal_isothermal.convergence_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                gal_isothermal.convergence_from_grid(np.array([[51.0, 0.0]])), 1e-5)

            assert gal_isothermal.convergence_from_grid(np.array([[0.0, 49.0]])) == pytest.approx(
                gal_isothermal.convergence_from_grid(np.array([[0.0, 51.0]])), 1e-5)

            assert gal_isothermal.convergence_from_grid(np.array([[100.0, 49.0]])) == pytest.approx(
                gal_isothermal.convergence_from_grid(np.array([[100.0, 51.0]])), 1e-5)

            assert gal_isothermal.convergence_from_grid(np.array([[49.0, 49.0]])) == pytest.approx(
                gal_isothermal.convergence_from_grid(np.array([[51.0, 51.0]])), 1e-5)

            assert gal_isothermal.potential_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                gal_isothermal.potential_from_grid(np.array([[51.0, 0.0]])), 1e-5)

            assert gal_isothermal.potential_from_grid(np.array([[0.0, 49.0]])) == pytest.approx(
                gal_isothermal.potential_from_grid(np.array([[0.0, 51.0]])), 1e-5)

            assert gal_isothermal.potential_from_grid(np.array([[100.0, 49.0]])) == pytest.approx(
                gal_isothermal.potential_from_grid(np.array([[100.0, 51.0]])), 1e-5)

            assert gal_isothermal.potential_from_grid(np.array([[49.0, 49.0]])) == pytest.approx(
                gal_isothermal.potential_from_grid(np.array([[51.0, 51.0]])), 1e-5)

            assert -1.0 * gal_isothermal.deflections_from_grid(np.array([[49.0, 0.0]]))[0, 0] == pytest.approx(
                gal_isothermal.deflections_from_grid(np.array([[51.0, 0.0]]))[0, 0], 1e-5)

            assert 1.0 * gal_isothermal.deflections_from_grid(np.array([[0.0, 49.0]]))[0, 0] == pytest.approx(
                gal_isothermal.deflections_from_grid(np.array([[0.0, 51.0]]))[0, 0], 1e-5)

            assert 1.0 * gal_isothermal.deflections_from_grid(np.array([[100.0, 49.0]]))[0, 0] == pytest.approx(
                gal_isothermal.deflections_from_grid(np.array([[100.0, 51.0]]))[0, 0], 1e-5)

            assert -1.0 * gal_isothermal.deflections_from_grid(np.array([[49.0, 49.0]]))[0, 0] == pytest.approx(
                gal_isothermal.deflections_from_grid(np.array([[51.0, 51.0]]))[0, 0], 1e-5)

            assert 1.0 * gal_isothermal.deflections_from_grid(np.array([[49.0, 0.0]]))[0, 1] == pytest.approx(
                gal_isothermal.deflections_from_grid(np.array([[51.0, 0.0]]))[0, 1], 1e-5)

            assert -1.0 * gal_isothermal.deflections_from_grid(np.array([[0.0, 49.0]]))[0, 1] == pytest.approx(
                gal_isothermal.deflections_from_grid(np.array([[0.0, 51.0]]))[0, 1], 1e-5)

            assert -1.0 * gal_isothermal.deflections_from_grid(np.array([[100.0, 49.0]]))[0, 1] == pytest.approx(
                gal_isothermal.deflections_from_grid(np.array([[100.0, 51.0]]))[0, 1], 1e-5)

            assert -1.0 * gal_isothermal.deflections_from_grid(np.array([[49.0, 49.0]]))[0, 1] == pytest.approx(
                gal_isothermal.deflections_from_grid(np.array([[51.0, 51.0]]))[0, 1], 1e-5)

    class TestEinsteinRadii:

        def test__x2_sis_different_einstein_radii__einstein_radii_is_sum(self):

            sis_0 = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
            sis_1 = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=0.5)

            galaxy = g.Galaxy(mass_0=sis_0, mass_1=sis_1)

            assert galaxy.einstein_radius == 1.5

        def test__x2_si2_different_einstein_radii__einstein_radii_is_sum(self):

            sis_0 = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=0.5, axis_ratio=0.9)
            sis_1 = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0, axis_ratio=0.7)

            galaxy = g.Galaxy(mass_0=sis_0, mass_1=sis_1)

            assert galaxy.einstein_radius == 1.5

        def test__x2_nfw__einstein_radii_is_sum(self):

            nfw_0 = mp.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=0.8, kappa_s=0.2, scale_radius=5.0)
            nfw_1 = mp.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, kappa_s=0.3, scale_radius=10.0)

            galaxy = g.Galaxy(mass_0=nfw_0, mass_1=nfw_1)

            assert galaxy.einstein_radius == nfw_0.einstein_radius + nfw_1.einstein_radius

        def test__includes_shear__does_not_impact_values(self):

            sis_0 = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
            shear = mp.ExternalShear()

            galaxy = g.Galaxy(mass_0=sis_0, shear=shear)

            assert galaxy.einstein_radius == 1.0


class TestMassAndLightProfiles(object):

    @pytest.fixture(name="mass_and_light")
    def make_mass_and_light_profile(self):
        return lmp.EllipticalSersicRadialGradient()

    def test_single_profile(self, mass_and_light):
        gal = g.Galaxy(profile=mass_and_light)
        assert 1 == len(gal.light_profiles)
        assert 1 == len(gal.mass_profiles)
        assert gal.mass_profiles[0] == mass_and_light
        assert gal.light_profiles[0] == mass_and_light

    def test_multiple_profile(self, mass_and_light, sersic_0, sie_0):
        gal = g.Galaxy(profile=mass_and_light, light=sersic_0, sie=sie_0)
        assert 2 == len(gal.light_profiles)
        assert 2 == len(gal.mass_profiles)


class TestHyperGalaxy(object):

    class TestContributionMaps(object):

        def test__model_image_all_1s__factor_is_0__contributions_all_1s(self):
            gal_image = np.ones((3,))

            hyp = g.HyperGalaxy(contribution_factor=0.0)
            contributions = hyp.contributions_from_model_image_and_galaxy_image(model_image=gal_image,
                                                                                galaxy_image=gal_image,
                                                                                minimum_value=0.0)

            assert (contributions == np.ones((3,))).all()

        def test__different_values__factor_is_1__contributions_are_value_divided_by_factor_and_max(self):
            gal_image = np.array([0.5, 1.0, 1.5])

            hyp = g.HyperGalaxy(contribution_factor=1.0)
            contributions = hyp.contributions_from_model_image_and_galaxy_image(model_image=gal_image,
                                                                                galaxy_image=gal_image,
                                                                                minimum_value=0.0)

            assert (contributions == np.array([(0.5 / 1.5) / (1.5 / 2.5), (1.0 / 2.0) / (1.5 / 2.5), 1.0])).all()

        def test__different_values__threshold_is_1_minimum_threshold_included__wipes_1st_value_to_0(self):
            gal_image = np.array([0.5, 1.0, 1.5])

            hyp = g.HyperGalaxy(contribution_factor=1.0)
            contributions = hyp.contributions_from_model_image_and_galaxy_image(model_image=gal_image,
                                                                                galaxy_image=gal_image,
                                                                                minimum_value=0.6)

            assert (contributions == np.array([0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0])).all()

    class TestScaledNoise(object):

        def test__contribution_all_1s__noise_factor_2__noise_adds_double(self):
            noise = np.array([1.0, 2.0, 3.0])
            gal_contributions = np.ones((3, 1))

            hyp = g.HyperGalaxy(contribution_factor=0.0, noise_factor=2.0, noise_power=1.0)

            scaled_noise = hyp.hyper_noise_from_contributions(noise_map=noise, contributions=gal_contributions)

            assert (scaled_noise == np.array([2.0, 4.0, 6.0])).all()

        def test__same_as_above_but_contributions_vary(self):
            noise = np.array([1.0, 2.0, 3.0])
            gal_contributions = np.array([[0.0, 0.5, 1.0]])

            hyp = g.HyperGalaxy(contribution_factor=0.0, noise_factor=2.0, noise_power=1.0)

            scaled_noise = hyp.hyper_noise_from_contributions(noise_map=noise, contributions=gal_contributions)

            assert (scaled_noise == np.array([0.0, 2.0, 6.0])).all()

        def test__same_as_above_but_change_noise_scale_terms(self):
            noise = np.array([1.0, 2.0, 3.0])
            gal_contributions = np.array([[0.0, 0.5, 1.0]])

            hyp = g.HyperGalaxy(contribution_factor=0.0, noise_factor=2.0, noise_power=2.0)

            scaled_noise = hyp.hyper_noise_from_contributions(noise_map=noise, contributions=gal_contributions)

            assert (scaled_noise == np.array([0.0, 2.0, 18.0])).all()


class TestBooleanProperties(object):

    def test_has_profile(self):
        assert g.Galaxy().has_profile is False
        assert g.Galaxy(light_profile=lp.LightProfile()).has_profile is True
        assert g.Galaxy(mass_profile=mp.MassProfile()).has_profile is True

    def test_has_light_profile(self):
        assert g.Galaxy().has_light_profile is False
        assert g.Galaxy(light_profile=lp.LightProfile()).has_light_profile is True
        assert g.Galaxy(mass_profile=mp.MassProfile()).has_light_profile is False

    def test_has_mass_profile(self):
        assert g.Galaxy().has_mass_profile is False
        assert g.Galaxy(light_profile=lp.LightProfile()).has_mass_profile is False
        assert g.Galaxy(mass_profile=mp.MassProfile()).has_mass_profile is True

    def test_has_redshift(self):

        assert g.Galaxy().has_redshift is False
        assert g.Galaxy(light_profile=lp.LightProfile()).has_redshift is False
        assert g.Galaxy(redshift=0.1).has_redshift is True

    def test_has_pixelization(self):
        assert g.Galaxy().has_pixelization is False
        assert g.Galaxy(pixelization=object(), regularization=object()).has_pixelization is True

    def test_has_regularization(self):
        assert g.Galaxy().has_regularization is False
        assert g.Galaxy(pixelization=object(), regularization=object()).has_regularization is True

    def test_has_hyper_galaxy(self):
        assert g.Galaxy().has_pixelization is False
        assert g.Galaxy(hyper_galaxy=object()).has_hyper_galaxy is True

    def test__only_pixelization_raises_error(self):
        with pytest.raises(exc.GalaxyException):
            g.Galaxy(pixelization=object())

    def test__only_regularization_raises_error(self):
        with pytest.raises(exc.GalaxyException):
            g.Galaxy(regularization=object())

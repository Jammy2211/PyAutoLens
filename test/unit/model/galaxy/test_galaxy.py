import numpy as np
import pytest

from autolens import exc, dimensions as dim
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_and_mass_profiles as lmp, light_profiles as lp, mass_profiles as mp

from test.unit.mock.mock_cosmology import MockCosmology

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

        def test__one_profile_gal__intensity_is_same_individual_profile(self, sersic_0, gal_sersic_x1, sersic_1, gal_sersic_x2):

            sersic_intensity = sersic_0.intensities_from_grid(grid=np.array([[1.05, -0.55]]))

            gal_sersic_intensity = gal_sersic_x1.intensities_from_grid(np.array([[1.05, -0.55]]))

            assert sersic_intensity == gal_sersic_intensity

            intensity = sersic_0.intensities_from_grid(np.array([[1.05, -0.55]]))
            intensity += sersic_1.intensities_from_grid(np.array([[1.05, -0.55]]))

            gal_intensity = gal_sersic_x2.intensities_from_grid(np.array([[1.05, -0.55]]))

            assert intensity == gal_intensity

    class TestLuminosityWithin:

        def test__in_eps__two_profile_galaxy__is_sum_of_individual_profiles(self):

            sersic_0 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                           effective_radius=2.0, sersic_index=1.0)

            sersic_1 = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                           effective_radius=3.0, sersic_index=2.0)

            gal_sersic = g.Galaxy(redshift=0.5, light_profile_0=sersic_0, light_profile_1=sersic_1)

            radius = dim.Length(0.5, 'arcsec')

            luminosity = sersic_0.luminosity_within_circle_in_units(radius=radius, unit_luminosity='eps')
            luminosity += sersic_1.luminosity_within_circle_in_units(radius=radius, unit_luminosity='eps')

            gal_luminosity = gal_sersic.luminosity_within_circle_in_units(radius=radius, unit_luminosity='eps')

            assert luminosity == gal_luminosity

            luminosity = sersic_0.luminosity_within_ellipse_in_units(major_axis=radius,
                                                                     unit_luminosity='eps')
            luminosity += sersic_1.luminosity_within_ellipse_in_units(major_axis=radius,
                                                                      unit_luminosity='eps')

            gal_sersic = g.Galaxy(redshift=0.5, light_profile_0=sersic_0, light_profile_1=sersic_1)

            gal_luminosity = gal_sersic.luminosity_within_ellipse_in_units(major_axis=radius, unit_luminosity='eps')

            assert luminosity == gal_luminosity

        def test__radius_unit_conversions__multiply_by_kpc_per_arcsec(self):

            cosmology = MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0)

            sersic = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0, effective_radius=2.0,
                                           sersic_index=1.0)

            galaxy_arcsec = g.Galaxy(redshift=0.5, light_profile=sersic)

            radius = dim.Length(0.5, 'arcsec')

            luminosity = sersic.luminosity_within_circle_in_units(radius=radius)

            g_luminosity_arcsec = galaxy_arcsec.luminosity_within_circle_in_units(radius=radius)

            assert luminosity == g_luminosity_arcsec

            radius = dim.Length(0.5, 'kpc')

            luminosity = sersic.luminosity_within_circle_in_units(radius=radius, redshift_profile=0.5,
                                                                  cosmology=cosmology)

            g_luminosity_kpc = galaxy_arcsec.luminosity_within_circle_in_units(radius=radius, cosmology=cosmology)

            assert luminosity == g_luminosity_kpc

        def test__luminosity_unit_conversions__multiply_by_exposure_time(self):

            sersic = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0, effective_radius=2.0, sersic_index=1.0)

            galaxy = g.Galaxy(redshift=0.5, light_profile=sersic)

            radius = dim.Length(0.5, 'arcsec')

            luminosity = sersic.luminosity_within_ellipse_in_units(major_axis=radius, unit_luminosity='eps',

                                                                   exposure_time=2.0)
            gal_luminosity = galaxy.luminosity_within_ellipse_in_units(major_axis=radius, unit_luminosity='eps',
                                                                       exposure_time=2.0)
            assert luminosity == gal_luminosity

            luminosity = sersic.luminosity_within_circle_in_units(radius=radius, unit_luminosity='counts',
                                                                  exposure_time=2.0)

            gal_luminosity = galaxy.luminosity_within_circle_in_units(radius=radius, unit_luminosity='counts',
                                                                      exposure_time=2.0)
            assert luminosity == gal_luminosity

        def test__no_light_profile__returns_none(self):

            gal = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal())

            assert gal.luminosity_within_circle_in_units(radius=1.0) == None
            assert gal.luminosity_within_ellipse_in_units(major_axis=1.0) == None

    class TestSymmetricProfiles(object):

        def test_1d_symmetry(self):
            sersic_0 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                           effective_radius=0.6,
                                           sersic_index=4.0)

            sersic_1 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                           effective_radius=0.6,
                                           sersic_index=4.0, centre=(100, 0))

            gal_sersic = g.Galaxy(redshift=0.5, light_profile_0=sersic_0, light_profile_1=sersic_1)

            assert gal_sersic.intensities_from_grid(
                np.array([[0.0, 0.0]])) == gal_sersic.intensities_from_grid(np.array([[100.0, 0.0]]))
            assert gal_sersic.intensities_from_grid(
                np.array([[49.0, 0.0]])) == gal_sersic.intensities_from_grid(np.array([[51.0, 0.0]]))

        def test_2d_symmetry(self):
            sersic_0 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                           effective_radius=0.6,
                                           sersic_index=4.0)

            sersic_1 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                           effective_radius=0.6,
                                           sersic_index=4.0, centre=(100, 0))

            sersic_3 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                           effective_radius=0.6,
                                           sersic_index=4.0, centre=(0, 100))

            sersic_4 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                           effective_radius=0.6,
                                           sersic_index=4.0, centre=(100, 100))

            gal_sersic = g.Galaxy(redshift=0.5, light_profile_0=sersic_0, light_profile_1=sersic_1,
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
    return g.Galaxy(redshift=0.5, mass_profile_0=sie_0)


@pytest.fixture(name="gal_sie_x2")
def make_gal_sie_x2(sie_0, sie_1):
    return g.Galaxy(redshift=0.5, mass_profile_0=sie_0, mass_profile_1=sie_1)


class TestMassProfiles(object):

    class TestConvergence:

        def test__convergence_is_same_individual_profiles(self, sie_0, gal_sie_x1, sie_1, gal_sie_x2):
            
            sie_convergence = sie_0.convergence_from_grid(np.array([[1.05, -0.55]]))

            gal_sie_convergence = gal_sie_x1.convergence_from_grid(np.array([[1.05, -0.55]]))

            assert sie_convergence == gal_sie_convergence

            convergence = sie_0.convergence_from_grid(np.array([[1.05, -0.55]]))
            convergence += sie_1.convergence_from_grid(np.array([[1.05, -0.55]]))

            gal_convergence = gal_sie_x2.convergence_from_grid(np.array([[1.05, -0.55]]))

            assert convergence == gal_convergence

    class TestPotential:

        def test__potential_is_same_individual_profiles(self, sie_0, gal_sie_x1, sie_1, gal_sie_x2):

            sie_potential = sie_0.potential_from_grid(np.array([[1.05, -0.55]]))

            gal_sie_potential = gal_sie_x1.potential_from_grid(np.array([[1.05, -0.55]]))

            assert sie_potential == gal_sie_potential

            potential = sie_0.potential_from_grid(np.array([[1.05, -0.55]]))
            potential += sie_1.potential_from_grid(np.array([[1.05, -0.55]]))

            gal_potential = gal_sie_x2.potential_from_grid(np.array([[1.05, -0.55]]))

            assert potential == gal_potential

    class TestDeflectionAngles:

        def test__deflection_angles_same_as_individual_profiles(self, sie_0, gal_sie_x1, sie_1, gal_sie_x2):

            sie_deflection_angles = sie_0.deflections_from_grid(np.array([[1.05, -0.55]]))

            gal_sie_deflection_angles = gal_sie_x1.deflections_from_grid(np.array([[1.05, -0.55]]))

            assert sie_deflection_angles[0, 0] == gal_sie_deflection_angles[0, 0]
            assert sie_deflection_angles[0, 1] == gal_sie_deflection_angles[0, 1]

            deflection_angles_0 = sie_0.deflections_from_grid(np.array([[1.05, -0.55]]))
            deflection_angles_1 = sie_1.deflections_from_grid(np.array([[1.05, -0.55]]))

            deflection_angles = deflection_angles_0 + deflection_angles_1

            gal_deflection_angles = gal_sie_x2.deflections_from_grid(np.array([[1.05, -0.55]]))

            assert deflection_angles[0, 0] == gal_deflection_angles[0, 0]
            assert deflection_angles[0, 1] == gal_deflection_angles[0, 1]

    class TestMassWithin:

        def test__within_circle_in_angular_units__two_profile_gal__is_sum_of_individual_profiles(self):

            sie_0 = mp.EllipticalIsothermal(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_1 = mp.EllipticalIsothermal(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)

            radius = dim.Length(0.5, 'arcsec')

            mass = sie_0.mass_within_circle_in_units(radius=radius, unit_mass='angular')
            mass += sie_1.mass_within_circle_in_units(radius=radius, unit_mass='angular')

            gal_sie = g.Galaxy(redshift=0.5, mass_profile_0=sie_0, mass_profile_1=sie_1)

            gal_mass = gal_sie.mass_within_circle_in_units(radius=radius, unit_mass='angular')

            assert mass == gal_mass

            mass = sie_0.mass_within_ellipse_in_units(major_axis=radius, unit_mass='angular')
            mass += sie_1.mass_within_ellipse_in_units(major_axis=radius, unit_mass='angular')

            gal_mass = gal_sie.mass_within_ellipse_in_units(major_axis=radius, unit_mass='angular')

            assert mass == gal_mass

        def test__radius_unit_conversions__multiply_by_kpc_per_arcsec(self):

            cosmology = MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=1.0)

            sie = mp.EllipticalIsothermal(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)

            galaxy_arcsec = g.Galaxy(redshift=0.5, mass_profile=sie)

            radius = dim.Length(0.5, 'arcsec')

            mass = sie.mass_within_circle_in_units(radius=radius, unit_mass='solMass', redshift_profile=0.5,
                                                   redshift_source=1.0, cosmology=cosmology)

            g_mass_arcsec = galaxy_arcsec.mass_within_circle_in_units(radius=radius, unit_mass='solMass',
                                                                      redshift_source=1.0, cosmology=cosmology)
            assert mass == g_mass_arcsec

            radius = dim.Length(0.5, 'kpc')

            mass = sie.mass_within_circle_in_units(radius=radius, unit_mass='solMass', redshift_profile=0.5,
                                                   redshift_source=1.0, cosmology=cosmology)

            g_mass_kpc = galaxy_arcsec.mass_within_circle_in_units(radius=radius, unit_mass='solMass',
                                                                   redshift_source=1.0, cosmology=cosmology)
            assert mass == g_mass_kpc

        def test__mass_unit_conversions__same_as_individual_profile(self):

            cosmology = MockCosmology(arcsec_per_kpc=1.0, kpc_per_arcsec=1.0, critical_surface_density=2.0)

            sie = mp.EllipticalIsothermal(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)

            galaxy = g.Galaxy(redshift=0.5, mass_profile=sie)

            radius = dim.Length(0.5, 'arcsec')

            mass = sie.mass_within_ellipse_in_units(major_axis=radius, unit_mass='angular',
                                                    redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology)

            gal_mass = galaxy.mass_within_ellipse_in_units(major_axis=radius, unit_mass='angular',
                                                           redshift_source=1.0, cosmology=cosmology)
            assert mass == gal_mass

            mass = sie.mass_within_circle_in_units(radius=radius, unit_mass='solMass',
                                                   redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology)

            gal_mass = galaxy.mass_within_circle_in_units(radius=radius, unit_mass='solMass',
                                                        redshift_source=1.0, cosmology=cosmology)
            assert mass == gal_mass

        def test__no_mass_profile__returns_none(self):

            gal = g.Galaxy(redshift=0.5, light=lp.SphericalSersic())

            assert gal.mass_within_circle_in_units(radius=1.0, critical_surface_density=1.0) == None
            assert gal.mass_within_ellipse_in_units(major_axis=1.0, critical_surface_density=1.0) == None

    class TestSymmetricProfiles:

        def test_1d_symmetry(self):
            isothermal_1 = mp.EllipticalIsothermal(axis_ratio=0.5, phi=45.0,
                                                   einstein_radius=1.0)

            isothermal_2 = mp.EllipticalIsothermal(centre=(100, 0), axis_ratio=0.5, phi=45.0,
                                                   einstein_radius=1.0)

            gal_isothermal = g.Galaxy(redshift=0.5, mass_profile_0=isothermal_1, mass_profile_1=isothermal_2)

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
                                      mass_profile_0=isothermal_1, mass_profile_1=isothermal_2,
                                      mass_profile_2=isothermal_3, mass_profile_3=isothermal_4)

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

    class TestEinsteinRadiiMass:

        def test__x2_sis_different_einstein_radii_and_mass__einstein_radii_and_mass_are_sum(self):

            sis_0 = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
            sis_1 = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=0.5)

            galaxy = g.Galaxy(mass_0=sis_0, mass_1=sis_1)

            assert galaxy.einstein_radius_in_units(unit_length='arcsec') == 1.5
            assert galaxy.einstein_mass_in_units(unit_mass='angular') == np.pi*(1.0 + 0.5**2.0)

        def test__includes_shear__does_not_impact_values(self):

            sis_0 = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
            shear = mp.ExternalShear()

            galaxy = g.Galaxy(mass_0=sis_0, shear=shear)

            assert galaxy.einstein_radius_in_units(unit_length='arcsec') == 1.0
            assert galaxy.einstein_mass_in_units(unit_mass='angular') == np.pi


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


class TestSummarizeInUnits(object):

    def test__galaxy_with_two_light_and_mass_profiles(self):
    
        sersic_0 = lp.SphericalSersic(intensity=1.0, effective_radius=2.0, sersic_index=2.0)
        sersic_1 = lp.SphericalSersic(intensity=2.0, effective_radius=2.0, sersic_index=2.0)


        sis_0 = mp.SphericalIsothermal(einstein_radius=1.0)
        sis_1 = mp.SphericalIsothermal(einstein_radius=2.0)

        gal = g.Galaxy(redshift=0.5, light_profile_0=sersic_0, light_profile_1=sersic_1, 
                       mass_profile_0=sis_0, mass_profile_1=sis_1)


        summary_text = gal.summarize_in_units(radii=[dim.Length(10.0), dim.Length(500.0)], whitespace=50,
                                              unit_length='arcsec', unit_luminosity='eps', unit_mass='angular')

        index = 0

        assert summary_text[index] == 'Galaxy' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] ==  'redshift                                          0.50' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] ==  'GALAXY LIGHT' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] == 'luminosity_within_10.00_arcsec                    1.8854e+02 eps' ; index += 1
        assert summary_text[index] == 'luminosity_within_500.00_arcsec                   1.9573e+02 eps' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] ==  'LIGHT PROFILES:' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] == 'Light Profile = SphericalSersic' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] == 'luminosity_within_10.00_arcsec                    6.2848e+01 eps' ; index += 1
        assert summary_text[index] == 'luminosity_within_500.00_arcsec                   6.5243e+01 eps' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] == 'Light Profile = SphericalSersic' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] == 'luminosity_within_10.00_arcsec                    1.2570e+02 eps' ; index += 1
        assert summary_text[index] == 'luminosity_within_500.00_arcsec                   1.3049e+02 eps' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] ==  'GALAXY MASS' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] == 'einstein_radius                                   3.00 arcsec' ; index += 1
        assert summary_text[index] == 'einstein_mass                                     1.5708e+01 angular' ; index += 1
        assert summary_text[index] == 'mass_within_10.00_arcsec                          9.4248e+01 angular' ; index += 1
        assert summary_text[index] == 'mass_within_500.00_arcsec                         4.7124e+03 angular' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] ==  'MASS PROFILES:' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] == 'Mass Profile = SphericalIsothermal' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] == 'einstein_radius                                   1.00 arcsec' ; index += 1
        assert summary_text[index] == 'einstein_mass                                     3.1416e+00 angular' ; index += 1
        assert summary_text[index] == 'mass_within_10.00_arcsec                          3.1416e+01 angular' ; index += 1
        assert summary_text[index] == 'mass_within_500.00_arcsec                         1.5708e+03 angular' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] == 'Mass Profile = SphericalIsothermal' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] == 'einstein_radius                                   2.00 arcsec' ; index += 1
        assert summary_text[index] == 'einstein_mass                                     1.2566e+01 angular' ; index += 1
        assert summary_text[index] == 'mass_within_10.00_arcsec                          6.2832e+01 angular' ; index += 1
        assert summary_text[index] == 'mass_within_500.00_arcsec                         3.1416e+03 angular' ; index += 1

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


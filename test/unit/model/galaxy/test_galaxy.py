import numpy as np
import pytest

from autolens import exc, dimensions as dim
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

from test.unit.mock.model import mock_cosmology

class TestLightProfiles(object):

    class TestIntensity:

        def test__galaxies_with_x1_and_x2_light_profiles__intensities_is_sum_of_profiles(self, lp_0, lp_1,
                                                                                         gal_x1_lp, gal_x2_lp):

            lp_intensity = lp_0.intensities_from_grid(grid=np.array([[1.05, -0.55]]))

            gal_lp_intensity = gal_x1_lp.intensities_from_grid(np.array([[1.05, -0.55]]))

            assert lp_intensity == gal_lp_intensity

            intensity = lp_0.intensities_from_grid(np.array([[1.05, -0.55]]))
            intensity += lp_1.intensities_from_grid(np.array([[1.05, -0.55]]))

            gal_intensity = gal_x2_lp.intensities_from_grid(np.array([[1.05, -0.55]]))

            assert intensity == gal_intensity

    class TestLuminosityWithin:

        def test__two_profile_galaxy__is_sum_of_individual_profiles(self, lp_0, lp_1, gal_x1_lp, gal_x2_lp):

            radius = dim.Length(0.5, 'arcsec')

            lp_luminosity = lp_0.luminosity_within_circle_in_units(radius=radius, unit_luminosity='eps')
            gal_luminosity = gal_x1_lp.luminosity_within_circle_in_units(radius=radius, unit_luminosity='eps')

            assert lp_luminosity == gal_luminosity

            lp_luminosity = lp_0.luminosity_within_ellipse_in_units(major_axis=radius, unit_luminosity='eps')
            lp_luminosity += lp_1.luminosity_within_ellipse_in_units(major_axis=radius, unit_luminosity='eps')

            gal_luminosity = gal_x2_lp.luminosity_within_ellipse_in_units(major_axis=radius, unit_luminosity='eps')

            assert lp_luminosity == gal_luminosity

        def test__radius_unit_conversions__multiply_by_kpc_per_arcsec(self, lp_0, gal_x1_lp):

            cosmology = mock_cosmology.MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0)

            radius = dim.Length(0.5, 'arcsec')

            lp_luminosity_arcsec = lp_0.luminosity_within_circle_in_units(radius=radius)
            gal_luminosity_arcsec = gal_x1_lp.luminosity_within_circle_in_units(radius=radius)

            assert lp_luminosity_arcsec == gal_luminosity_arcsec

            radius = dim.Length(0.5, 'kpc')

            lp_luminosity_kpc = lp_0.luminosity_within_circle_in_units(
                radius=radius, redshift_profile=0.5,cosmology=cosmology)
            gal_luminosity_kpc = gal_x1_lp.luminosity_within_circle_in_units(radius=radius, cosmology=cosmology)

            assert lp_luminosity_kpc == gal_luminosity_kpc

        def test__luminosity_unit_conversions__multiply_by_exposure_time(self, lp_0, gal_x1_lp):

            radius = dim.Length(0.5, 'arcsec')

            lp_luminosity_eps = lp_0.luminosity_within_ellipse_in_units(
                major_axis=radius, unit_luminosity='eps', exposure_time=2.0)
            gal_luminosity_eps = gal_x1_lp.luminosity_within_ellipse_in_units(
                major_axis=radius, unit_luminosity='eps',exposure_time=2.0)

            assert lp_luminosity_eps == gal_luminosity_eps

            lp_luminosity_counts = lp_0.luminosity_within_circle_in_units(
                radius=radius, unit_luminosity='counts', exposure_time=2.0)

            gal_luminosity_counts = gal_x1_lp.luminosity_within_circle_in_units(
                radius=radius, unit_luminosity='counts', exposure_time=2.0)

            assert lp_luminosity_counts == gal_luminosity_counts

        def test__no_light_profile__returns_none(self):

            gal_no_lp = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal())

            assert gal_no_lp.luminosity_within_circle_in_units(radius=1.0) == None
            assert gal_no_lp.luminosity_within_ellipse_in_units(major_axis=1.0) == None

    class TestSymmetricProfiles(object):

        def test_1d_symmetry(self):

            lp_0 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0)

            lp_1 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0,
                                       centre=(100, 0))

            gal_x2_lp = g.Galaxy(redshift=0.5, light_profile_0=lp_0, light_profile_1=lp_1)

            assert gal_x2_lp.intensities_from_grid(
                np.array([[0.0, 0.0]])) == gal_x2_lp.intensities_from_grid(np.array([[100.0, 0.0]]))
            assert gal_x2_lp.intensities_from_grid(
                np.array([[49.0, 0.0]])) == gal_x2_lp.intensities_from_grid(np.array([[51.0, 0.0]]))

        def test_2d_symmetry(self):

            lp_0 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,effective_radius=0.6, sersic_index=4.0)

            lp_1 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0,
                                       centre=(100, 0))

            lp_2 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0,
                                       centre=(0, 100))

            lp_3 = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0,
                                       centre=(100, 100))

            gal_x4_lp = g.Galaxy(redshift=0.5, light_profile_0=lp_0, light_profile_1=lp_1,
                                  light_profile_3=lp_2, light_profile_4=lp_3)

            assert gal_x4_lp.intensities_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                gal_x4_lp.intensities_from_grid(np.array([[51.0, 0.0]])), 1e-5)

            assert gal_x4_lp.intensities_from_grid(np.array([[0.0, 49.0]])) == pytest.approx(
                gal_x4_lp.intensities_from_grid(np.array([[0.0, 51.0]])), 1e-5)

            assert gal_x4_lp.intensities_from_grid(np.array([[100.0, 49.0]])) == pytest.approx(
                gal_x4_lp.intensities_from_grid(np.array([[100.0, 51.0]])), 1e-5)

            assert gal_x4_lp.intensities_from_grid(np.array([[49.0, 49.0]])) == pytest.approx(
                gal_x4_lp.intensities_from_grid(np.array([[51.0, 51.0]])), 1e-5)


class TestMassProfiles(object):

    class TestConvergence:

        def test__galaxies_with_x1_and_x2_mass_profiles__convergence_is_same_individual_profiles(self, mp_0, gal_x1_mp,
                                                                                                 mp_1, gal_x2_mp):

            mp_convergence = mp_0.convergence_from_grid(np.array([[1.05, -0.55]]))

            gal_mp_convergence = gal_x1_mp.convergence_from_grid(np.array([[1.05, -0.55]]))

            assert mp_convergence == gal_mp_convergence

            mp_convergence = mp_0.convergence_from_grid(np.array([[1.05, -0.55]]))
            mp_convergence += mp_1.convergence_from_grid(np.array([[1.05, -0.55]]))

            gal_convergence = gal_x2_mp.convergence_from_grid(np.array([[1.05, -0.55]]))

            assert mp_convergence == gal_convergence

    class TestPotential:

        def test__galaxies_with_x1_and_x2_mass_profiles__potential_is_same_individual_profiles(self, mp_0, gal_x1_mp,
                                                                                               mp_1, gal_x2_mp):

            mp_potential = mp_0.potential_from_grid(np.array([[1.05, -0.55]]))

            gal_mp_potential = gal_x1_mp.potential_from_grid(np.array([[1.05, -0.55]]))

            assert mp_potential == gal_mp_potential

            mp_potential = mp_0.potential_from_grid(np.array([[1.05, -0.55]]))
            mp_potential += mp_1.potential_from_grid(np.array([[1.05, -0.55]]))

            gal_potential = gal_x2_mp.potential_from_grid(np.array([[1.05, -0.55]]))

            assert mp_potential == gal_potential

    class TestDeflectionAngles:

        def test__galaxies_with_x1_and_x2_mass_profiles__deflection_same_as_individual_profiles(self, mp_0, gal_x1_mp,
                                                                                                mp_1, gal_x2_mp):

            mp_deflections = mp_0.deflections_from_grid(np.array([[1.05, -0.55]]))

            gal_mp_deflections = gal_x1_mp.deflections_from_grid(np.array([[1.05, -0.55]]))

            assert mp_deflections[0, 0] == gal_mp_deflections[0, 0]
            assert mp_deflections[0, 1] == gal_mp_deflections[0, 1]

            mp_deflections_0 = mp_0.deflections_from_grid(np.array([[1.05, -0.55]]))
            mp_deflections_1 = mp_1.deflections_from_grid(np.array([[1.05, -0.55]]))

            mp_deflections = mp_deflections_0 + mp_deflections_1

            gal_mp_deflections = gal_x2_mp.deflections_from_grid(np.array([[1.05, -0.55]]))

            assert mp_deflections[0, 0] == gal_mp_deflections[0, 0]
            assert mp_deflections[0, 1] == gal_mp_deflections[0, 1]

    class TestMassWithin:

        def test__two_profile_galaxy__is_sum_of_individual_profiles(self, mp_0, gal_x1_mp, mp_1, gal_x2_mp):

            radius = dim.Length(0.5, 'arcsec')

            mp_mass = mp_0.mass_within_circle_in_units(radius=radius, unit_mass='angular')

            gal_mass = gal_x1_mp.mass_within_circle_in_units(radius=radius, unit_mass='angular')

            assert mp_mass == gal_mass

            mp_mass = mp_0.mass_within_ellipse_in_units(major_axis=radius, unit_mass='angular')
            mp_mass += mp_1.mass_within_ellipse_in_units(major_axis=radius, unit_mass='angular')

            gal_mass = gal_x2_mp.mass_within_ellipse_in_units(major_axis=radius, unit_mass='angular')

            assert mp_mass == gal_mass

        def test__radius_unit_conversions__multiply_by_kpc_per_arcsec(self, mp_0, gal_x1_mp):

            cosmology = mock_cosmology.MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=1.0)

            radius = dim.Length(0.5, 'arcsec')

            mp_mass_arcsec = mp_0.mass_within_circle_in_units(radius=radius, unit_mass='solMass', redshift_profile=0.5,
                                                   redshift_source=1.0, cosmology=cosmology)

            gal_mass_arcsec = gal_x1_mp.mass_within_circle_in_units(radius=radius, unit_mass='solMass',
                                                                      redshift_source=1.0, cosmology=cosmology)
            assert mp_mass_arcsec == gal_mass_arcsec

            radius = dim.Length(0.5, 'kpc')

            mp_mass_kpc = mp_0.mass_within_circle_in_units(radius=radius, unit_mass='solMass', redshift_profile=0.5,
                                                   redshift_source=1.0, cosmology=cosmology)

            gal_mass_kpc = gal_x1_mp.mass_within_circle_in_units(radius=radius, unit_mass='solMass',
                                                                   redshift_source=1.0, cosmology=cosmology)
            assert mp_mass_kpc == gal_mass_kpc

        def test__mass_unit_conversions__same_as_individual_profile(self, mp_0, gal_x1_mp):

            cosmology = mock_cosmology.MockCosmology(arcsec_per_kpc=1.0, kpc_per_arcsec=1.0,
                                                     critical_surface_density=2.0)

            radius = dim.Length(0.5, 'arcsec')

            mp_mass_angular = mp_0.mass_within_ellipse_in_units(major_axis=radius, unit_mass='angular',
                                                    redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology)

            gal_mass_angular = gal_x1_mp.mass_within_ellipse_in_units(major_axis=radius, unit_mass='angular',
                                                           redshift_source=1.0, cosmology=cosmology)
            assert mp_mass_angular == gal_mass_angular

            mp_mass_sol = mp_0.mass_within_circle_in_units(radius=radius, unit_mass='solMass',
                                                   redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology)

            gal_mass_sol = gal_x1_mp.mass_within_circle_in_units(radius=radius, unit_mass='solMass',
                                                        redshift_source=1.0, cosmology=cosmology)
            assert mp_mass_sol == gal_mass_sol

        def test__no_mass_profile__returns_none(self):

            gal_no_mp = g.Galaxy(redshift=0.5, light=lp.SphericalSersic())

            assert gal_no_mp.mass_within_circle_in_units(radius=1.0, critical_surface_density=1.0) == None
            assert gal_no_mp.mass_within_ellipse_in_units(major_axis=1.0, critical_surface_density=1.0) == None

    class TestSymmetricProfiles:

        def test_1d_symmetry(self):

            mp_0 = mp.EllipticalIsothermal(axis_ratio=0.5, phi=45.0,
                                                   einstein_radius=1.0)

            mp_1 = mp.EllipticalIsothermal(centre=(100, 0), axis_ratio=0.5, phi=45.0,
                                                   einstein_radius=1.0)

            gal_x4_mp = g.Galaxy(redshift=0.5, mass_profile_0=mp_0, mass_profile_1=mp_1)

            assert gal_x4_mp.convergence_from_grid(
                np.array([[1.0, 0.0]])) == gal_x4_mp.convergence_from_grid(np.array([[99.0, 0.0]]))

            assert gal_x4_mp.convergence_from_grid(
                np.array([[49.0, 0.0]])) == gal_x4_mp.convergence_from_grid(np.array([[51.0, 0.0]]))

            assert gal_x4_mp.potential_from_grid(np.array([[1.0, 0.0]])) == pytest.approx(
                gal_x4_mp.potential_from_grid(np.array([[99.0, 0.0]])), 1e-6)

            assert gal_x4_mp.potential_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                gal_x4_mp.potential_from_grid(np.array([[51.0, 0.0]])), 1e-6)

            assert gal_x4_mp.deflections_from_grid(np.array([[1.0, 0.0]])) == pytest.approx(
                gal_x4_mp.deflections_from_grid(np.array([[99.0, 0.0]])), 1e-6)

            assert gal_x4_mp.deflections_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                gal_x4_mp.deflections_from_grid(np.array([[51.0, 0.0]])), 1e-6)

        def test_2d_symmetry(self):

            mp_0 = mp.SphericalIsothermal(einstein_radius=1.0)

            mp_1 = mp.SphericalIsothermal(centre=(100, 0), einstein_radius=1.0)

            mp_2 = mp.SphericalIsothermal(centre=(0, 100), einstein_radius=1.0)

            mp_3 = mp.SphericalIsothermal(centre=(100, 100), einstein_radius=1.0)

            gal_x4_mp = g.Galaxy(redshift=0.5,
                                      mass_profile_0=mp_0, mass_profile_1=mp_1,
                                      mass_profile_2=mp_2, mass_profile_3=mp_3)

            assert gal_x4_mp.convergence_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                gal_x4_mp.convergence_from_grid(np.array([[51.0, 0.0]])), 1e-5)

            assert gal_x4_mp.convergence_from_grid(np.array([[0.0, 49.0]])) == pytest.approx(
                gal_x4_mp.convergence_from_grid(np.array([[0.0, 51.0]])), 1e-5)

            assert gal_x4_mp.convergence_from_grid(np.array([[100.0, 49.0]])) == pytest.approx(
                gal_x4_mp.convergence_from_grid(np.array([[100.0, 51.0]])), 1e-5)

            assert gal_x4_mp.convergence_from_grid(np.array([[49.0, 49.0]])) == pytest.approx(
                gal_x4_mp.convergence_from_grid(np.array([[51.0, 51.0]])), 1e-5)

            assert gal_x4_mp.potential_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                gal_x4_mp.potential_from_grid(np.array([[51.0, 0.0]])), 1e-5)

            assert gal_x4_mp.potential_from_grid(np.array([[0.0, 49.0]])) == pytest.approx(
                gal_x4_mp.potential_from_grid(np.array([[0.0, 51.0]])), 1e-5)

            assert gal_x4_mp.potential_from_grid(np.array([[100.0, 49.0]])) == pytest.approx(
                gal_x4_mp.potential_from_grid(np.array([[100.0, 51.0]])), 1e-5)

            assert gal_x4_mp.potential_from_grid(np.array([[49.0, 49.0]])) == pytest.approx(
                gal_x4_mp.potential_from_grid(np.array([[51.0, 51.0]])), 1e-5)

            assert -1.0 * gal_x4_mp.deflections_from_grid(np.array([[49.0, 0.0]]))[0, 0] == pytest.approx(
                gal_x4_mp.deflections_from_grid(np.array([[51.0, 0.0]]))[0, 0], 1e-5)

            assert 1.0 * gal_x4_mp.deflections_from_grid(np.array([[0.0, 49.0]]))[0, 0] == pytest.approx(
                gal_x4_mp.deflections_from_grid(np.array([[0.0, 51.0]]))[0, 0], 1e-5)

            assert 1.0 * gal_x4_mp.deflections_from_grid(np.array([[100.0, 49.0]]))[0, 0] == pytest.approx(
                gal_x4_mp.deflections_from_grid(np.array([[100.0, 51.0]]))[0, 0], 1e-5)

            assert -1.0 * gal_x4_mp.deflections_from_grid(np.array([[49.0, 49.0]]))[0, 0] == pytest.approx(
                gal_x4_mp.deflections_from_grid(np.array([[51.0, 51.0]]))[0, 0], 1e-5)

            assert 1.0 * gal_x4_mp.deflections_from_grid(np.array([[49.0, 0.0]]))[0, 1] == pytest.approx(
                gal_x4_mp.deflections_from_grid(np.array([[51.0, 0.0]]))[0, 1], 1e-5)

            assert -1.0 * gal_x4_mp.deflections_from_grid(np.array([[0.0, 49.0]]))[0, 1] == pytest.approx(
                gal_x4_mp.deflections_from_grid(np.array([[0.0, 51.0]]))[0, 1], 1e-5)

            assert -1.0 * gal_x4_mp.deflections_from_grid(np.array([[100.0, 49.0]]))[0, 1] == pytest.approx(
                gal_x4_mp.deflections_from_grid(np.array([[100.0, 51.0]]))[0, 1], 1e-5)

            assert -1.0 * gal_x4_mp.deflections_from_grid(np.array([[49.0, 49.0]]))[0, 1] == pytest.approx(
                gal_x4_mp.deflections_from_grid(np.array([[51.0, 51.0]]))[0, 1], 1e-5)

    class TestEinsteinRadiiMass:

        def test__x2_sis_different_einstein_radii_and_mass__einstein_radii_and_mass_are_sum(self):

            mp_0 = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
            mp_1 = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=0.5)

            gal_x2_mp = g.Galaxy(redshift=0.5, mass_0=mp_0, mass_1=mp_1)

            assert gal_x2_mp.einstein_radius_in_units(unit_length='arcsec') == 1.5
            assert gal_x2_mp.einstein_mass_in_units(unit_mass='angular') == np.pi*(1.0 + 0.5**2.0)

        def test__includes_shear__does_not_impact_values(self):

            mp_0 = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
            shear = mp.ExternalShear()

            gal_shear = g.Galaxy(redshift=0.5, mass_0=mp_0, shear=shear)

            assert gal_shear.einstein_radius_in_units(unit_length='arcsec') == 1.0
            assert gal_shear.einstein_mass_in_units(unit_mass='angular') == np.pi


class TestMassAndLightProfiles(object):

    def test_single_profile(self, lmp_0):

        gal_x1_lmp = g.Galaxy(redshift=0.5, profile=lmp_0)

        assert 1 == len(gal_x1_lmp.light_profiles)
        assert 1 == len(gal_x1_lmp.mass_profiles)

        assert gal_x1_lmp.mass_profiles[0] == lmp_0
        assert gal_x1_lmp.light_profiles[0] == lmp_0

    def test_multiple_profile(self, lmp_0, lp_0, mp_0):

        gal_multi_profiles = g.Galaxy(redshift=0.5, profile=lmp_0, light=lp_0, sie=mp_0)

        assert 2 == len(gal_multi_profiles.light_profiles)
        assert 2 == len(gal_multi_profiles.mass_profiles)


class TestSummarizeInUnits(object):

    def test__galaxy_with_two_light_and_mass_profiles(self, lp_0, lp_1, mp_0, mp_1):

        gal_summarize = g.Galaxy(redshift=0.5, light_profile_0=lp_0, light_profile_1=lp_1,
                       mass_profile_0=mp_0, mass_profile_1=mp_1)


        summary_text = gal_summarize.summarize_in_units(radii=[dim.Length(10.0), dim.Length(500.0)], whitespace=50,
                                              unit_length='arcsec', unit_luminosity='eps', unit_mass='angular')

        i = 0

        assert summary_text[i] == 'Galaxy\n' ; i += 1
        assert summary_text[i] == 'redshift                                          0.50' ; i += 1
        assert summary_text[i] == '\nGALAXY LIGHT\n\n' ; i += 1
        assert summary_text[i] == 'luminosity_within_10.00_arcsec                    1.8854e+02 eps' ; i += 1
        assert summary_text[i] == 'luminosity_within_500.00_arcsec                   1.9573e+02 eps' ; i += 1
        assert summary_text[i] ==  '\nLIGHT PROFILES:\n\n' ; i += 1
        assert summary_text[i] == 'Light Profile = SphericalSersic\n' ; i += 1
        assert summary_text[i] == 'luminosity_within_10.00_arcsec                    6.2848e+01 eps' ; i += 1
        assert summary_text[i] == 'luminosity_within_500.00_arcsec                   6.5243e+01 eps' ; i += 1
        assert summary_text[i] == '\n' ; i += 1
        assert summary_text[i] == 'Light Profile = SphericalSersic\n' ; i += 1
        assert summary_text[i] == 'luminosity_within_10.00_arcsec                    1.2570e+02 eps' ; i += 1
        assert summary_text[i] == 'luminosity_within_500.00_arcsec                   1.3049e+02 eps' ; i += 1
        assert summary_text[i] == '\n' ; i += 1
        assert summary_text[i] ==  '\nGALAXY MASS\n\n' ; i += 1
        assert summary_text[i] == 'einstein_radius                                   3.00 arcsec' ; i += 1
        assert summary_text[i] == 'einstein_mass                                     1.5708e+01 angular' ; i += 1
        assert summary_text[i] == 'mass_within_10.00_arcsec                          9.4248e+01 angular' ; i += 1
        assert summary_text[i] == 'mass_within_500.00_arcsec                         4.7124e+03 angular' ; i += 1
        assert summary_text[i] ==  '\nMASS PROFILES:\n\n' ; i += 1
        assert summary_text[i] == 'Mass Profile = SphericalIsothermal\n' ; i += 1
        assert summary_text[i] == 'einstein_radius                                   1.00 arcsec' ; i += 1
        assert summary_text[i] == 'einstein_mass                                     3.1416e+00 angular' ; i += 1
        assert summary_text[i] == 'mass_within_10.00_arcsec                          3.1416e+01 angular' ; i += 1
        assert summary_text[i] == 'mass_within_500.00_arcsec                         1.5708e+03 angular' ; i += 1
        assert summary_text[i] == '\n' ; i += 1
        assert summary_text[i] == 'Mass Profile = SphericalIsothermal\n' ; i += 1
        assert summary_text[i] == 'einstein_radius                                   2.00 arcsec' ; i += 1
        assert summary_text[i] == 'einstein_mass                                     1.2566e+01 angular' ; i += 1
        assert summary_text[i] == 'mass_within_10.00_arcsec                          6.2832e+01 angular' ; i += 1
        assert summary_text[i] == 'mass_within_500.00_arcsec                         3.1416e+03 angular' ; i += 1
        assert summary_text[i] == '\n' ; i += 1


class TestHyperGalaxy(object):

    class TestContributionMaps(object):

        def test__model_image_all_1s__factor_is_0__contributions_all_1s(self):

            hyper_image = np.ones((3,))

            hyp = g.HyperGalaxy(contribution_factor=0.0)
            contribution_map = hyp.contribution_map_from_hyper_images(
                hyper_model_image=hyper_image, hyper_galaxy_image=hyper_image)

            assert (contribution_map == np.ones((3,))).all()

        def test__different_values__factor_is_1__contributions_are_value_divided_by_factor_and_max(self):

            hyper_image = np.array([0.5, 1.0, 1.5])

            hyp = g.HyperGalaxy(contribution_factor=1.0)
            contribution_map = hyp.contribution_map_from_hyper_images(
                hyper_model_image=hyper_image, hyper_galaxy_image=hyper_image)

            assert (contribution_map == np.array([(0.5 / 1.5) / (1.5 / 2.5), (1.0 / 2.0) / (1.5 / 2.5), 1.0])).all()

    class TestHyperNoiseMap(object):

        def test__contribution_all_1s__noise_factor_2__noise_adds_double(self):

            noise_map = np.array([1.0, 2.0, 3.0])
            contribution_map = np.ones((3, 1))

            hyper_galaxy = g.HyperGalaxy(contribution_factor=0.0, noise_factor=2.0, noise_power=1.0)

            hyper_noise_map = hyper_galaxy.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map)

            assert (hyper_noise_map == np.array([2.0, 4.0, 6.0])).all()

        def test__same_as_above_but_contributions_vary(self):

            noise_map = np.array([1.0, 2.0, 3.0])
            contribution_map = np.array([[0.0, 0.5, 1.0]])

            hyper_galaxy = g.HyperGalaxy(contribution_factor=0.0, noise_factor=2.0, noise_power=1.0)

            hyper_noise_map = hyper_galaxy.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map)

            assert (hyper_noise_map == np.array([0.0, 2.0, 6.0])).all()

        def test__same_as_above_but_change_noise_scale_terms(self):

            noise_map = np.array([1.0, 2.0, 3.0])
            contribution_map = np.array([[0.0, 0.5, 1.0]])

            hyper_galaxy = g.HyperGalaxy(contribution_factor=0.0, noise_factor=2.0, noise_power=2.0)

            hyper_noise_map = hyper_galaxy.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map)

            assert (hyper_noise_map == np.array([0.0, 2.0, 18.0])).all()


class TestUsesBools(object):

    def test__uses_cluster_inversion__tests_depend_on_any_pixelizations(self):

        galaxy = g.Galaxy(
            redshift=0.5)
        assert galaxy.uses_inversion == False

        galaxy = g.Galaxy(
            redshift=0.5,
            pixelization=pix.Rectangular(),
            regularization=reg.Constant())

        assert galaxy.uses_inversion == True

        galaxy = g.Galaxy(
            redshift=0.5,
            pixelization=pix.VoronoiBrightnessImage(),
            regularization=reg.AdaptiveBrightness())

        assert galaxy.uses_inversion == True

    def test__uses_cluster_inversion__tests_depend_specific_pixelizations(self):

        galaxy = g.Galaxy(
            redshift=0.5)
        assert galaxy.uses_cluster_inversion == False

        galaxy = g.Galaxy(
            redshift=0.5,
            pixelization=pix.Rectangular(),
            regularization=reg.Constant())

        assert galaxy.uses_cluster_inversion == False

        galaxy = g.Galaxy(
            redshift=0.5,
            pixelization=pix.VoronoiBrightnessImage(),
            regularization=reg.AdaptiveBrightness())

        assert galaxy.uses_cluster_inversion == True

    def test__uses_hyper_images__tests_depend_on_hyper_galaxy_and_specific_pixelizations_and_regularizations(self):
        
        galaxy = g.Galaxy(
            redshift=0.5)

        assert galaxy.uses_hyper_images == False

        galaxy = g.Galaxy(
            redshift=0.5,
            hyper_galaxy=g.HyperGalaxy())

        assert galaxy.uses_hyper_images == True

        galaxy = g.Galaxy(
            redshift=0.5,
            pixelization=pix.Rectangular(),
            regularization=reg.Constant())

        assert galaxy.uses_hyper_images == False

        galaxy = g.Galaxy(
            redshift=0.5,
            pixelization=pix.Rectangular(),
            regularization=reg.AdaptiveBrightness())

        assert galaxy.uses_hyper_images == True


class TestBooleanProperties(object):

    def test_has_profile(self):

        assert g.Galaxy(redshift=0.5).has_profile is False
        assert g.Galaxy(redshift=0.5, light_profile=lp.LightProfile()).has_profile is True
        assert g.Galaxy(redshift=0.5, mass_profile=mp.MassProfile()).has_profile is True

    def test_has_light_profile(self):
        assert g.Galaxy(redshift=0.5).has_light_profile is False
        assert g.Galaxy(redshift=0.5, light_profile=lp.LightProfile()).has_light_profile is True
        assert g.Galaxy(redshift=0.5, mass_profile=mp.MassProfile()).has_light_profile is False

    def test_has_mass_profile(self):
        assert g.Galaxy(redshift=0.5).has_mass_profile is False
        assert g.Galaxy(redshift=0.5, light_profile=lp.LightProfile()).has_mass_profile is False
        assert g.Galaxy(redshift=0.5, mass_profile=mp.MassProfile()).has_mass_profile is True

    def test_has_redshift(self):

        assert g.Galaxy(redshift=0.1).has_redshift is True

    def test_has_pixelization(self):
        assert g.Galaxy(redshift=0.5).has_pixelization is False
        assert g.Galaxy(redshift=0.5, pixelization=object(), regularization=object()).has_pixelization is True

    def test_has_regularization(self):
        assert g.Galaxy(redshift=0.5).has_regularization is False
        assert g.Galaxy(redshift=0.5, pixelization=object(), regularization=object()).has_regularization is True

    def test_has_hyper_galaxy(self):
        assert g.Galaxy(redshift=0.5).has_pixelization is False
        assert g.Galaxy(redshift=0.5, hyper_galaxy=object()).has_hyper_galaxy is True

    def test__only_pixelization_raises_error(self):
        with pytest.raises(exc.GalaxyException):
            g.Galaxy(redshift=0.5, pixelization=object())

    def test__only_regularization_raises_error(self):
        with pytest.raises(exc.GalaxyException):
            g.Galaxy(redshift=0.5, regularization=object())


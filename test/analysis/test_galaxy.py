from autolens.analysis import galaxy
from autolens.profiles import mass_profiles, light_profiles

import pytest
import numpy as np


class TestLightProfiles(object):

    class TestIntensity:

        def test__one_profile_galaxy__intensity_is_same_individual_profile(self):

            sersic = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                     effective_radius=0.6, sersic_index=4.0)

            sersic_intensity = sersic.intensity_from_grid(grid=np.array([[1.05, -0.55]]))

            galaxy_sersic = galaxy.Galaxy(redshift=0.5,
                                          light_profile_1=light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0,
                                                                                          intensity=1.0,
                                                                                          effective_radius=0.6,
                                                                                          sersic_index=4.0))

            galaxy_sersic_intensity = galaxy_sersic.intensity_from_grid(np.array([[1.05, -0.55]]))

            assert sersic_intensity == galaxy_sersic_intensity

        def test__two_profile_galaxy__intensity_is_sum_of_individual_profiles(self):
            sersic_1 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                       effective_radius=2.0,
                                                       sersic_index=1.0)

            sersic_2 = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            intensity = sersic_1.intensity_from_grid(np.array([[1.05, -0.55]]))
            intensity += sersic_2.intensity_from_grid(np.array([[1.05, -0.55]]))

            galaxy_sersic = galaxy.Galaxy(redshift=0.5,
                                          light_profile_1=light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0,
                                                                                          intensity=3.0,
                                                                                          effective_radius=2.0,
                                                                                          sersic_index=1.0),
                                          light_profile_2=light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0,
                                                                                          intensity=7.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0))

            galaxy_intensity = galaxy_sersic.intensity_from_grid(np.array([[1.05, -0.55]]))

            assert intensity == galaxy_intensity

        def test__three_profile_galaxy__intensity_is_sum_of_individual_profiles(self):
            sersic_1 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                       effective_radius=2.0,
                                                       sersic_index=1.0)

            sersic_2 = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            sersic_3 = light_profiles.EllipticalSersic(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            intensity = sersic_1.intensity_from_grid(np.array([[1.05, -0.55]]))
            intensity += sersic_2.intensity_from_grid(np.array([[1.05, -0.55]]))
            intensity += sersic_3.intensity_from_grid(np.array([[1.05, -0.55]]))

            galaxy_sersic = galaxy.Galaxy(redshift=0.5,
                                          light_profile_1=light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0,
                                                                                          intensity=3.0,
                                                                                          effective_radius=2.0,
                                                                                          sersic_index=1.0),
                                          light_profile_2=light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0,
                                                                                          intensity=7.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0),
                                          light_profile_3=light_profiles.EllipticalSersic(axis_ratio=0.8, phi=50.0,
                                                                                          intensity=2.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0))

            galaxy_intensity = galaxy_sersic.intensity_from_grid(np.array([[1.05, -0.55]]))

            assert intensity == galaxy_intensity

        def test__three_profile_galaxy__individual_intensities_can_be_extracted(self):

            sersic_1 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                       effective_radius=2.0,
                                                       sersic_index=1.0)

            sersic_2 = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            sersic_3 = light_profiles.EllipticalSersic(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            intensity_1 = sersic_1.intensity_from_grid(np.array([[1.05, -0.55]]))
            intensity_2 = sersic_2.intensity_from_grid(np.array([[1.05, -0.55]]))
            intensity_3 = sersic_3.intensity_from_grid(np.array([[1.05, -0.55]]))

            galaxy_sersic = galaxy.Galaxy(redshift=0.5,
                                          light_profile_1=light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0,
                                                                                          intensity=3.0,
                                                                                          effective_radius=2.0,
                                                                                          sersic_index=1.0),
                                          light_profile_2=light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0,
                                                                                          intensity=7.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0),
                                          light_profile_3=light_profiles.EllipticalSersic(axis_ratio=0.8, phi=50.0,
                                                                                          intensity=2.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0))

            galaxy_intensity = galaxy_sersic.intensity_from_grid_individual(np.array([[1.05, -0.55]]))

            assert intensity_1 == galaxy_intensity[0]
            assert intensity_2 == galaxy_intensity[1]
            assert intensity_3 == galaxy_intensity[2]

    class TestLuminosityWithinCircle:

        def test__one_profile_galaxy__integral_is_same_as_individual_profile(self):
            sersic = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0, effective_radius=2.0,
                                                     sersic_index=1.0)

            integral_radius = 5.5

            intensity_integral = sersic.luminosity_within_circle(radius=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5,
                                          light_profile_1=light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0,
                                                                                          intensity=3.0,
                                                                                          effective_radius=2.0,
                                                                                          sersic_index=1.0))

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_circle(radius=integral_radius)

            assert intensity_integral == galaxy_intensity_integral

        def test__two_profile_galaxy__integral_is_sum_of_individual_profiles(self):
            sersic_1 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                       effective_radius=2.0,
                                                       sersic_index=1.0)

            sersic_2 = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral = sersic_1.luminosity_within_circle(radius=integral_radius)
            intensity_integral += sersic_2.luminosity_within_circle(radius=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5,
                                          light_profile_1=light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0,
                                                                                          intensity=3.0,
                                                                                          effective_radius=2.0,
                                                                                          sersic_index=1.0),
                                          light_profile_2=light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0,
                                                                                          intensity=7.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0))

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_circle(radius=integral_radius)

            assert intensity_integral == galaxy_intensity_integral

        def test__three_profile_galaxy__integral_is_sum_of_individual_profiles(self):
            sersic_1 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                       effective_radius=2.0,
                                                       sersic_index=1.0)

            sersic_2 = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            sersic_3 = light_profiles.EllipticalSersic(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral = sersic_1.luminosity_within_circle(radius=integral_radius)
            intensity_integral += sersic_2.luminosity_within_circle(radius=integral_radius)
            intensity_integral += sersic_3.luminosity_within_circle(radius=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5,
                                          light_profile_1=light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0,
                                                                                          intensity=3.0,
                                                                                          effective_radius=2.0,
                                                                                          sersic_index=1.0),
                                          light_profile_2=light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0,
                                                                                          intensity=7.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0),
                                          light_profile_3=light_profiles.EllipticalSersic(axis_ratio=0.8, phi=50.0,
                                                                                          intensity=2.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0))

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_circle(radius=integral_radius)

            assert intensity_integral == galaxy_intensity_integral

        def test__three_profile_galaxy__individual_integrals_can_be_extracted(self):
            sersic_1 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                       effective_radius=2.0,
                                                       sersic_index=1.0)

            sersic_2 = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            sersic_3 = light_profiles.EllipticalSersic(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral_1 = sersic_1.luminosity_within_circle(radius=integral_radius)
            intensity_integral_2 = sersic_2.luminosity_within_circle(radius=integral_radius)
            intensity_integral_3 = sersic_3.luminosity_within_circle(radius=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5,
                                          light_profile_1=light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0,
                                                                                          intensity=3.0,
                                                                                          effective_radius=2.0,
                                                                                          sersic_index=1.0),
                                          light_profile_2=light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0,
                                                                                          intensity=7.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0),
                                          light_profile_3=light_profiles.EllipticalSersic(axis_ratio=0.8, phi=50.0,
                                                                                          intensity=2.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0))

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_circle_individual(
                radius=integral_radius)

            assert intensity_integral_1 == galaxy_intensity_integral[0]
            assert intensity_integral_2 == galaxy_intensity_integral[1]
            assert intensity_integral_3 == galaxy_intensity_integral[2]

    class TestLuminosityWithinEllipse:

        def test__one_profile_galaxy__integral_is_same_as_individual_profile(self):
            sersic = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0,
                                                     sersic_index=1.0)

            integral_radius = 0.5

            intensity_integral = sersic.luminosity_within_ellipse(major_axis=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5,
                                          light_profile_1=light_profiles.EllipticalSersic(axis_ratio=0.5,
                                                                                          phi=0.0,
                                                                                          intensity=3.0,
                                                                                          effective_radius=2.0,
                                                                                          sersic_index=1.0))

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_ellipse(major_axis=integral_radius)

            assert intensity_integral == galaxy_intensity_integral

        def test__two_profile_galaxy__integral_is_sum_of_individual_profiles(self):
            sersic_1 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                       effective_radius=2.0,
                                                       sersic_index=1.0)

            sersic_2 = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral = sersic_1.luminosity_within_ellipse(major_axis=integral_radius)
            intensity_integral += sersic_2.luminosity_within_ellipse(major_axis=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5,
                                          light_profile_1=light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0,
                                                                                          intensity=3.0,
                                                                                          effective_radius=2.0,
                                                                                          sersic_index=1.0),
                                          light_profile_2=light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0,
                                                                                          intensity=7.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0))

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_ellipse(major_axis=integral_radius)

            assert intensity_integral == galaxy_intensity_integral

        def test__three_profile_galaxy__integral_is_sum_of_individual_profiles(self):
            sersic_1 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                       effective_radius=2.0,
                                                       sersic_index=1.0)

            sersic_2 = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            sersic_3 = light_profiles.EllipticalSersic(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral = sersic_1.luminosity_within_ellipse(major_axis=integral_radius)
            intensity_integral += sersic_2.luminosity_within_ellipse(major_axis=integral_radius)
            intensity_integral += sersic_3.luminosity_within_ellipse(major_axis=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5,
                                          light_profile_1=light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0,
                                                                                          intensity=3.0,
                                                                                          effective_radius=2.0,
                                                                                          sersic_index=1.0),
                                          light_profile_2=light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0,
                                                                                          intensity=7.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0),
                                          light_profile_3=light_profiles.EllipticalSersic(axis_ratio=0.8, phi=50.0,
                                                                                          intensity=2.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0))

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_ellipse(major_axis=integral_radius)

            assert intensity_integral == galaxy_intensity_integral

        def test__three_profile_galaxy__individual_integrals_can_be_extracted(self):
            sersic_1 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                       effective_radius=2.0,
                                                       sersic_index=1.0)

            sersic_2 = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            sersic_3 = light_profiles.EllipticalSersic(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                       effective_radius=3.0,
                                                       sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral_1 = sersic_1.luminosity_within_ellipse(major_axis=integral_radius)
            intensity_integral_2 = sersic_2.luminosity_within_ellipse(major_axis=integral_radius)
            intensity_integral_3 = sersic_3.luminosity_within_ellipse(major_axis=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5,
                                          light_profile_1=light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0,
                                                                                          intensity=3.0,
                                                                                          effective_radius=2.0,
                                                                                          sersic_index=1.0),
                                          light_profile_2=light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0,
                                                                                          intensity=7.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0),
                                          light_profile_3=light_profiles.EllipticalSersic(axis_ratio=0.8, phi=50.0,
                                                                                          intensity=2.0,
                                                                                          effective_radius=3.0,
                                                                                          sersic_index=2.0))

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_ellipse_individual(
                major_axis=integral_radius)

            assert intensity_integral_1 == galaxy_intensity_integral[0]
            assert intensity_integral_2 == galaxy_intensity_integral[1]
            assert intensity_integral_3 == galaxy_intensity_integral[2]

    class TestSymmetricProfiles(object):

        def test_1d_symmetry(self):

            sersic_1 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0)

            sersic_2 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0, centre=(100, 0))

            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profile_1=sersic_1, light_profile_2=sersic_2)

            assert galaxy_sersic.intensity_from_grid(
                np.array([[0.0, 0.0]])) == galaxy_sersic.intensity_from_grid(np.array([[100.0, 0.0]]))
            assert galaxy_sersic.intensity_from_grid(
                np.array([[49.0, 0.0]])) == galaxy_sersic.intensity_from_grid(np.array([[51.0, 0.0]]))

        def test_2d_symmetry(self):
            sersic_1 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0)

            sersic_2 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0, centre=(100, 0))

            sersic_3 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0, centre=(0, 100))

            sersic_4 = light_profiles.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0, centre=(100, 100))

            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profile_1=sersic_1, light_profile_2=sersic_2,
                                          light_profile_3=sersic_3, light_profile_4=sersic_4)

            assert galaxy_sersic.intensity_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                galaxy_sersic.intensity_from_grid(np.array([[51.0, 0.0]])), 1e-5)

            assert galaxy_sersic.intensity_from_grid(np.array([[0.0, 49.0]])) == pytest.approx(
                galaxy_sersic.intensity_from_grid(np.array([[0.0, 51.0]])), 1e-5)

            assert galaxy_sersic.intensity_from_grid(np.array([[100.0, 49.0]])) == pytest.approx(
                galaxy_sersic.intensity_from_grid(np.array([[100.0, 51.0]])), 1e-5)

            assert galaxy_sersic.intensity_from_grid(np.array([[49.0, 49.0]])) == pytest.approx(
                galaxy_sersic.intensity_from_grid(np.array([[51.0, 51.0]])), 1e-5)


@pytest.fixture(name="sie_1")
def make_sie_1():
    return mass_profiles.EllipticalIsothermal(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)


@pytest.fixture(name="sie_2")
def make_sie_2():
    return mass_profiles.EllipticalIsothermal(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)


@pytest.fixture(name="sie_3")
def make_sie_3():
    return mass_profiles.EllipticalIsothermal(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)


@pytest.fixture(name="galaxy_sie_1")
def make_galaxy_sie_1(sie_1):
    return galaxy.Galaxy(redshift=0.5, mass_profile_1=sie_1)


@pytest.fixture(name="galaxy_sie_2")
def make_galaxy_sie_2(sie_1, sie_2):
    return galaxy.Galaxy(redshift=0.5, mass_profile_1=sie_1, mass_profile_2=sie_2)


@pytest.fixture(name="galaxy_sie_3")
def make_galaxy_sie_3(sie_1, sie_2, sie_3):
    return galaxy.Galaxy(redshift=0.5, mass_profile_1=sie_1, mass_profile_2=sie_2, mass_profile_3=sie_3)


class TestMassProfiles(object):

    class TestSurfaceDensity:

        def test__one_profile_galaxy__surface_density_is_same_individual_profile(self, sie_1, galaxy_sie_1):

            sie_surface_density = sie_1.surface_density_from_grid(np.array([[1.05, -0.55]]))

            galaxy_sie_surface_density = galaxy_sie_1.surface_density_from_grid(np.array([[1.05, -0.55]]))

            assert sie_surface_density == galaxy_sie_surface_density

        def test__two_profile_galaxy__surface_density_is_sum_of_individual_profiles(self, sie_1, sie_2, galaxy_sie_2):
            surface_density = sie_1.surface_density_from_grid(np.array([[1.05, -0.55]]))
            surface_density += sie_2.surface_density_from_grid(np.array([[1.05, -0.55]]))

            galaxy_surface_density = galaxy_sie_2.surface_density_from_grid(np.array([[1.05, -0.55]]))

            assert surface_density == galaxy_surface_density

        def test__three_profile_galaxy__surface_density_is_sum_of_individual_profiles(self, sie_1, sie_2, sie_3,
                                                                                      galaxy_sie_3):
            surface_density = sie_1.surface_density_from_grid(np.array([[1.05, -0.55]]))
            surface_density += sie_2.surface_density_from_grid(np.array([[1.05, -0.55]]))
            surface_density += sie_3.surface_density_from_grid(np.array([[1.05, -0.55]]))

            galaxy_surface_density = galaxy_sie_3.surface_density_from_grid(np.array([[1.05, -0.55]]))

            assert surface_density == galaxy_surface_density

        def test__three_profile_galaxy__individual_surface_densities_can_be_extracted(self, sie_1, sie_2, sie_3,
                                                                                      galaxy_sie_3):

            surface_density_1 = sie_1.surface_density_from_grid(np.array([[1.05, -0.55]]))
            surface_density_2 = sie_2.surface_density_from_grid(np.array([[1.05, -0.55]]))
            surface_density_3 = sie_3.surface_density_from_grid(np.array([[1.05, -0.55]]))

            galaxy_surface_density = galaxy_sie_3.surface_density_from_grid_individual(np.array([[1.05, -0.55]]))

            assert surface_density_1 == galaxy_surface_density[0]
            assert surface_density_2 == galaxy_surface_density[1]
            assert surface_density_3 == galaxy_surface_density[2]

    class TestPotential:

        def test__one_profile_galaxy__potential_is_same_individual_profile(self, sie_1, galaxy_sie_1):

            sie_potential = sie_1.potential_from_grid(np.array([[1.05, -0.55]]))

            galaxy_sie_potential = galaxy_sie_1.potential_from_grid(np.array([[1.05, -0.55]]))

            assert sie_potential == galaxy_sie_potential

        def test__two_profile_galaxy__potential_is_sum_of_individual_profiles(self, sie_1, sie_2, galaxy_sie_2):
            potential = sie_1.potential_from_grid(np.array([[1.05, -0.55]]))
            potential += sie_2.potential_from_grid(np.array([[1.05, -0.55]]))

            galaxy_potential = galaxy_sie_2.potential_from_grid(np.array([[1.05, -0.55]]))

            assert potential == galaxy_potential

        def test__three_profile_galaxy__potential_is_sum_of_individual_profiles(self, sie_1, sie_2, sie_3,
                                                                                galaxy_sie_3):
            potential = sie_1.potential_from_grid(np.array([[1.05, -0.55]]))
            potential += sie_2.potential_from_grid(np.array([[1.05, -0.55]]))
            potential += sie_3.potential_from_grid(np.array([[1.05, -0.55]]))

            galaxy_potential = galaxy_sie_3.potential_from_grid(np.array([[1.05, -0.55]]))

            assert potential == galaxy_potential

        def test__three_profile_galaxy__individual_potentials_can_be_extracted(self, sie_1, sie_2, sie_3,
                                                                               galaxy_sie_3):

            potential_1 = sie_1.potential_from_grid(np.array([[1.05, -0.55]]))
            potential_2 = sie_2.potential_from_grid(np.array([[1.05, -0.55]]))
            potential_3 = sie_3.potential_from_grid(np.array([[1.05, -0.55]]))

            galaxy_potential = galaxy_sie_3.potential_from_grid_individual(np.array([[1.05, -0.55]]))

            assert potential_1 == galaxy_potential[0]
            assert potential_2 == galaxy_potential[1]
            assert potential_3 == galaxy_potential[2]

    class TestDeflectionAngles:

        def test__one_profile_galaxy__deflection_angles_is_same_individual_profile(self, sie_1, galaxy_sie_1):

            sie_deflection_angles = sie_1.deflections_from_grid(np.array([[1.05, -0.55]]))

            galaxy_sie_deflection_angles = galaxy_sie_1.deflections_from_grid(np.array([[1.05, -0.55]]))

            assert sie_deflection_angles[0,0] == galaxy_sie_deflection_angles[0,0]
            assert sie_deflection_angles[0,1] == galaxy_sie_deflection_angles[0,1]

        def test__two_profile_galaxy__deflection_angles_is_sum_of_individual_profiles(self):

            sie_1 = mass_profiles.EllipticalIsothermal(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermal(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)

            deflection_angles_0 = sie_1.deflections_from_grid(np.array([[1.05, -0.55]]))
            deflection_angles_1 = sie_2.deflections_from_grid(np.array([[1.05, -0.55]]))

            deflection_angles = deflection_angles_0 + deflection_angles_1

            galaxy_sie = galaxy.Galaxy(redshift=0.5,
                                       mass_profile_1=mass_profiles.EllipticalIsothermal(axis_ratio=0.8, phi=10.0,
                                                                                         einstein_radius=1.0),
                                       mass_profile_2=mass_profiles.EllipticalIsothermal(axis_ratio=0.6, phi=30.0,
                                                                                         einstein_radius=1.2))

            galaxy_deflection_angles = galaxy_sie.deflections_from_grid(np.array([[1.05, -0.55]]))

            assert deflection_angles[0,0] == galaxy_deflection_angles[0,0]
            assert deflection_angles[0,1] == galaxy_deflection_angles[0,1]

        def test__three_profile_galaxy__deflection_angles_is_sum_of_individual_profiles(self, sie_1, sie_2, sie_3,
                                                                                        galaxy_sie_3):

            deflection_angles_0 = sie_1.deflections_from_grid(np.array([[1.05, -0.55]]))
            deflection_angles_1 = sie_2.deflections_from_grid(np.array([[1.05, -0.55]]))
            deflection_angles_2 = sie_3.deflections_from_grid(np.array([[1.05, -0.55]]))

            deflection_angles = deflection_angles_0 + deflection_angles_1 + deflection_angles_2

            galaxy_deflection_angles = galaxy_sie_3.deflections_from_grid(np.array([[1.05, -0.55]]))

            assert deflection_angles[0,0] == galaxy_deflection_angles[0,0]
            assert deflection_angles[0,1] == galaxy_deflection_angles[0,1]

        def test__three_profile_galaxy__individual_deflection_angles_can_be_extracted(self, sie_1, sie_2, sie_3,
                                                                                      galaxy_sie_3):

            deflection_angles_0 = sie_1.deflections_from_grid(np.array([[1.05, -0.55]]))
            deflection_angles_1 = sie_2.deflections_from_grid(np.array([[1.05, -0.55]]))
            deflection_angles_2 = sie_3.deflections_from_grid(np.array([[1.05, -0.55]]))

            galaxy_deflection_angles = galaxy_sie_3.deflections_from_grid_individual(np.array([[1.05, -0.55]]))

            assert (deflection_angles_0 == galaxy_deflection_angles[0]).all()
            assert (deflection_angles_1 == galaxy_deflection_angles[1]).all()
            assert (deflection_angles_2 == galaxy_deflection_angles[2]).all()

    class TestDimensionlessMassWithinCircle:

        def test__one_profile_galaxy__integral_is_same_as_individual_profile(self, sie_1, galaxy_sie_1):
            integral_radius = 5.5

            mass_integral = sie_1.dimensionless_mass_within_circle(radius=integral_radius)

            galaxy_mass_integral = galaxy_sie_1.dimensionless_mass_within_circles(radius=integral_radius)

            assert mass_integral == galaxy_mass_integral

        def test__two_profile_galaxy__integral_is_sum_of_individual_profiles(self):
            sie_1 = mass_profiles.EllipticalIsothermal(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermal(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)

            integral_radius = 5.5

            mass_integral = sie_1.dimensionless_mass_within_circle(radius=integral_radius)
            mass_integral += sie_2.dimensionless_mass_within_circle(radius=integral_radius)

            galaxy_sie = galaxy.Galaxy(redshift=0.5,
                                       mass_profile_1=mass_profiles.EllipticalIsothermal(axis_ratio=0.8, phi=10.0,
                                                                                         einstein_radius=1.0),
                                       mass_profile_2=mass_profiles.EllipticalIsothermal(axis_ratio=0.6, phi=30.0,
                                                                                         einstein_radius=1.2))

            galaxy_mass_integral = galaxy_sie.dimensionless_mass_within_circles(radius=integral_radius)

            assert mass_integral == galaxy_mass_integral

        def test__three_profile_galaxy__integral_is_sum_of_individual_profiles(self, sie_1, sie_2, sie_3,
                                                                               galaxy_sie_3):
            integral_radius = 5.5

            mass_integral = sie_1.dimensionless_mass_within_circle(radius=integral_radius)
            mass_integral += sie_2.dimensionless_mass_within_circle(radius=integral_radius)
            mass_integral += sie_3.dimensionless_mass_within_circle(radius=integral_radius)

            galaxy_mass_integral = galaxy_sie_3.dimensionless_mass_within_circles(radius=integral_radius)

            assert mass_integral == galaxy_mass_integral

        def test__three_profile_galaxy__individual_integrals_can_be_extracted(self, sie_1, sie_2, sie_3,
                                                                              galaxy_sie_3):
            integral_radius = 5.5

            mass_integral_1 = sie_1.dimensionless_mass_within_circle(radius=integral_radius)
            mass_integral_2 = sie_2.dimensionless_mass_within_circle(radius=integral_radius)
            mass_integral_3 = sie_3.dimensionless_mass_within_circle(radius=integral_radius)

            galaxy_mass_integral = galaxy_sie_3.dimensionless_mass_within_circles_individual(
                radius=integral_radius)

            assert mass_integral_1 == galaxy_mass_integral[0]
            assert mass_integral_2 == galaxy_mass_integral[1]
            assert mass_integral_3 == galaxy_mass_integral[2]

    class TestDimensionlessMassWithinEllipse:

        def test__one_profile_galaxy__integral_is_same_as_individual_profile(self, sie_1, galaxy_sie_1):
            integral_radius = 0.5

            dimensionless_mass_integral = sie_1.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            galaxy_dimensionless_mass_integral = galaxy_sie_1.dimensionless_mass_within_ellipses(
                major_axis=integral_radius)

            assert dimensionless_mass_integral == galaxy_dimensionless_mass_integral

        def test__two_profile_galaxy__integral_is_sum_of_individual_profiles(self, sie_1, sie_2, galaxy_sie_2):
            integral_radius = 5.5

            dimensionless_mass_integral = sie_1.dimensionless_mass_within_ellipse(major_axis=integral_radius)
            dimensionless_mass_integral += sie_2.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            galaxy_dimensionless_mass_integral = galaxy_sie_2.dimensionless_mass_within_ellipses(
                major_axis=integral_radius)

            assert dimensionless_mass_integral == galaxy_dimensionless_mass_integral

        def test__three_profile_galaxy__integral_is_sum_of_individual_profiles(self, sie_1, sie_2, sie_3,
                                                                               galaxy_sie_3):
            integral_radius = 5.5

            dimensionless_mass_integral = sie_1.dimensionless_mass_within_ellipse(major_axis=integral_radius)
            dimensionless_mass_integral += sie_2.dimensionless_mass_within_ellipse(major_axis=integral_radius)
            dimensionless_mass_integral += sie_3.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            galaxy_dimensionless_mass_integral = galaxy_sie_3.dimensionless_mass_within_ellipses(
                major_axis=integral_radius)

            assert dimensionless_mass_integral == galaxy_dimensionless_mass_integral

        def test__three_profile_galaxy__individual_integrals_can_be_extracted(self, sie_1, sie_2, sie_3,
                                                                              galaxy_sie_3):
            integral_radius = 5.5

            dimensionless_mass_integral_1 = sie_1.dimensionless_mass_within_ellipse(major_axis=integral_radius)
            dimensionless_mass_integral_2 = sie_2.dimensionless_mass_within_ellipse(major_axis=integral_radius)
            dimensionless_mass_integral_3 = sie_3.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            galaxy_dimensionless_mass_integral = galaxy_sie_3.dimensionless_mass_within_ellipses_individual(
                major_axis=integral_radius)

            assert dimensionless_mass_integral_1 == galaxy_dimensionless_mass_integral[0]
            assert dimensionless_mass_integral_2 == galaxy_dimensionless_mass_integral[1]
            assert dimensionless_mass_integral_3 == galaxy_dimensionless_mass_integral[2]

    class TestSymmetricProfiles:

        def test_1d_symmetry(self):

            isothermal_1 = mass_profiles.EllipticalIsothermal(axis_ratio=0.5, phi=45.0,
                                                              einstein_radius=1.0)

            isothermal_2 = mass_profiles.EllipticalIsothermal(centre=(100, 0), axis_ratio=0.5, phi=45.0,
                                                              einstein_radius=1.0)

            galaxy_isothermal = galaxy.Galaxy(redshift=0.5, mass_profile_1=isothermal_1, mass_profile_2=isothermal_2)

            assert galaxy_isothermal.surface_density_from_grid(
                np.array([[1.0, 0.0]])) == galaxy_isothermal.surface_density_from_grid(np.array([[99.0, 0.0]]))

            assert galaxy_isothermal.surface_density_from_grid(
                np.array([[49.0, 0.0]])) == galaxy_isothermal.surface_density_from_grid(np.array([[51.0, 0.0]]))

            assert galaxy_isothermal.potential_from_grid(np.array([[1.0, 0.0]])) == pytest.approx(
                galaxy_isothermal.potential_from_grid(np.array([[99.0, 0.0]])), 1e-6)

            assert galaxy_isothermal.potential_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                galaxy_isothermal.potential_from_grid(np.array([[51.0, 0.0]])), 1e-6)

            assert galaxy_isothermal.deflections_from_grid(np.array([[1.0, 0.0]])) == pytest.approx(
                galaxy_isothermal.deflections_from_grid(np.array([[99.0, 0.0]])), 1e-6)

            assert galaxy_isothermal.deflections_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                galaxy_isothermal.deflections_from_grid(np.array([[51.0, 0.0]])), 1e-6)

        def test_2d_symmetry(self):

            isothermal_1 = mass_profiles.SphericalIsothermal(einstein_radius=1.0)

            isothermal_2 = mass_profiles.SphericalIsothermal(centre=(100, 0), einstein_radius=1.0)

            isothermal_3 = mass_profiles.SphericalIsothermal(centre=(0, 100), einstein_radius=1.0)

            isothermal_4 = mass_profiles.SphericalIsothermal(centre=(100, 100), einstein_radius=1.0)

            galaxy_isothermal = galaxy.Galaxy(redshift=0.5,
                                              mass_profile_1=isothermal_1, mass_profile_2=isothermal_2,
                                              mass_profile_3=isothermal_3, mass_profile_4=isothermal_4)

            assert galaxy_isothermal.surface_density_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                galaxy_isothermal.surface_density_from_grid(np.array([[51.0, 0.0]])), 1e-5)

            assert galaxy_isothermal.surface_density_from_grid(np.array([[0.0, 49.0]])) == pytest.approx(
                galaxy_isothermal.surface_density_from_grid(np.array([[0.0, 51.0]])), 1e-5)

            assert galaxy_isothermal.surface_density_from_grid(np.array([[100.0, 49.0]])) == pytest.approx(
                galaxy_isothermal.surface_density_from_grid(np.array([[100.0, 51.0]])), 1e-5)

            assert galaxy_isothermal.surface_density_from_grid(np.array([[49.0, 49.0]])) == pytest.approx(
                galaxy_isothermal.surface_density_from_grid(np.array([[51.0, 51.0]])), 1e-5)

            assert galaxy_isothermal.potential_from_grid(np.array([[49.0, 0.0]])) == pytest.approx(
                galaxy_isothermal.potential_from_grid(np.array([[51.0, 0.0]])), 1e-5)

            assert galaxy_isothermal.potential_from_grid(np.array([[0.0, 49.0]])) == pytest.approx(
                galaxy_isothermal.potential_from_grid(np.array([[0.0, 51.0]])), 1e-5)

            assert galaxy_isothermal.potential_from_grid(np.array([[100.0, 49.0]])) == pytest.approx(
                galaxy_isothermal.potential_from_grid(np.array([[100.0, 51.0]])), 1e-5)

            assert galaxy_isothermal.potential_from_grid(np.array([[49.0, 49.0]])) == pytest.approx(
                galaxy_isothermal.potential_from_grid(np.array([[51.0, 51.0]])), 1e-5)

            assert -1.0 * galaxy_isothermal.deflections_from_grid(np.array([[49.0, 0.0]]))[0, 0] == pytest.approx(
                galaxy_isothermal.deflections_from_grid(np.array([[51.0, 0.0]]))[0, 0], 1e-5)

            assert 1.0 * galaxy_isothermal.deflections_from_grid(np.array([[0.0, 49.0]]))[0, 0] == pytest.approx(
                galaxy_isothermal.deflections_from_grid(np.array([[0.0, 51.0]]))[0, 0], 1e-5)

            assert 1.0 * galaxy_isothermal.deflections_from_grid(np.array([[100.0, 49.0]]))[0, 0] == pytest.approx(
                galaxy_isothermal.deflections_from_grid(np.array([[100.0, 51.0]]))[0, 0], 1e-5)

            assert -1.0 * galaxy_isothermal.deflections_from_grid(np.array([[49.0, 49.0]]))[0, 0] == pytest.approx(
                galaxy_isothermal.deflections_from_grid(np.array([[51.0, 51.0]]))[0, 0], 1e-5)

            assert 1.0 * galaxy_isothermal.deflections_from_grid(np.array([[49.0, 0.0]]))[0, 1] == pytest.approx(
                galaxy_isothermal.deflections_from_grid(np.array([[51.0, 0.0]]))[0, 1], 1e-5)

            assert -1.0 * galaxy_isothermal.deflections_from_grid(np.array([[0.0, 49.0]]))[0, 1] == pytest.approx(
                galaxy_isothermal.deflections_from_grid(np.array([[0.0, 51.0]]))[0, 1], 1e-5)

            assert -1.0 * galaxy_isothermal.deflections_from_grid(np.array([[100.0, 49.0]]))[0, 1] == pytest.approx(
                galaxy_isothermal.deflections_from_grid(np.array([[100.0, 51.0]]))[0, 1], 1e-5)

            assert -1.0 * galaxy_isothermal.deflections_from_grid(np.array([[49.0, 49.0]]))[0, 1] == pytest.approx(
                galaxy_isothermal.deflections_from_grid(np.array([[51.0, 51.0]]))[0, 1], 1e-5)


class TestHyperGalaxy(object):

    class TestContributionMaps(object):

        def test__model_image_all_1s__factor_is_0__contributions_all_1s(self):
            galaxy_image = np.ones((3,))

            hyp = galaxy.HyperGalaxy(contribution_factor=0.0)
            contributions = hyp.contributions_from_model_images(model_image=galaxy_image, galaxy_image=galaxy_image,
                                                                minimum_value=0.0)

            assert (contributions == np.ones((3,))).all()

        def test__different_values__factor_is_1__contributions_are_value_divided_by_factor_and_max(self):
            galaxy_image = np.array([0.5, 1.0, 1.5])

            hyp = galaxy.HyperGalaxy(contribution_factor=1.0)
            contributions = hyp.contributions_from_model_images(model_image=galaxy_image, galaxy_image=galaxy_image,
                                                                minimum_value=0.0)

            assert (contributions == np.array([(0.5 / 1.5) / (1.5 / 2.5), (1.0 / 2.0) / (1.5 / 2.5), 1.0])).all()

        def test__different_values__threshold_is_1_minimum_threshold_included__wipes_1st_value_to_0(self):
            galaxy_image = np.array([0.5, 1.0, 1.5])

            hyp = galaxy.HyperGalaxy(contribution_factor=1.0)
            contributions = hyp.contributions_from_model_images(model_image=galaxy_image, galaxy_image=galaxy_image,
                                                                minimum_value=0.6)

            assert (contributions == np.array([0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0])).all()

    class TestScaledNoise(object):

        def test__contribution_all_1s__noise_factor_2__noise_adds_double(self):
            noise = np.array([1.0, 2.0, 3.0])
            galaxy_contributions = np.ones((3, 1))

            hyp = galaxy.HyperGalaxy(contribution_factor=0.0, noise_factor=2.0, noise_power=1.0)

            scaled_noise = hyp.scaled_noise_for_contributions(noise=noise, contributions=galaxy_contributions)

            assert (scaled_noise == np.array([2.0, 4.0, 6.0])).all()

        def test__same_as_above_but_contributions_vary(self):
            noise = np.array([1.0, 2.0, 3.0])
            galaxy_contributions = np.array([[0.0, 0.5, 1.0]])

            hyp = galaxy.HyperGalaxy(contribution_factor=0.0, noise_factor=2.0, noise_power=1.0)

            scaled_noise = hyp.scaled_noise_for_contributions(noise=noise, contributions=galaxy_contributions)

            assert (scaled_noise == np.array([0.0, 2.0, 6.0])).all()

        def test__same_as_above_but_change_noise_scale_terms(self):
            noise = np.array([1.0, 2.0, 3.0])
            galaxy_contributions = np.array([[0.0, 0.5, 1.0]])

            hyp = galaxy.HyperGalaxy(contribution_factor=0.0, noise_factor=2.0, noise_power=2.0)

            scaled_noise = hyp.scaled_noise_for_contributions(noise=noise, contributions=galaxy_contributions)

            assert (scaled_noise == np.array([0.0, 2.0, 18.0])).all()


class TestBooleanProperties(object):
    def test_has_pixelization(self):
        assert galaxy.Galaxy().has_pixelization is False
        assert galaxy.Galaxy(pixelization=object()).has_pixelization is True

    def test_has_hyper_galaxy(self):
        assert galaxy.Galaxy().has_pixelization is False
        assert galaxy.Galaxy(hyper_galaxy=object()).has_hyper_galaxy is True

    def test_has_profile(self):
        assert galaxy.Galaxy().has_profile is False
        assert galaxy.Galaxy(light_profile=light_profiles.LightProfile()).has_profile is True
        assert galaxy.Galaxy(mass_profile=mass_profiles.MassProfile()).has_profile is True

from auto_lens import ray_tracing
from auto_lens import galaxy
from auto_lens.profiles import geometry_profiles, mass_profiles, light_profiles

import pytest
import numpy as np

class TestLensingPlanes(object):
    def test_trivial_ordering(self):

        g0 = galaxy.Galaxy(redshift=0)
        g1 = galaxy.Galaxy(redshift=1)
        g2 = galaxy.Galaxy(redshift=2)

        lensing_planes = galaxy.LensingPlanes(galaxies=[g0, g1, g2])

        assert lensing_planes == [g0, g1, g2]

    def test_reverse_ordering(self):

        g0 = galaxy.Galaxy(redshift=0)
        g1 = galaxy.Galaxy(redshift=1)
        g2 = galaxy.Galaxy(redshift=2)

        lensing_planes = galaxy.LensingPlanes(galaxies=[g2, g1, g0])

        assert lensing_planes == [g0, g1, g2]

    def test_out_of_order(self):

        g0 = galaxy.Galaxy(redshift=0)
        g1 = galaxy.Galaxy(redshift=1)
        g2 = galaxy.Galaxy(redshift=2)

        lensing_planes = galaxy.LensingPlanes(galaxies=[g0, g2, g1])

        assert lensing_planes == [g0, g1, g2]

    def test_angular_diameter_distances(self):

        g0 = galaxy.Galaxy(redshift=0.1)
        g1 = galaxy.Galaxy(redshift=1)
        g2 = galaxy.Galaxy(redshift=2)

        lensing_planes = galaxy.LensingPlanes(galaxies=[g0, g2, g1], cosmological_model=cosmology.Planck15)

        assert lensing_planes[0].arcsec_per_kpc == pytest.approx(0.525060, 1e-5)
        assert lensing_planes[0].kpc_per_arcsec == pytest.approx(1.904544, 1e-5)

        assert lensing_planes[0].ang_to_earth_kpc == pytest.approx(392840, 1e-5)
        assert lensing_planes[0].ang_to_next_galaxy_kpc == pytest.approx(1481890.4, 1e-5)

        assert lensing_planes[0].ang_to_earth == pytest.approx(206264, 1e-5)
        assert lensing_planes[0].ang_to_next_galaxy == pytest.approx(778081.4, 1e-5)


        assert lensing_planes[1].arcsec_per_kpc == pytest.approx(0.121478, 1e-5)
        assert lensing_planes[1].kpc_per_arcsec == pytest.approx(8.231907, 1e-5)

        assert lensing_planes[1].ang_to_earth_kpc == pytest.approx(1697952, 1e-5)
        assert lensing_planes[1].ang_to_previous_galaxy_kpc == pytest.approx(1481890.4, 1e-5)
        assert lensing_planes[1].ang_to_next_galaxy_kpc == pytest.approx(638544, 1e-5)

        assert lensing_planes[1].ang_to_earth == pytest.approx(206264, 1e-5)
        assert lensing_planes[1].ang_to_previous_galaxy == pytest.approx(180017, 1e-5)
        assert lensing_planes[1].ang_to_next_galaxy == pytest.approx(77569, 1e-5)


        assert lensing_planes[2].arcsec_per_kpc == pytest.approx(0.1165000, 1e-5)
        assert lensing_planes[2].kpc_per_arcsec == pytest.approx(8.583688, 1e-5)

        assert lensing_planes[2].ang_to_earth_kpc == pytest.approx(1770513, 1e-5)
        assert lensing_planes[2].ang_to_previous_galaxy_kpc == pytest.approx(638544, 1e-5)

        assert lensing_planes[2].ang_to_earth == pytest.approx(206264, 1e-5)
        assert lensing_planes[2].ang_to_previous_galaxy == pytest.approx(74390, 1e-5)

    def test_critical_densitieis(self):

        g0 = galaxy.Galaxy(redshift=0.1)
        g1 = galaxy.Galaxy(redshift=1)

        lensing_planes = galaxy.LensingPlanes(galaxies=[g0, g1], cosmological_model=cosmology.Planck15)

        assert lensing_planes[0].critical_density_kpc == pytest.approx(4.85e9, 1e-2)
        assert lensing_planes[0].critical_density == pytest.approx(17593241668, 1e-2)


class TestLightProfiles(object):
    
    class TestIntensity:

        def test__one_profile_galaxy__intensity_is_same_individual_profile(self):

            sersic = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                       effective_radius=0.6, sersic_index=4.0)

            sersic_intensity = sersic.intensity_at_coordinates(np.array([1.05, -0.55]))
            
            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[
                     light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0, 
                                                     effective_radius=0.6, sersic_index=4.0)])

            galaxy_sersic_intensity = galaxy_sersic.intensity_at_coordinates(np.array([1.05, -0.55]))

            assert sersic_intensity == galaxy_sersic_intensity

        def test__two_profile_galaxy__intensity_is_sum_of_individual_profiles(self):
            
            sersic_1 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                         effective_radius=2.0,
                                                         sersic_index=1.0)
    
            sersic_2 = light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)
    
            intensity = sersic_1.intensity_at_coordinates(np.array([1.05, -0.55]))
            intensity += sersic_2.intensity_at_coordinates(np.array([1.05, -0.55]))
    
            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[
                light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                  effective_radius=2.0, sersic_index=1.0),
                light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                  effective_radius=3.0, sersic_index=2.0)])
    
            galaxy_intensity = galaxy_sersic.intensity_at_coordinates([1.05, -0.55])
    
            assert intensity == galaxy_intensity
    
        def test__three_profile_galaxy__intensity_is_sum_of_individual_profiles(self):
            sersic_1 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                         effective_radius=2.0,
                                                         sersic_index=1.0)
    
            sersic_2 = light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)
    
            sersic_3 = light_profiles.SersicLightProfile(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)
    
            intensity = sersic_1.intensity_at_coordinates(np.array([1.05, -0.55]))
            intensity += sersic_2.intensity_at_coordinates(np.array([1.05, -0.55]))
            intensity += sersic_3.intensity_at_coordinates(np.array([1.05, -0.55]))
    
            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[
                light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                  effective_radius=2.0, sersic_index=1.0),
                light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                  effective_radius=3.0, sersic_index=2.0),
                light_profiles.SersicLightProfile(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                  effective_radius=3.0, sersic_index=2.0)])
    
            galaxy_intensity = galaxy_sersic.intensity_at_coordinates(np.array([1.05, -0.55]))
    
            assert intensity == galaxy_intensity
    
        def test__three_profile_galaxy__individual_intensities_can_be_extracted(self):
            sersic_1 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                         effective_radius=2.0,
                                                         sersic_index=1.0)
    
            sersic_2 = light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)
    
            sersic_3 = light_profiles.SersicLightProfile(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)
    
            intensity_1 = sersic_1.intensity_at_coordinates(np.array([1.05, -0.55]))
            intensity_2 = sersic_2.intensity_at_coordinates(np.array([1.05, -0.55]))
            intensity_3 = sersic_3.intensity_at_coordinates(np.array([1.05, -0.55]))
    
            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[
                light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                  effective_radius=2.0, sersic_index=1.0),
                light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                  effective_radius=3.0, sersic_index=2.0),
                light_profiles.SersicLightProfile(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                  effective_radius=3.0, sersic_index=2.0)])
    
            galaxy_intensity = galaxy_sersic.intensity_at_coordinates_individual((1.05, -0.55))
    
            assert intensity_1 == galaxy_intensity[0]
            assert intensity_2 == galaxy_intensity[1]
            assert intensity_3 == galaxy_intensity[2]
    
    class TestLuminosityWithinCircle:

        def test__one_profile_galaxy__integral_is_same_as_individual_profile(self):

            sersic = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0, effective_radius=2.0,
                                                       sersic_index=1.0)

            integral_radius = 5.5

            intensity_integral = sersic.luminosity_within_circle(radius=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5,
                                            light_profiles=
                                            [light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                                              effective_radius=2.0, sersic_index=1.0)])

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_circle(radius=integral_radius)

            assert intensity_integral == galaxy_intensity_integral

        def test__two_profile_galaxy__integral_is_sum_of_individual_profiles(self):
            
            sersic_1 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                         effective_radius=2.0,
                                                         sersic_index=1.0)

            sersic_2 = light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral = sersic_1.luminosity_within_circle(radius=integral_radius)
            intensity_integral += sersic_2.luminosity_within_circle(radius=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[
                light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                  effective_radius=2.0, sersic_index=1.0),
                light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                  effective_radius=3.0, sersic_index=2.0)])

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_circle(radius=integral_radius)

            assert intensity_integral == galaxy_intensity_integral

        def test__three_profile_galaxy__integral_is_sum_of_individual_profiles(self):

            sersic_1 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                         effective_radius=2.0,
                                                         sersic_index=1.0)

            sersic_2 = light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)

            sersic_3 = light_profiles.SersicLightProfile(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral = sersic_1.luminosity_within_circle(radius=integral_radius)
            intensity_integral += sersic_2.luminosity_within_circle(radius=integral_radius)
            intensity_integral += sersic_3.luminosity_within_circle(radius=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[
                light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                  effective_radius=2.0, sersic_index=1.0),
                light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                  effective_radius=3.0, sersic_index=2.0),
                light_profiles.SersicLightProfile(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                  effective_radius=3.0, sersic_index=2.0)])

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_circle(radius=integral_radius)

            assert intensity_integral == galaxy_intensity_integral

        def test__three_profile_galaxy__individual_integrals_can_be_extracted(self):
            
            sersic_1 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                         effective_radius=2.0,
                                                         sersic_index=1.0)

            sersic_2 = light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)

            sersic_3 = light_profiles.SersicLightProfile(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral_1 = sersic_1.luminosity_within_circle(radius=integral_radius)
            intensity_integral_2 = sersic_2.luminosity_within_circle(radius=integral_radius)
            intensity_integral_3 = sersic_3.luminosity_within_circle(radius=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[
                light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                  effective_radius=2.0, sersic_index=1.0),
                light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                  effective_radius=3.0, sersic_index=2.0),
                light_profiles.SersicLightProfile(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                  effective_radius=3.0, sersic_index=2.0)])

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_circle_individual(
                radius=integral_radius)

            assert intensity_integral_1 == galaxy_intensity_integral[0]
            assert intensity_integral_2 == galaxy_intensity_integral[1]
            assert intensity_integral_3 == galaxy_intensity_integral[2]

    class TestLuminosityWithinEllipse:

        def test__one_profile_galaxy__integral_is_same_as_individual_profile(self):
            
            sersic = light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0,
                                                       sersic_index=1.0)

            integral_radius = 0.5

            intensity_integral = sersic.luminosity_within_ellipse(major_axis=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[light_profiles.SersicLightProfile(axis_ratio=0.5,
                                            phi=0.0, intensity=3.0, effective_radius=2.0, sersic_index=1.0)])

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_ellipse(major_axis=integral_radius)

            assert intensity_integral == galaxy_intensity_integral

        def test__two_profile_galaxy__integral_is_sum_of_individual_profiles(self):
            
            sersic_1 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                         effective_radius=2.0,
                                                         sersic_index=1.0)

            sersic_2 = light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral = sersic_1.luminosity_within_ellipse(major_axis=integral_radius)
            intensity_integral += sersic_2.luminosity_within_ellipse(major_axis=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[
                light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                  effective_radius=2.0, sersic_index=1.0),
                light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                  effective_radius=3.0, sersic_index=2.0)])

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_ellipse(major_axis=integral_radius)

            assert intensity_integral == galaxy_intensity_integral

        def test__three_profile_galaxy__integral_is_sum_of_individual_profiles(self):
            
            sersic_1 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                         effective_radius=2.0,
                                                         sersic_index=1.0)

            sersic_2 = light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)

            sersic_3 = light_profiles.SersicLightProfile(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral = sersic_1.luminosity_within_ellipse(major_axis=integral_radius)
            intensity_integral += sersic_2.luminosity_within_ellipse(major_axis=integral_radius)
            intensity_integral += sersic_3.luminosity_within_ellipse(major_axis=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[
                light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                  effective_radius=2.0, sersic_index=1.0),
                light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                  effective_radius=3.0, sersic_index=2.0),
                light_profiles.SersicLightProfile(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                  effective_radius=3.0, sersic_index=2.0)])

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_ellipse(major_axis=integral_radius)

            assert intensity_integral == galaxy_intensity_integral

        def test__three_profile_galaxy__individual_integrals_can_be_extracted(self):
            sersic_1 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                         effective_radius=2.0,
                                                         sersic_index=1.0)

            sersic_2 = light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)

            sersic_3 = light_profiles.SersicLightProfile(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                         effective_radius=3.0,
                                                         sersic_index=2.0)

            integral_radius = 5.5

            intensity_integral_1 = sersic_1.luminosity_within_ellipse(major_axis=integral_radius)
            intensity_integral_2 = sersic_2.luminosity_within_ellipse(major_axis=integral_radius)
            intensity_integral_3 = sersic_3.luminosity_within_ellipse(major_axis=integral_radius)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[
                light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                  effective_radius=2.0, sersic_index=1.0),
                light_profiles.SersicLightProfile(axis_ratio=0.5, phi=0.0, intensity=7.0,
                                                  effective_radius=3.0, sersic_index=2.0),
                light_profiles.SersicLightProfile(axis_ratio=0.8, phi=50.0, intensity=2.0,
                                                  effective_radius=3.0, sersic_index=2.0)])

            galaxy_intensity_integral = galaxy_sersic.luminosity_within_ellipse_individual(
                major_axis=integral_radius)

            assert intensity_integral_1 == galaxy_intensity_integral[0]
            assert intensity_integral_2 == galaxy_intensity_integral[1]
            assert intensity_integral_3 == galaxy_intensity_integral[2]

    class TestSymmetricProfiles(object):
    
        def test_1d_symmetry(self):

            sersic_1 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                  sersic_index=4.0)

            sersic_2 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                  sersic_index=4.0, centre=(100, 0))
            
            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[sersic_1, sersic_2])
    
            assert galaxy_sersic.intensity_at_coordinates(np.array([0.0, 0.0])) == galaxy_sersic.intensity_at_coordinates(np.array([100.0, 0.0]))
            assert galaxy_sersic.intensity_at_coordinates(np.array([49.0, 0.0])) == galaxy_sersic.intensity_at_coordinates(np.array([51.0, 0.0]))
    
        def test_2d_symmetry(self):

            sersic_1 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                        sersic_index=4.0)
    
            sersic_2 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                        sersic_index=4.0, centre=(100, 0))

            sersic_3 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                        sersic_index=4.0, centre=(0, 100))
    
            sersic_4 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                        sersic_index=4.0, centre=(100, 100))

            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[sersic_1, sersic_2, sersic_3, sersic_4])
    
            assert galaxy_sersic.intensity_at_coordinates(np.array([49.0, 0.0])) == \
                   pytest.approx(galaxy_sersic.intensity_at_coordinates(np.array([51.0, 0.0])), 1e-5)

            assert galaxy_sersic.intensity_at_coordinates(np.array([0.0, 49.0])) == \
                   pytest.approx(galaxy_sersic.intensity_at_coordinates(np.array([0.0, 51.0])), 1e-5)

            assert galaxy_sersic.intensity_at_coordinates(np.array([100.0, 49.0])) == \
                   pytest.approx(galaxy_sersic.intensity_at_coordinates(np.array([100.0, 51.0])), 1e-5)

            assert galaxy_sersic.intensity_at_coordinates(np.array([49.0, 49.0])) == \
                   pytest.approx(galaxy_sersic.intensity_at_coordinates(np.array([51.0, 51.0])), 1e-5)

        def test_combined_array(self):

            sersic_1 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                        sersic_index=4.0)

            sersic_2 = light_profiles.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                        sersic_index=4.0)

            galaxy_sersic = galaxy.Galaxy(redshift=0.5, light_profiles=[sersic_1, sersic_2])

            assert all(map(lambda i: i == 2,
                           geometry_profiles.array_function(galaxy_sersic.intensity_at_coordinates)().flatten() /
                           geometry_profiles.array_function(sersic_1.intensity_at_coordinates)().flatten()))


class TestMassProfiles(object):

    class TestSurfaceDensity:

        def test__one_profile_galaxy__surface_density_is_same_individual_profile(self):
            sie = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)

            sie_surface_density = sie.surface_density_at_coordinates(np.array([1.05, -0.55]))

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)])

            galaxy_sie_surface_density = galaxy_sie.surface_density_at_coordinates(np.array([1.05, -0.55]))

            assert sie_surface_density == galaxy_sie_surface_density

        def test__two_profile_galaxy__surface_density_is_sum_of_individual_profiles(self):

            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)

            surface_density = sie_1.surface_density_at_coordinates(np.array([1.05, -0.55]))
            surface_density += sie_2.surface_density_at_coordinates(np.array([1.05, -0.55]))

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)])

            galaxy_surface_density = galaxy_sie.surface_density_at_coordinates(np.array([1.05, -0.55]))

            assert surface_density == galaxy_surface_density

        def test__three_profile_galaxy__surface_density_is_sum_of_individual_profiles(self):

            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)
            sie_3 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)

            surface_density = sie_1.surface_density_at_coordinates(np.array([1.05, -0.55]))
            surface_density += sie_2.surface_density_at_coordinates(np.array([1.05, -0.55]))
            surface_density += sie_3.surface_density_at_coordinates(np.array([1.05, -0.55]))

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)])

            galaxy_surface_density = galaxy_sie.surface_density_at_coordinates(np.array([1.05, -0.55]))

            assert surface_density == galaxy_surface_density

        def test__three_profile_galaxy__individual_surface_densities_can_be_extracted(self):

            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)
            sie_3 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)

            surface_density_1 = sie_1.surface_density_at_coordinates(np.array([1.05, -0.55]))
            surface_density_2 = sie_2.surface_density_at_coordinates(np.array([1.05, -0.55]))
            surface_density_3 = sie_3.surface_density_at_coordinates(np.array([1.05, -0.55]))

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)])

            galaxy_surface_density = galaxy_sie.surface_density_at_coordinates_individual((1.05, -0.55))

            assert surface_density_1 == galaxy_surface_density[0]
            assert surface_density_2 == galaxy_surface_density[1]
            assert surface_density_3 == galaxy_surface_density[2]
            
    class TestPotential:

        def test__one_profile_galaxy__potential_is_same_individual_profile(self):
            sie = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)

            sie_potential = sie.potential_at_coordinates(np.array([1.05, -0.55]))

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)])

            galaxy_sie_potential = galaxy_sie.potential_at_coordinates(np.array([1.05, -0.55]))

            assert sie_potential == galaxy_sie_potential

        def test__two_profile_galaxy__potential_is_sum_of_individual_profiles(self):

            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)

            potential = sie_1.potential_at_coordinates(np.array([1.05, -0.55]))
            potential += sie_2.potential_at_coordinates(np.array([1.05, -0.55]))

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)])

            galaxy_potential = galaxy_sie.potential_at_coordinates(np.array([1.05, -0.55]))

            assert potential == galaxy_potential

        def test__three_profile_galaxy__potential_is_sum_of_individual_profiles(self):

            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)
            sie_3 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)

            potential = sie_1.potential_at_coordinates(np.array([1.05, -0.55]))
            potential += sie_2.potential_at_coordinates(np.array([1.05, -0.55]))
            potential += sie_3.potential_at_coordinates(np.array([1.05, -0.55]))

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)])

            galaxy_potential = galaxy_sie.potential_at_coordinates(np.array([1.05, -0.55]))

            assert potential == galaxy_potential

        def test__three_profile_galaxy__individual_potentials_can_be_extracted(self):

            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)
            sie_3 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)

            potential_1 = sie_1.potential_at_coordinates(np.array([1.05, -0.55]))
            potential_2 = sie_2.potential_at_coordinates(np.array([1.05, -0.55]))
            potential_3 = sie_3.potential_at_coordinates(np.array([1.05, -0.55]))

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)])

            galaxy_potential = galaxy_sie.potential_at_coordinates_individual((1.05, -0.55))

            assert potential_1 == galaxy_potential[0]
            assert potential_2 == galaxy_potential[1]
            assert potential_3 == galaxy_potential[2]
            
    class TestDeflectionAngles:

        def test__one_profile_galaxy__deflection_angles_is_same_individual_profile(self):
            sie = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)

            sie_deflection_angles = sie.deflection_angles_at_coordinates(np.array([1.05, -0.55]))

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)])

            galaxy_sie_deflection_angles = galaxy_sie.deflection_angles_at_coordinates(np.array([1.05, -0.55]))

            assert sie_deflection_angles[0] == galaxy_sie_deflection_angles[0]
            assert sie_deflection_angles[1] == galaxy_sie_deflection_angles[1]

        def test__two_profile_galaxy__deflection_angles_is_sum_of_individual_profiles(self):

            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)

            deflection_angles_1 = sie_1.deflection_angles_at_coordinates(np.array([1.05, -0.55]))
            deflection_angles_2 = sie_2.deflection_angles_at_coordinates(np.array([1.05, -0.55]))

            deflection_angles = (deflection_angles_1[0]+deflection_angles_2[0],
                                 deflection_angles_1[1]+deflection_angles_2[1])

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)])

            galaxy_deflection_angles = galaxy_sie.deflection_angles_at_coordinates(np.array([1.05, -0.55]))

            assert deflection_angles[0] == galaxy_deflection_angles[0]
            assert deflection_angles[1] == galaxy_deflection_angles[1]

        def test__three_profile_galaxy__deflection_angles_is_sum_of_individual_profiles(self):

            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)
            sie_3 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)

            deflection_angles_1 = sie_1.deflection_angles_at_coordinates(np.array([1.05, -0.55]))
            deflection_angles_2 = sie_2.deflection_angles_at_coordinates(np.array([1.05, -0.55]))
            deflection_angles_3 = sie_3.deflection_angles_at_coordinates(np.array([1.05, -0.55]))

            deflection_angles = (deflection_angles_1[0]+deflection_angles_2[0]+deflection_angles_3[0],
                                 deflection_angles_1[1]+deflection_angles_2[1]+deflection_angles_3[1])

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)])

            galaxy_deflection_angles = galaxy_sie.deflection_angles_at_coordinates(np.array([1.05, -0.55]))

            assert deflection_angles[0] == galaxy_deflection_angles[0]
            assert deflection_angles[1] == galaxy_deflection_angles[1]

        def test__three_profile_galaxy__individual_deflection_angles_can_be_extracted(self):

            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)
            sie_3 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)

            deflection_angles_1 = sie_1.deflection_angles_at_coordinates(np.array([1.05, -0.55]))
            deflection_angles_2 = sie_2.deflection_angles_at_coordinates(np.array([1.05, -0.55]))
            deflection_angles_3 = sie_3.deflection_angles_at_coordinates(np.array([1.05, -0.55]))

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)])

            galaxy_deflection_angles = galaxy_sie.deflection_angles_at_coordinates_individual(np.array([1.05, -0.55]))

            assert (deflection_angles_1 == galaxy_deflection_angles[0]).all()
            assert (deflection_angles_2 == galaxy_deflection_angles[1]).all()
            assert (deflection_angles_3 == galaxy_deflection_angles[2]).all()

    # TODO : At the moment, these tests set the critical density =1.0 to pass. Once we have properly set up our \
    # TODO : LensingPlanes class, we should fix this to be less by-hand.

    class TestMassWithinCircle:

        def test__one_profile_galaxy__integral_is_same_as_individual_profile(self):

            integral_radius = 5.5
            
            sie = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)

            mass_integral = sie.dimensionless_mass_within_circle(radius=integral_radius)

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)])

            galaxy_sie.critical_density = 1.0

            galaxy_mass_integral = galaxy_sie.mass_within_circles(radius=integral_radius)

            assert mass_integral == galaxy_mass_integral

        def test__two_profile_galaxy__integral_is_sum_of_individual_profiles(self):

            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)

            integral_radius = 5.5

            mass_integral = sie_1.dimensionless_mass_within_circle(radius=integral_radius)
            mass_integral += sie_2.dimensionless_mass_within_circle(radius=integral_radius)

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)])

            galaxy_sie.critical_density = 1.0

            galaxy_mass_integral = galaxy_sie.mass_within_circles(radius=integral_radius)

            assert mass_integral == galaxy_mass_integral

        def test__three_profile_galaxy__integral_is_sum_of_individual_profiles(self):
            
            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)
            sie_3 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)
            
            integral_radius = 5.5

            mass_integral = sie_1.dimensionless_mass_within_circle(radius=integral_radius)
            mass_integral += sie_2.dimensionless_mass_within_circle(radius=integral_radius)
            mass_integral += sie_3.dimensionless_mass_within_circle(radius=integral_radius)

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)])

            galaxy_sie.critical_density = 1.0

            galaxy_mass_integral = galaxy_sie.mass_within_circles(radius=integral_radius)

            assert mass_integral == galaxy_mass_integral

        def test__three_profile_galaxy__individual_integrals_can_be_extracted(self):
            
            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)
            sie_3 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)

            integral_radius = 5.5

            mass_integral_1 = sie_1.dimensionless_mass_within_circle(radius=integral_radius)
            mass_integral_2 = sie_2.dimensionless_mass_within_circle(radius=integral_radius)
            mass_integral_3 = sie_3.dimensionless_mass_within_circle(radius=integral_radius)

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)])

            galaxy_sie.critical_density = 1.0

            galaxy_mass_integral = galaxy_sie.mass_within_circles_individual(
                radius=integral_radius)

            assert mass_integral_1 == galaxy_mass_integral[0]
            assert mass_integral_2 == galaxy_mass_integral[1]
            assert mass_integral_3 == galaxy_mass_integral[2]

    class TestMassWithinEllipse:

        def test__one_profile_galaxy__integral_is_same_as_individual_profile(self):

            sie = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)

            integral_radius = 0.5

            dimensionless_mass_integral = sie.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)])

            galaxy_sie.critical_density = 1.0

            galaxy_dimensionless_mass_integral = galaxy_sie.mass_within_ellipses(major_axis=integral_radius)

            assert dimensionless_mass_integral == galaxy_dimensionless_mass_integral

        def test__two_profile_galaxy__integral_is_sum_of_individual_profiles(self):

            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)

            integral_radius = 5.5

            dimensionless_mass_integral = sie_1.dimensionless_mass_within_ellipse(major_axis=integral_radius)
            dimensionless_mass_integral += sie_2.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)])

            galaxy_sie.critical_density = 1.0

            galaxy_dimensionless_mass_integral = galaxy_sie.mass_within_ellipses(major_axis=integral_radius)

            assert dimensionless_mass_integral == galaxy_dimensionless_mass_integral

        def test__three_profile_galaxy__integral_is_sum_of_individual_profiles(self):

            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)
            sie_3 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)

            integral_radius = 5.5

            dimensionless_mass_integral = sie_1.dimensionless_mass_within_ellipse(major_axis=integral_radius)
            dimensionless_mass_integral += sie_2.dimensionless_mass_within_ellipse(major_axis=integral_radius)
            dimensionless_mass_integral += sie_3.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)])

            galaxy_sie.critical_density = 1.0

            galaxy_dimensionless_mass_integral = galaxy_sie.mass_within_ellipses(major_axis=integral_radius)

            assert dimensionless_mass_integral == galaxy_dimensionless_mass_integral

        def test__three_profile_galaxy__individual_integrals_can_be_extracted(self):

            sie_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0)
            sie_2 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2)
            sie_3 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)

            integral_radius = 5.5

            dimensionless_mass_integral_1 = sie_1.dimensionless_mass_within_ellipse(major_axis=integral_radius)
            dimensionless_mass_integral_2 = sie_2.dimensionless_mass_within_ellipse(major_axis=integral_radius)
            dimensionless_mass_integral_3 = sie_3.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            galaxy_sie = galaxy.Galaxy(redshift=0.5, mass_profiles=[
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.8, phi=10.0, einstein_radius=1.0),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.6, phi=30.0, einstein_radius=1.2),
                mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.9, phi=130.0, einstein_radius=1.6)])

            galaxy_sie.critical_density = 1.0

            galaxy_dimensionless_mass_integral = galaxy_sie.mass_within_ellipses_individual(
                major_axis=integral_radius)

            assert dimensionless_mass_integral_1 == galaxy_dimensionless_mass_integral[0]
            assert dimensionless_mass_integral_2 == galaxy_dimensionless_mass_integral[1]
            assert dimensionless_mass_integral_3 == galaxy_dimensionless_mass_integral[2]
            
    class TestSymmetricProfiles:

        def test_1d_symmetry(self):
            
            isothermal_1 = mass_profiles.EllipticalIsothermalMassProfile(axis_ratio=0.5, phi=45.0,
                                                                       einstein_radius=1.0)
            
            isothermal_2 = mass_profiles.EllipticalIsothermalMassProfile(centre=(100, 0), axis_ratio=0.5, phi=45.0,
                                                                       einstein_radius=1.0)

            galaxy_isothermal = galaxy.Galaxy(redshift=0.5, mass_profiles=[isothermal_1, isothermal_2])

            assert galaxy_isothermal.surface_density_at_coordinates(np.array([1.0, 0.0])) == \
                   galaxy_isothermal.surface_density_at_coordinates(np.array([99.0, 0.0]))

            assert galaxy_isothermal.surface_density_at_coordinates(np.array([49.0, 0.0])) == \
                   galaxy_isothermal.surface_density_at_coordinates(np.array([51.0, 0.0]))
            
            assert galaxy_isothermal.potential_at_coordinates(np.array([1.0, 0.0])) == \
                   pytest.approx(galaxy_isothermal.potential_at_coordinates(np.array([99.0, 0.0])), 1e-6)

            assert galaxy_isothermal.potential_at_coordinates(np.array([49.0, 0.0])) == \
                   pytest.approx(galaxy_isothermal.potential_at_coordinates(np.array([51.0, 0.0])), 1e-6)

            assert galaxy_isothermal.deflection_angles_at_coordinates(np.array([1.0, 0.0])) == \
                   pytest.approx(galaxy_isothermal.deflection_angles_at_coordinates(np.array([99.0, 0.0])), 1e-6)

            assert galaxy_isothermal.deflection_angles_at_coordinates(np.array([49.0, 0.0])) == \
                   pytest.approx(galaxy_isothermal.deflection_angles_at_coordinates(np.array([51.0, 0.0])), 1e-6)

        def test_2d_symmetry(self):
            isothermal_1 = mass_profiles.SphericalIsothermalMassProfile(einstein_radius=1.0)

            isothermal_2 = mass_profiles.SphericalIsothermalMassProfile(centre=(100, 0), einstein_radius=1.0)

            isothermal_3 = mass_profiles.SphericalIsothermalMassProfile(centre=(0, 100), einstein_radius=1.0)

            isothermal_4 = mass_profiles.SphericalIsothermalMassProfile(centre=(100, 100), einstein_radius=1.0)

            galaxy_isothermal = galaxy.Galaxy(redshift=0.5, 
                                              mass_profiles=[isothermal_1, isothermal_2, isothermal_3, isothermal_4])

            assert galaxy_isothermal.surface_density_at_coordinates(np.array([49.0, 0.0])) == pytest.approx(
                galaxy_isothermal.surface_density_at_coordinates(np.array([51.0, 0.0])), 1e-5)
            
            assert galaxy_isothermal.surface_density_at_coordinates(np.array([0.0, 49.0])) == pytest.approx(
                galaxy_isothermal.surface_density_at_coordinates(np.array([0.0, 51.0])), 1e-5)
            
            assert galaxy_isothermal.surface_density_at_coordinates(np.array([100.0, 49.0])) == pytest.approx(
                galaxy_isothermal.surface_density_at_coordinates(np.array([100.0, 51.0])), 1e-5)
            
            assert galaxy_isothermal.surface_density_at_coordinates(np.array([49.0, 49.0])) == pytest.approx(
                galaxy_isothermal.surface_density_at_coordinates(np.array([51.0, 51.0])), 1e-5)


            assert galaxy_isothermal.potential_at_coordinates(np.array([49.0, 0.0])) == pytest.approx(
                galaxy_isothermal.potential_at_coordinates(np.array([51.0, 0.0])), 1e-5)

            assert galaxy_isothermal.potential_at_coordinates(np.array([0.0, 49.0])) == pytest.approx(
                galaxy_isothermal.potential_at_coordinates(np.array([0.0, 51.0])), 1e-5)

            assert galaxy_isothermal.potential_at_coordinates(np.array([100.0, 49.0])) == pytest.approx(
                galaxy_isothermal.potential_at_coordinates(np.array([100.0, 51.0])), 1e-5)

            assert galaxy_isothermal.potential_at_coordinates(np.array([49.0, 49.0])) == pytest.approx(
                galaxy_isothermal.potential_at_coordinates(np.array([51.0, 51.0])), 1e-5)


            assert -1.0*galaxy_isothermal.deflection_angles_at_coordinates(np.array([49.0, 0.0]))[0] == pytest.approx(
                galaxy_isothermal.deflection_angles_at_coordinates(np.array([51.0, 0.0]))[0], 1e-5)

            assert 1.0*galaxy_isothermal.deflection_angles_at_coordinates(np.array([0.0, 49.0]))[0] == pytest.approx(
                galaxy_isothermal.deflection_angles_at_coordinates(np.array([0.0, 51.0]))[0], 1e-5)

            assert 1.0*galaxy_isothermal.deflection_angles_at_coordinates(np.array([100.0, 49.0]))[0] == pytest.approx(
                galaxy_isothermal.deflection_angles_at_coordinates(np.array([100.0, 51.0]))[0], 1e-5)

            assert -1.0*galaxy_isothermal.deflection_angles_at_coordinates(np.array([49.0, 49.0]))[0] == pytest.approx(
                galaxy_isothermal.deflection_angles_at_coordinates(np.array([51.0, 51.0]))[0], 1e-5)

            assert 1.0*galaxy_isothermal.deflection_angles_at_coordinates(np.array([49.0, 0.0]))[1] == pytest.approx(
                galaxy_isothermal.deflection_angles_at_coordinates(np.array([51.0, 0.0]))[1], 1e-5)

            assert -1.0*galaxy_isothermal.deflection_angles_at_coordinates(np.array([0.0, 49.0]))[1] == pytest.approx(
                galaxy_isothermal.deflection_angles_at_coordinates(np.array([0.0, 51.0]))[1], 1e-5)

            assert -1.0*galaxy_isothermal.deflection_angles_at_coordinates(np.array([100.0, 49.0]))[1] == pytest.approx(
                galaxy_isothermal.deflection_angles_at_coordinates(np.array([100.0, 51.0]))[1], 1e-5)

            assert -1.0*galaxy_isothermal.deflection_angles_at_coordinates(np.array([49.0, 49.0]))[1] == pytest.approx(
                galaxy_isothermal.deflection_angles_at_coordinates(np.array([51.0, 51.0]))[1], 1e-5)
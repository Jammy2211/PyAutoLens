import image
from profiles import geometry_profiles, light_profiles, mass_profiles
import galaxy
from analysis import analysis_data
from analysis import ray_tracing
import pytest

import numpy as np

class TestCalcDefl(object):

    def test__simmple_sis_mass_profile__analysis_data_coordinates(self):

        coordinates = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        sis = mass_profiles.SphericalIsothermalMassProfile(einstein_radius=1.0)
        lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profiles=[sis])
        
        ray_tracing_defls = ray_tracing.compute_deflection_angles(coordinates, lens_galaxy)
        
        assert ray_tracing_defls == pytest.approx(np.array([[0.707, 0.707], [1.0, 0.0], [0.0, 1.0]]), 1e-3)

class TestDeflArray(object):

    def test__analysis_data_coordinates__simple_mass_model__deflection_angles_follow_coordinate_structure(self):

        coordinates = np.array([[1.0, 0.0], [0.0, 1.0]])

        sis = mass_profiles.SphericalIsothermalMassProfile(einstein_radius=1.0)

        lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profiles=[sis])

        ray_tracing_defls = ray_tracing.deflection_angles_analysis_array(lens_galaxy, coordinates)

        assert ray_tracing_defls == pytest.approx(np.array([[1.0, 0.0], [0.0, 1.0]]), 1e-3)

    def test__analysis_data_coordinates__complex_mass_model__deflection_angles_follow_coordinate_structure(self):

        coordinates = np.array([[1.0, 0.0], [0.0, 1.0]])

        power_law = mass_profiles.EllipticalPowerLawMassProfile(centre=(1.0, 4.0), axis_ratio=0.7, phi=30.0,
                                                                einstein_radius=1.0, slope=2.2)
        nfw = mass_profiles.SphericalNFWMassProfile(kappa_s=0.1, scale_radius=5.0)

        defls_0 = power_law.deflection_angles_at_coordinates(coordinates[0]) + \
                  nfw.deflection_angles_at_coordinates(coordinates[0])

        defls_1 = power_law.deflection_angles_at_coordinates(coordinates[1]) + \
                  nfw.deflection_angles_at_coordinates(coordinates[1])

        lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profiles=[power_law, nfw])

        ray_tracing_defls = ray_tracing.deflection_angles_analysis_array(lens_galaxy, coordinates)

        assert ray_tracing_defls == pytest.approx(np.array([defls_0, defls_1]), 1e-3)

    def test__analysis_data_sub_coordinates__sub_coordinates_structure(self):

        sub_coordinates = np.array([[[1.0, 0.0], [0.0, 1.0]],
                                    [[0.0, 1.0], [1.0, 0.0]]])

        sis = mass_profiles.SphericalIsothermalMassProfile(einstein_radius=1.0)

        lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profiles=[sis])

        ray_tracing_defls = ray_tracing.deflection_angles_analysis_sub_array(lens_galaxy, sub_coordinates)

        assert ray_tracing_defls[0,0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
        assert ray_tracing_defls[0,1] == pytest.approx(np.array([0.0, 1.0]), 1e-3)
        assert ray_tracing_defls[1,0] == pytest.approx(np.array([0.0, 1.0]), 1e-3)
        assert ray_tracing_defls[1,1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        assert ray_tracing.deflection_angles_analysis_sub_array(lens_galaxy, sub_coordinates) ==\
               pytest.approx(np.array([[[1.0, 0.0], [0.0, 1.0]],
                                       [[0.0, 1.0], [1.0, 0.0]]]), 1e-3)

    def test__analysis_data_coordinates__simple_mass_model__deflection_angles_follow_sub_coordinate_structure(self):

        sub_coordinates = np.array([[[1.0, 0.0], [0.0, 1.0]],
                                    [[0.0, 1.0], [1.0, 0.0]]])

        power_law = mass_profiles.EllipticalPowerLawMassProfile(centre=(1.0, 4.0), axis_ratio=0.7, phi=30.0,
                                                                einstein_radius=1.0, slope=2.2)
        nfw = mass_profiles.SphericalNFWMassProfile(kappa_s=0.1, scale_radius=5.0)

        defls_0 = power_law.deflection_angles_at_coordinates(sub_coordinates[0,0]) + \
                            nfw.deflection_angles_at_coordinates(sub_coordinates[0,0])

        defls_1 = power_law.deflection_angles_at_coordinates(sub_coordinates[0,1]) + \
                              nfw.deflection_angles_at_coordinates(sub_coordinates[0,1])

        defls_2 = power_law.deflection_angles_at_coordinates(sub_coordinates[1,0]) + \
                              nfw.deflection_angles_at_coordinates(sub_coordinates[1,0])

        defls_3 = power_law.deflection_angles_at_coordinates(sub_coordinates[1,1]) + \
                              nfw.deflection_angles_at_coordinates(sub_coordinates[1,1])

        lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profiles=[power_law, nfw])

        ray_tracing_defls = ray_tracing.deflection_angles_analysis_sub_array(lens_galaxy, sub_coordinates)

        assert ray_tracing_defls[0,0] == pytest.approx(defls_0, 1e-3)
        assert ray_tracing_defls[0,1] == pytest.approx(defls_1, 1e-3)
        assert ray_tracing_defls[1,0] == pytest.approx(defls_2, 1e-3)
        assert ray_tracing_defls[1,1] == pytest.approx(defls_3, 1e-3)

# class TestLightProfileCalc(object):
#
#     def test__sersic_galaxy_in__computes_correct_values(self):
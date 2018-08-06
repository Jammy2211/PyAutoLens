from autolens.profiles import mass_profiles, light_profiles
import pytest
import numpy as np


class TestConstructors(object):

    def test__setup_elliptical_power_law(self):
        power_law = mass_profiles.EllipticalPowerLaw(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                     einstein_radius=1.0, slope=2.0)

        assert power_law.x_cen == 1.0
        assert power_law.y_cen == 1.0
        assert power_law.axis_ratio == 1.0
        assert power_law.phi == 45.0
        assert power_law.einstein_radius == 1.0
        assert power_law.slope == 2.0
        assert power_law.einstein_radius_rescaled == 0.5  # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5

    def test__setup_spherical_power_law(self):
        power_law = mass_profiles.SphericalPowerLaw(centre=(1, 1), einstein_radius=1.0, slope=2.0)

        assert power_law.x_cen == 1.0
        assert power_law.y_cen == 1.0
        assert power_law.axis_ratio == 1.0
        assert power_law.phi == 0.0
        assert power_law.einstein_radius == 1.0
        assert power_law.slope == 2.0
        assert power_law.einstein_radius_rescaled == 0.5  # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5

    def test__setup_cored_elliptical_power_law(self):
        power_law = mass_profiles.EllipticalCoredPowerLaw(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                          einstein_radius=1.0, slope=2.2, core_radius=0.1)

        assert power_law.x_cen == 1.0
        assert power_law.y_cen == 1.0
        assert power_law.axis_ratio == 1.0
        assert power_law.phi == 45.0
        assert power_law.einstein_radius == 1.0
        assert power_law.slope == 2.2
        assert power_law.core_radius == 0.1
        # (3 - slope) / (1 + axis_ratio) * (1.0) = (3 - 2) / (1 + 1) * (1.1)**1.2 = 0.5
        assert power_law.einstein_radius_rescaled == pytest.approx(0.4, 1e-3)

    def test__setup_cored_spherical_power_law(self):
        power_law = mass_profiles.SphericalCoredPowerLaw(centre=(1, 1), einstein_radius=1.0, slope=2.2,
                                                         core_radius=0.1)

        assert power_law.x_cen == 1.0
        assert power_law.y_cen == 1.0
        assert power_law.axis_ratio == 1.0
        assert power_law.phi == 0.0
        assert power_law.einstein_radius == 1.0
        assert power_law.slope == 2.2
        assert power_law.core_radius == 0.1
        assert power_law.einstein_radius_rescaled == pytest.approx(0.4, 1e-3)

    def test__setup_elliptical_isothermal(self):
        isothermal = mass_profiles.EllipticalIsothermal(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                        einstein_radius=1.0)

        assert isothermal.x_cen == 1.0
        assert isothermal.y_cen == 1.0
        assert isothermal.axis_ratio == 1.0
        assert isothermal.phi == 45.0
        assert isothermal.einstein_radius == 1.0
        assert isothermal.slope == 2.0
        assert isothermal.einstein_radius_rescaled == 0.5  # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5

    def test__setup_spherical_isothermal(self):
        isothermal = mass_profiles.SphericalIsothermal(centre=(1, 1), einstein_radius=1.0)

        assert isothermal.x_cen == 1.0
        assert isothermal.y_cen == 1.0
        assert isothermal.axis_ratio == 1.0
        assert isothermal.phi == 0.0
        assert isothermal.einstein_radius == 1.0
        assert isothermal.slope == 2.0
        assert isothermal.einstein_radius_rescaled == 0.5  # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5

    def test__setup_elliptical_isothermal_core(self):
        isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                                  einstein_radius=1.0, core_radius=0.2)

        assert isothermal_core.x_cen == 1.0
        assert isothermal_core.y_cen == 1.0
        assert isothermal_core.axis_ratio == 1.0
        assert isothermal_core.phi == 45.0
        assert isothermal_core.einstein_radius == 1.0
        assert isothermal_core.slope == 2.0
        assert isothermal_core.core_radius == 0.2
        # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5
        assert isothermal_core.einstein_radius_rescaled == 0.5

    def test__setup_spherical_isothermal_core(self):
        isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(1, 1),
                                                                 einstein_radius=1.0, core_radius=0.2)

        assert isothermal_core.x_cen == 1.0
        assert isothermal_core.y_cen == 1.0
        assert isothermal_core.axis_ratio == 1.0
        assert isothermal_core.phi == 0.0
        assert isothermal_core.einstein_radius == 1.0
        assert isothermal_core.slope == 2.0
        assert isothermal_core.core_radius == 0.2
        # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5
        assert isothermal_core.einstein_radius_rescaled == 0.5

    def test__setup_elliptical_nfw(self):
        nfw = mass_profiles.EllipticalNFW(centre=(0.7, 1.0), axis_ratio=0.7, phi=60.0, kappa_s=2.0,
                                          scale_radius=10.0)

        assert nfw.centre == (0.7, 1.0)
        assert nfw.axis_ratio == 0.7
        assert nfw.phi == 60.0
        assert nfw.kappa_s == 2.0
        assert nfw.scale_radius == 10.0

    def test__setup_spherical_nfw(self):
        nfw = mass_profiles.SphericalNFW(centre=(0.7, 1.0), kappa_s=2.0, scale_radius=10.0)

        assert nfw.centre == (0.7, 1.0)
        assert nfw.axis_ratio == 1.0
        assert nfw.phi == 0.0
        assert nfw.kappa_s == 2.0
        assert nfw.scale_radius == 10.0

    def test__setup_elliptical_generalized_nfw(self):
        gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(0.7, 1.0), axis_ratio=0.7, phi=45.0,
                                                      kappa_s=2.0, inner_slope=1.5, scale_radius=10.0)

        assert gnfw.centre == (0.7, 1.0)
        assert gnfw.axis_ratio == 0.7
        assert gnfw.phi == 45.0
        assert gnfw.kappa_s == 2.0
        assert gnfw.inner_slope == 1.5
        assert gnfw.scale_radius == 10.0

    def test__setup_spherical_generalized_nfw(self):
        gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(0.7, 1.0),
                                                     kappa_s=2.0, inner_slope=1.5, scale_radius=10.0)

        assert gnfw.centre == (0.7, 1.0)
        assert gnfw.axis_ratio == 1.0
        assert gnfw.phi == 0.0
        assert gnfw.kappa_s == 2.0
        assert gnfw.inner_slope == 1.5
        assert gnfw.scale_radius == 10.0

    def test__setup_sersic(self):
        sersic = mass_profiles.EllipticalSersicMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                    effective_radius=0.6, sersic_index=2.0, mass_to_light_ratio=1.0)

        assert sersic.x_cen == 0.0
        assert sersic.y_cen == 0.0
        assert sersic.axis_ratio == 1.0
        assert sersic.phi == 0.0
        assert sersic.intensity == 1.0
        assert sersic.effective_radius == 0.6
        assert sersic.sersic_index == 2.0
        assert sersic.sersic_constant == pytest.approx(3.67206, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6
        assert sersic.mass_to_light_ratio == 1.0

    def test__setup_exponential(self):
        exponential = mass_profiles.EllipticalExponentialMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                              effective_radius=0.6, mass_to_light_ratio=1.0)

        assert exponential.x_cen == 0.0
        assert exponential.y_cen == 0.0
        assert exponential.axis_ratio == 1.0
        assert exponential.phi == 0.0
        assert exponential.intensity == 1.0
        assert exponential.effective_radius == 0.6
        assert exponential.sersic_index == 1.0
        assert exponential.sersic_constant == pytest.approx(1.678388, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6
        assert exponential.mass_to_light_ratio == 1.0

    def test__setup_dev_vaucouleurs(self):
        dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                     intensity=1.0,
                                                                     effective_radius=0.6, mass_to_light_ratio=1.0)

        assert dev_vaucouleurs.x_cen == 0.0
        assert dev_vaucouleurs.y_cen == 0.0
        assert dev_vaucouleurs.axis_ratio == 1.0
        assert dev_vaucouleurs.phi == 0.0
        assert dev_vaucouleurs.intensity == 1.0
        assert dev_vaucouleurs.effective_radius == 0.6
        assert dev_vaucouleurs.sersic_index == 4.0
        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6
        assert dev_vaucouleurs.mass_to_light_ratio == 1.0

    def test_component_numbers_four_profiles(self):
        # TODO : Perform Counting reset better in code

        from itertools import count

        mass_profiles.EllipticalMassProfile._ids = count()

        power_law_0 = mass_profiles.EllipticalPowerLaw(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                       einstein_radius=1.0, slope=2.0)

        power_law_1 = mass_profiles.EllipticalPowerLaw(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                       einstein_radius=1.0, slope=2.0)

        power_law_2 = mass_profiles.EllipticalPowerLaw(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                       einstein_radius=1.0, slope=2.0)

        power_law_3 = mass_profiles.EllipticalPowerLaw(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                       einstein_radius=1.0, slope=2.0)
        assert power_law_0.component_number == 0
        assert power_law_1.component_number == 1
        assert power_law_2.component_number == 2
        assert power_law_3.component_number == 3


class TestProfiles(object):

    class TestEllipticalPowerLaw(object):

        class TestSurfaceDensity(object):

            def test__flip_coordinates_lens_center__same_value(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                surface_density_0 = power_law.surface_density_from_grid(grid=np.array([[1.0, 1.0]]))

                power_law = mass_profiles.EllipticalPowerLaw(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                surface_density_1 = power_law.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_coordinates_90_circular__same_value(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                surface_density_0 = power_law.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                             einstein_radius=1.0, slope=2.0)

                surface_density_1 = power_law.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                             einstein_radius=1.0, slope=2.2)

                surface_density_0 = power_law.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                             einstein_radius=1.0, slope=2.2)

                surface_density_1 = power_law.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__simple_case__correct_value(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                surface_density = power_law.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.5, 1e-3)

            def test__double_einr__doubles_value(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                             einstein_radius=2.0, slope=2.0)

                surface_density = power_law.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.5 * 2.0, 1e-3)

            def test__different_axis_ratio__new_value(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                surface_density = power_law.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.66666, 1e-3)

            def test__slope_increase__new_value(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                             einstein_radius=1.0, slope=2.3)

                surface_density = power_law.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.466666, 1e-3)

            def test__slope_decrease__new_value(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                             einstein_radius=2.0, slope=1.7)

                surface_density = power_law.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(1.4079, 1e-3)

        class TestPotential(object):
            def test__flip_coordinates_lens_center__same_value(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                potential_0 = power_law.potential_from_grid(grid=np.array([[1.0, 1.0]]))

                power_law = mass_profiles.EllipticalPowerLaw(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                potential_1 = power_law.potential_from_grid(grid=np.array([[0.0, 0.0]]))

                assert potential_0 == potential_1

            def test__rotation_coordinates_90_circular__same_value(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                potential_0 = power_law.potential_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                             einstein_radius=1.0, slope=2.0)

                potential_1 = power_law.potential_from_grid(grid=np.array([[0.0, 1.0]]))

                assert potential_0 == potential_1

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                             einstein_radius=1.0, slope=2.2)

                potential_0 = power_law.potential_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                             einstein_radius=1.0, slope=2.2)

                potential_1 = power_law.potential_from_grid(grid=np.array([[0.0, 1.0]]))

                assert potential_0 == potential_1

            def test__compare_to_isothermal__same_potential(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.5, phi=45.0,
                                                                einstein_radius=1.0)

                potential_power_law = isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))
                
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=45.0,
                                                             einstein_radius=1.0, slope=2.0)

                potential_isothermal = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential_isothermal == pytest.approx(potential_power_law, 1e-3)

            def test__compare_to_fortran_values_slope_22__same_potential(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                             einstein_radius=1.3, slope=2.2)

                potential = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential == pytest.approx(1.53341, 1e-3)

            def test__compare_to_fortran_values_slope_20__same_potential(self):

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                             einstein_radius=1.3, slope=2.0)

                potential = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential == pytest.approx(1.19268, 1e-3)

            def test__compare_to_fortran_values_slope_18__same_potential(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                             einstein_radius=1.3, slope=1.8)

                potential = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential == pytest.approx(0.96723, 1e-3)

            def test__2_coordinates__returns_both_potentials(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                             einstein_radius=1.3, slope=1.8)

                potential = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))

                assert potential[0] == pytest.approx(0.96723, 1e-3)
                assert potential[1] == pytest.approx(0.96723, 1e-3)

        class TestDeflections(object):

            def test__flip_coordinates_lens_center__same_value(self):

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                defls_0 = power_law.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                power_law = mass_profiles.EllipticalPowerLaw(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                defls_1 = power_law.deflections_from_grid(grid=np.array([[0.0, 0.0]]))

                assert defls_0[0,0] == pytest.approx(-defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(-defls_1[0,1], 1e-5)

            def test__rotation_coordinates_90_circular__same_value(self):

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                defls_0 = power_law.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                             einstein_radius=1.0, slope=2.0)

                defls_1 = power_law.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                # Foro deflection angles, a 90 degree rtation flips the x / y image_grid

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                             einstein_radius=1.0, slope=2.2)

                defls_0 = power_law.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                             einstein_radius=1.0, slope=2.2)

                defls_1 = power_law.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__identical_as_sie_compare_ratio(self):

                isothermal = mass_profiles.EllipticalIsothermal(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                                einstein_radius=1.0)

                defls_isothermal = isothermal.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                defls_power_law = power_law.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                ratio_isothermal = defls_isothermal[0,0] / defls_isothermal[0,1]
                ratio_power_law = defls_power_law[0,0] / defls_power_law[0,1]

                assert ratio_isothermal == pytest.approx(ratio_power_law, 1e-3)
                assert defls_isothermal[0,0] == pytest.approx(defls_power_law[0,0], 1e-3)
                assert defls_isothermal[0,1] == pytest.approx(defls_power_law[0,1], 1e-3)

            def test__compare_to_fortran_slope_isothermal(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.50734, 1e-3)
                assert defls[0,1] == pytest.approx(0.79421, 1e-3)

            def test__compare_to_fortran_slope_above_isothermal(self):

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                             einstein_radius=1.0, slope=2.5)

                defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.99629, 1e-3)
                assert defls[0,1] == pytest.approx(1.29641, 1e-3)

            def test__compare_to_fortran_slope_below_isothermal(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                             einstein_radius=1.0, slope=1.5)

                defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.26729, 1e-3)
                assert defls[0,1] == pytest.approx(0.48036, 1e-3)

            def test__compare_to_fortran_different_values(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                             einstein_radius=1.3, slope=1.9)

                defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] / defls[0,1] == pytest.approx(-0.53353, 1e-3)
                assert defls[0,0] == pytest.approx(-0.60205, 1e-3)
                assert defls[0,1] == pytest.approx(1.12841, 1e-3)

            def test__compare_to_fortran_different_values_2(self):
                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.5, -0.7), axis_ratio=0.7, phi=150.0,
                                                             einstein_radius=1.3, slope=2.2)

                defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] / defls[0,1] == pytest.approx(-0.27855, 1e-3)
                assert defls[0,0] == pytest.approx(-0.35096, 1e-3)
                assert defls[0,1] == pytest.approx(1.25995, 1e-3)

            def test__2_coordinates__gives_multiple_deflection_angles(self):

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                             einstein_radius=1.0, slope=2.0)

                defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))

                assert defls[0, 0] == pytest.approx(0.50734, 1e-3)
                assert defls[0, 1] == pytest.approx(0.79421, 1e-3)
                assert defls[1, 0] == pytest.approx(0.50734, 1e-3)
                assert defls[1, 1] == pytest.approx(0.79421, 1e-3)


    class TestSphericalPowerLaw(object):
        
        class TestSurfaceDensity(object):

            def test__flip_coordinates_lens_center__same_value(self):
                isothermal = mass_profiles.SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0,
                                                             slope=2.0)

                surface_density_0 = isothermal.surface_density_from_grid(grid=np.array([[1.0, 1.0]]))

                isothermal = mass_profiles.SphericalPowerLaw(centre=(1.0, 1.0), einstein_radius=1.0,
                                                             slope=2.0)

                surface_density_1 = isothermal.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_coordinates_90_circular__same_value(self):
                isothermal = mass_profiles.SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0,
                                                             slope=2.0)

                surface_density_0 = isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal = mass_profiles.SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0,
                                                             slope=2.0)

                surface_density_1 = isothermal.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__simple_case__correct_value(self):
                isothermal = mass_profiles.SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0,
                                                             slope=2.0)

                surface_density = isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx(0.5, 1e-3)

            def test__simple_case_2__correct_value(self):
                isothermal = mass_profiles.SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=2.0,
                                                             slope=2.2)

                surface_density = isothermal.surface_density_from_grid(grid=np.array([[2.0, 0.0]]))

                assert surface_density == pytest.approx(0.4, 1e-3)

            def test__double_einr__doubles_value(self):
                isothermal = mass_profiles.SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=2.0,
                                                             slope=2.0)

                surface_density = isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.5 * 2.0, 1e-3)

        class TestPotential(object):

            def test__flip_coordinates_lens_center__same_value(self):
                power_law = mass_profiles.SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0,
                                                             slope=2.0)

                potential_0 = power_law.potential_from_grid(grid=np.array([[1.0, 1.0]]))

                power_law = mass_profiles.SphericalPowerLaw(centre=(1.0, 1.0), einstein_radius=1.0,
                                                             slope=2.0)

                potential_1 = power_law.potential_from_grid(grid=np.array([[0.0, 0.0]]))

                assert potential_0 == potential_1

            def test__compare_to_fortran_values_high_slope__same_potential(self):
                power_law = mass_profiles.SphericalPowerLaw(centre=(0.5, -0.7), einstein_radius=1.3,
                                                             slope=2.3)

                potential = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential == pytest.approx(1.90421, 1e-3)

            def test__compare_to_fortran_values_low_slope__same_potential(self):
                power_law = mass_profiles.SphericalPowerLaw(centre=(0.5, -0.7), einstein_radius=1.3,
                                                             slope=1.8)

                potential = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential == pytest.approx(0.93758, 1e-3)

            def test__compare_to_elliptical_power_law__same_values(self):
                power_law = mass_profiles.SphericalPowerLaw(centre=(0.8, -0.4), einstein_radius=3.0,
                                                             slope=1.7)

                potential_0 = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.8, -0.4), axis_ratio=1.0, phi=0.0,
                                                              einstein_radius=3.0, slope=1.7)

                potential_1 = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential_0 == pytest.approx(potential_1, 1e-4)

            def test__compare_to_spherical_power_law__same_values(self):
                power_law = mass_profiles.SphericalIsothermal(centre=(0.2, -0.4), einstein_radius=3.0)

                potential_0 = power_law.potential_from_grid(grid=np.array([[0.888, 0.888]]))

                power_law = mass_profiles.SphericalPowerLaw(centre=(0.2, -0.4), einstein_radius=3.0,
                                                            slope=2.0)

                potential_1 = power_law.potential_from_grid(grid=np.array([[0.888, 0.888]]))

                assert potential_0 == pytest.approx(potential_1, 1e-4)

            def test__2_coordinates__returns_both_potentials(self):

                power_law = mass_profiles.SphericalPowerLaw(centre=(0.5, -0.7), einstein_radius=1.3,
                                                             slope=1.8)

                potential = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))

                assert potential[0] == pytest.approx(0.93758, 1e-3)
                assert potential[1] == pytest.approx(0.93758, 1e-3)

        class TestDeflections(object):

            def test__compare_to_elliptical_power_law__same_value(self):
                isothermal = mass_profiles.EllipticalPowerLaw(centre=(0.9, 0.5), axis_ratio=1.0, phi=0.0,
                                                              einstein_radius=4.0, slope=2.3)

                defls_0 = isothermal.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                power_law = mass_profiles.SphericalPowerLaw(centre=(0.9, 0.5), einstein_radius=4.0,
                                                             slope=2.3)

                defls_1 = power_law.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,0], 1e-4)
                assert defls_0[0,1] == pytest.approx(defls_1[0,1], 1e-4)

            def test__compare_to_elliptical_power_law_2__same_value(self):
                isothermal = mass_profiles.EllipticalPowerLaw(centre=(0.1, -0.5), axis_ratio=1.0, phi=0.0,
                                                              einstein_radius=2.0, slope=1.6)

                defls_0 = isothermal.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                isothermal = mass_profiles.SphericalPowerLaw(centre=(0.1, -0.5), einstein_radius=2.0,
                                                             slope=1.6)

                defls_1 = isothermal.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,0], 1e-4)
                assert defls_0[0,1] == pytest.approx(defls_1[0,1], 1e-4)

            def test__compare_to_spherical_isothermal__same_values(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.2, -0.4), einstein_radius=3.0)

                defls_0 = isothermal.deflections_from_grid(grid=np.array([[0.888, 0.888]]))

                power_law = mass_profiles.SphericalPowerLaw(centre=(0.2, -0.4), einstein_radius=3.0,
                                                            slope=2.0)

                defls_1 = power_law.deflections_from_grid(grid=np.array([[0.888, 0.888]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,0], 1e-4)
                assert defls_0[0,1] == pytest.approx(defls_1[0,1], 1e-4)

                # TODO : Add fortran comparison

            def test__compare_to_fortran_slope_isothermal(self):
                
                power_law = mass_profiles.SphericalPowerLaw(centre=(0.2, 0.2), einstein_radius=1.0,
                                                            slope=2.0)

                defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert defls[0,0] == pytest.approx(-0.94868, 1e-3)
                assert defls[0,1] == pytest.approx(-0.31622, 1e-3)

            def test__compare_to_fortran_slope_above_isothermal(self):

                power_law = mass_profiles.SphericalPowerLaw(centre=(0.2, 0.2), einstein_radius=1.0,
                                                            slope=2.5)

                defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert defls[0,0] == pytest.approx(-4.77162, 1e-3)
                assert defls[0,1] == pytest.approx(-1.59054, 1e-3)

            def test__compare_to_fortran_slope_below_isothermal(self):
                power_law = mass_profiles.SphericalPowerLaw(centre=(0.2, 0.2), einstein_radius=1.0,
                                                            slope=1.5)

                defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert defls[0,0] == pytest.approx(-0.18861, 1e-3)
                assert defls[0,1] == pytest.approx(-0.06287, 1e-3)

            def test__2_coordinates__gives_multiple_deflection_angles(self):

                power_law = mass_profiles.SphericalPowerLaw(centre=(0.2, 0.2), einstein_radius=1.0,
                                                            slope=1.5)

                defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1875], [0.1625, 0.1875]]))

                assert defls[0, 0] == pytest.approx(-0.18861, 1e-3)
                assert defls[0, 1] == pytest.approx(-0.06287, 1e-3)
                assert defls[1, 0] == pytest.approx(-0.18861, 1e-3)
                assert defls[1, 1] == pytest.approx(-0.06287, 1e-3)


    class TestSphericalIsothermal(object):

        class TestSurfaceDensity(object):
            def test__flip_coordinates_lens_center__same_value(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

                surface_density_0 = isothermal.surface_density_from_grid(grid=np.array([[1.0, 1.0]]))

                isothermal = mass_profiles.SphericalIsothermal(centre=(1.0, 1.0), einstein_radius=1.0)

                surface_density_1 = isothermal.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_coordinates_90_circular__same_value(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

                surface_density_0 = isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal = mass_profiles.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

                surface_density_1 = isothermal.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__simple_case__correct_value(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

                surface_density = isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.5, 1e-3)

            def test__double_einr__doubles_value(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

                surface_density = isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.5 * 2.0, 1e-3)

        class TestPotential(object):
            def test__flip_coordinates_lens_center__same_value(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

                potential_0 = isothermal.potential_from_grid(grid=np.array([[1.0, 1.0]]))

                isothermal = mass_profiles.SphericalIsothermal(centre=(1.0, 1.0), einstein_radius=1.0)

                potential_1 = isothermal.potential_from_grid(grid=np.array([[0.0, 0.0]]))

                assert potential_0 == potential_1

            def test__rotation_coordinates_90_circular__same_value(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

                potential_0 = isothermal.potential_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal = mass_profiles.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

                potential_1 = isothermal.potential_from_grid(grid=np.array([[0.0, 1.0]]))

                assert potential_0 == potential_1

            def test__compare_to_elliptical_isothermal__same_values(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.8, -0.4), einstein_radius=3.0)

                potential_0 = isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.8, -0.4), axis_ratio=1.0, phi=0.0,
                                                                einstein_radius=3.0)

                potential_1 = isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential_0 == pytest.approx(potential_1, 1e-4)

            def test__compare_to_elliptical_isothermal_different_coordinates__same_values(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.8, -0.4), einstein_radius=3.0)

                potential_0 = isothermal.potential_from_grid(grid=np.array([[-1.1, 0.1625]]))

                isothermal = mass_profiles.EllipticalPowerLaw(centre=(0.8, -0.4), axis_ratio=1.0, phi=100.0,
                                                              einstein_radius=3.0, slope=2.0)

                potential_1 = isothermal.potential_from_grid(grid=np.array([[-1.1, 0.1625]]))

                assert potential_0 == pytest.approx(potential_1, 1e-4)

            def test__compare_to_fortran_values__same_potential(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.5, -0.7), einstein_radius=1.3)

                potential = isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert potential == pytest.approx(1.23435, 1e-3)

            def test__2_coordinates__returns_2_potentials(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.5, -0.7), einstein_radius=1.3)

                potential = isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1875], [0.1625, 0.1875]]))

                assert potential[0] == pytest.approx(1.23435, 1e-3)
                assert potential[1] == pytest.approx(1.23435, 1e-3)


        class TestDeflections(object):

            def test__compare_to_elliptical_isothermal__same_value(self):

                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.9, 0.5), axis_ratio=0.99999,
                                                                phi=0.0,
                                                                einstein_radius=4.0)

                defls_0 = isothermal.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                isothermal = mass_profiles.SphericalIsothermal(centre=(0.9, 0.5), einstein_radius=4.0)

                defls_1 = isothermal.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,0], 1e-4)
                assert defls_0[0,1] == pytest.approx(defls_1[0,1], 1e-4)

            def test__compare_to_fortran__same_value(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.5, -0.7), einstein_radius=1.3)

                deflections = isothermal.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert deflections[0,0] == pytest.approx(-0.46208, 1e-4)
                assert deflections[0,1] == pytest.approx(1.21510, 1e-4)

            def test__compare_to_fortran_2__same_value(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.1, -0.1), einstein_radius=5.0)

                deflections = isothermal.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert deflections[0,0] == pytest.approx(1.06214, 1e-4)
                assert deflections[0,1] == pytest.approx(4.88588, 1e-4)

                # TODO : Add fortran comparison

            def test__2_coordinates__2_sets_of_deflection_angles(self):
                isothermal = mass_profiles.SphericalIsothermal(centre=(0.1, -0.1), einstein_radius=5.0)

                deflections = isothermal.deflections_from_grid(grid=np.array([[0.1625, 0.1875],
                                                                              [0.1625, 0.1875]]))

                assert deflections[0,0] == pytest.approx(1.06214, 1e-4)
                assert deflections[0,1] == pytest.approx(4.88588, 1e-4)
                assert deflections[1,0] == pytest.approx(1.06214, 1e-4)
                assert deflections[1,1] == pytest.approx(4.88588, 1e-4)


    class TestEllipticalIsothermal(object):

        class TestSurfaceDensity(object):

            def test__flip_coordinates_lens_center__same_value(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                einstein_radius=1.0)

                surface_density_0 = isothermal.surface_density_from_grid(grid=np.array([[1.0, 1.0]]))

                isothermal = mass_profiles.EllipticalIsothermal(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                einstein_radius=1.0)

                surface_density_1 = isothermal.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_coordinates_90_circular__same_value(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                einstein_radius=1.0)

                surface_density_0 = isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                                einstein_radius=1.0)

                surface_density_1 = isothermal.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                einstein_radius=1.0)

                surface_density_0 = isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                                einstein_radius=1.0)

                surface_density_1 = isothermal.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__simple_case__correct_value(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                einstein_radius=1.0)

                surface_density = isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.5, 1e-3)

            def test__double_einr__doubles_value(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                einstein_radius=2.0)

                surface_density = isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.5 * 2.0, 1e-3)

            def test__different_axis_ratio__new_value(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                                einstein_radius=1.0)

                surface_density = isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.66666, 1e-3)

        class TestPotential(object):

            def test__flip_coordinates_lens_center__same_value(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                einstein_radius=1.0)

                potential_0 = isothermal.potential_from_grid(grid=np.array([[1.0, 1.0]]))

                isothermal = mass_profiles.EllipticalIsothermal(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                einstein_radius=1.0)

                potential_1 = isothermal.potential_from_grid(grid=np.array([[0.0, 0.0]]))

                assert potential_0 == potential_1

            def test__rotation_coordinates_90_circular__same_value(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                einstein_radius=1.0)

                potential_0 = isothermal.potential_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                                einstein_radius=1.0)

                potential_1 = isothermal.potential_from_grid(grid=np.array([[0.0, 1.0]]))

                assert potential_0 == potential_1

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                einstein_radius=1.0)

                potential_0 = isothermal.potential_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                                einstein_radius=1.0)

                potential_1 = isothermal.potential_from_grid(grid=np.array([[0.0, 1.0]]))

                assert potential_0 == potential_1

            def test__compare_to_isothermal_ratio_of_two_potentials__same_ratio(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.5, phi=45.0,
                                                                einstein_radius=1.0)

                potential_isothermal_1 = isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                                einstein_radius=1.6)

                potential_isothermal_2 = isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                isothermal_ratio = potential_isothermal_1 / potential_isothermal_2

                assert isothermal_ratio == pytest.approx(isothermal_ratio, 1e-3)

            def test__compare_to_fortran_values__same_potential(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                                einstein_radius=1.3)

                potential = isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential == pytest.approx(1.19268, 1e-3)

            def test__compare_to_power_law__same_values(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.8, -0.4), axis_ratio=0.5,
                                                                phi=170.0,
                                                                einstein_radius=3.0)

                potential_0 = isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                isothermal = mass_profiles.EllipticalPowerLaw(centre=(0.8, -0.4), axis_ratio=0.5, phi=170.0,
                                                              einstein_radius=3.0, slope=2.0)

                potential_1 = isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential_0 == potential_1

            def test__2_coordinates__2_potentials(self):

                isothermal = mass_profiles.EllipticalIsothermal(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                                einstein_radius=1.3)

                potential = isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1625],
                                                                          [0.1625, 0.1625]]))

                assert potential[0] == pytest.approx(1.19268, 1e-3)
                assert potential[1] == pytest.approx(1.19268, 1e-3)


        class TestDeflections(object):

            def test_no_coordinate_rotation__correct_values(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                                einstein_radius=1.0)

                defls = isothermal.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                # normalization = 2.0*(1/(1+q))*einr*q / (sqrt(1-q**2))
                # normalization = (1/1.5)*1*0.5 / (sqrt(0.75) = 0.7698
                # Psi = sqrt (q ** 2 * (x**2) + y**2 = 0.25 + 1) = sqrt(1.25)

                # defl_x = normalization * atan(sqrt(1-q**2) x / Psi )
                # defl_x = 0.7698 * atan(sqrt(0.75)/sqrt(1.25) = 0.50734

                # defl_y = normalization * atanh(sqrt(1-q**2) y / (Psi) )

                assert defls[0,0] == pytest.approx(0.50734, 1e-3)
                assert defls[0,1] == pytest.approx(0.79420, 1e-3)

            def test_coordinate_rotation_90__defl_x_same_defl_y_flip(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0, 0), axis_ratio=0.5, phi=90.0,
                                                                einstein_radius=1.0)

                defls = isothermal.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                assert defls[0,0] == pytest.approx(0.79420, 1e-3)
                assert defls[0,1] == pytest.approx(0.50734, 1e-3)

            def test_coordinate_rotation_180__both_defl_flip(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0, 0), axis_ratio=0.5, phi=180.0,
                                                                einstein_radius=1.0)

                defls = isothermal.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                assert defls[0,0] == pytest.approx(0.50734, 1e-3)
                assert defls[0,1] == pytest.approx(0.79420, 1e-3)

            def test_coordinate_rotation_45__defl_y_zero_new_defl_x(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0, 0), axis_ratio=0.5, phi=45.0,
                                                                einstein_radius=1.0)

                defls = isothermal.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                # 45 degree aligns the mass profiles with the axes, so there is no deflection acoss y.

                assert defls[0,0] == pytest.approx(0.5698, 1e-3)
                assert defls[0,1] == pytest.approx(0.5700, 1e-3)

            def test_double_einr__double_defl_angles(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0, 0), axis_ratio=0.5, phi=45.0,
                                                                einstein_radius=2.0)

                defls = isothermal.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                assert defls[0,0] == pytest.approx(0.5698 * 2.0, 1e-3)
                assert defls[0,1] == pytest.approx(0.5700 * 2.0, 1e-3)

            def test_flip_coordinaates_and_centren__same_defl(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(-1, -1), axis_ratio=0.5, phi=0.0,
                                                                einstein_radius=1.0)

                defls = isothermal.deflections_from_grid(grid=np.array([[0.0, 0.0]]))

                assert defls[0,0] == pytest.approx(0.50734, 1e-3)
                assert defls[0,1] == pytest.approx(0.79420, 1e-3)

            def test_another_q__new_defl_values(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0, 0), axis_ratio=0.25, phi=0.0,
                                                                einstein_radius=2.0)

                defls = isothermal.deflections_from_grid(grid=np.array([[-1.0, -1.0]]))

                assert defls[0,0] == pytest.approx(-0.62308, 1e-3)
                assert defls[0,1] == pytest.approx(-1.43135, 1e-3)

            def test_compare_to_fortran__same_values(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                                einstein_radius=1.0)

                defls = isothermal.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.50734, 1e-3)
                assert defls[0,1] == pytest.approx(0.79421, 1e-3)

            def test_compare_to_fortran__same_values2(self):
                isothermal = mass_profiles.EllipticalIsothermal(centre=(0, 0), axis_ratio=0.5, phi=45.0,
                                                                einstein_radius=1.0)

                defls = isothermal.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.57002, 1e-3)
                assert defls[0,1] == pytest.approx(0.57002, 1e-3)

            def test__2_coordinates__gives_multiple_deflection_angles(self):

                isothermal = mass_profiles.EllipticalIsothermal(centre=(0, 0), axis_ratio=0.5, phi=45.0,
                                                                einstein_radius=1.0)

                defls = isothermal.deflections_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.57002, 1e-3)
                assert defls[0,1] == pytest.approx(0.57002, 1e-3)
                assert defls[1,0] == pytest.approx(0.57002, 1e-3)
                assert defls[1,1] == pytest.approx(0.57002, 1e-3)


    class TestEllipticalCoredPowerLaw(object):

        class TestSurfaceDensity(object):

            def test__function__gives_correct_values(self):
                power_law = mass_profiles.EllipticalCoredPowerLaw(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                                  einstein_radius=1.0, slope=2.2,
                                                                  core_radius=0.1)

                kappa = power_law.surface_density_func(radius=1.0)

                assert kappa == pytest.approx(0.39762, 1e-4)

            def test__function__same_as_power_law_no_core(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(1, 1), axis_ratio=1.0,
                                                                       phi=45.0,
                                                                       einstein_radius=1.0, slope=2.2,
                                                                       core_radius=0.)

                kappa_core = power_law_core.surface_density_func(radius=3.0)

                power_law = mass_profiles.EllipticalPowerLaw(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                             einstein_radius=1.0, slope=2.2)

                kappa = power_law.surface_density_func(radius=3.0)

                assert kappa == kappa_core

            def test__flip_coordinates_lens_center__same_value(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.2)

                surface_density_0 = power_law_core.surface_density_from_grid(grid=np.array([[1.0, 1.0]]))

                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(1.0, 1.0), axis_ratio=1.0,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.2)

                surface_density_1 = power_law_core.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_coordinates_90_circular__same_value(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.2)

                surface_density_0 = power_law_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                       phi=90.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.2)

                surface_density_1 = power_law_core.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.2,
                                                                       core_radius=0.2)

                surface_density_0 = power_law_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                       phi=90.0,
                                                                       einstein_radius=1.0, slope=2.2,
                                                                       core_radius=0.2)

                surface_density_1 = power_law_core.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__simple_case__correct_value(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.2)

                surface_density = power_law_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.49029, 1e-3)

            def test__double_einr__double_value(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                       phi=0.0,
                                                                       einstein_radius=2.0, slope=2.0,
                                                                       core_radius=0.2)

                surface_density = power_law_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx(2.0 * 0.49029, 1e-3)

            def test__different_axis_ratio__new_value(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.2)

                surface_density = power_law_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # for axis_ratio = 1.0, the factor is 1/2
                # for axis_ratio = 0.5, the factor is 1/(1.5)
                # So the change in the value is 0.5 / (1/1.5) = 1.0 / 0.75

                # axis ratio changes only einstein_rescaled, so wwe can use the above value and times by 1.0/1.5.

                assert surface_density == pytest.approx((1.0 / 0.75) * 0.49029, 1e-3)

            def test__slope_increase__new_value(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.3,
                                                                       core_radius=0.2)

                surface_density = power_law_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.45492, 1e-3)

            def test__slope_decrease__new_value(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5,
                                                                       phi=0.0,
                                                                       einstein_radius=2.0, slope=1.7,
                                                                       core_radius=0.2)

                surface_density = power_law_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(1.3887, 1e-3)

        class TestPotential(object):

            def test__flip_coordinates_lens_center__same_value(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.2)

                potential_0 = power_law_core.potential_from_grid(grid=np.array([[1.0, 1.0]]))

                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(1.0, 1.0), axis_ratio=1.0,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.2)

                potential_1 = power_law_core.potential_from_grid(grid=np.array([[0.0, 0.0]]))

                assert potential_0 == potential_1

            def test__rotation_coordinates_90_circular__same_value(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.2)

                potential_0 = power_law_core.potential_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                       phi=90.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.2)

                potential_1 = power_law_core.potential_from_grid(grid=np.array([[0.0, 1.0]]))

                assert potential_0 == potential_1

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.2,
                                                                       core_radius=0.2)

                potential_0 = power_law_core.potential_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                       phi=90.0,
                                                                       einstein_radius=1.0, slope=2.2,
                                                                       core_radius=0.2)

                potential_1 = power_law_core.potential_from_grid(grid=np.array([[0.0, 1.0]]))

                assert potential_0 == potential_1

            def test__same_as_sie_for_no_core(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(1, 1), axis_ratio=1.0,
                                                                       phi=45.0,
                                                                       einstein_radius=1.0, slope=2.2,
                                                                       core_radius=0.)

                potential_core = power_law_core.potential_from_grid(grid=np.array([[0.1, 0.1]]))

                power_law = mass_profiles.EllipticalPowerLaw(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                             einstein_radius=1.0, slope=2.2)

                potential = power_law.potential_from_grid(grid=np.array([[0.1, 0.1]]))

                assert potential_core == potential

            def test__value_via_fortran__same_value(self):
                power_law = mass_profiles.EllipticalCoredPowerLaw(centre=(0.5, -0.7), axis_ratio=0.7,
                                                                  phi=60.0,
                                                                  einstein_radius=1.3, slope=1.8,
                                                                  core_radius=0.2)

                potential = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential == pytest.approx(0.71185, 1e-3)

            def test__value_via_fortran_2__same_value(self):
                power_law = mass_profiles.EllipticalCoredPowerLaw(centre=(-0.2, 0.2), axis_ratio=0.6,
                                                                  phi=120.0,
                                                                  einstein_radius=0.5, slope=2.4,
                                                                  core_radius=0.5)

                potential = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential == pytest.approx(0.02319, 1e-3)

            def test__2_coordinates__2_potentials(self):
                power_law = mass_profiles.EllipticalCoredPowerLaw(centre=(-0.2, 0.2), axis_ratio=0.6,
                                                                  phi=120.0,
                                                                  einstein_radius=0.5, slope=2.4,
                                                                  core_radius=0.5)

                potential = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625],
                                                                         [0.1625, 0.1625]]))

                assert potential[0] == pytest.approx(0.02319, 1e-3)
                assert potential[1] == pytest.approx(0.02319, 1e-3)


        class TestDeflections(object):

            def test__flip_coordinates_lens_center__flips_deflection_angles(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.3)

                defls_0 = power_law_core.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(1.0, 1.0), axis_ratio=1.0,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.3)

                defls_1 = power_law_core.deflections_from_grid(grid=np.array([[0.0, 0.0]]))

                assert defls_0[0,0] == pytest.approx(-defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(-defls_1[0,1], 1e-5)

            def test__rotation_coordinates_90_circular__flips_x_and_y_deflection_angles(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.3)

                defls_0 = power_law_core.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                       phi=90.0,
                                                                       einstein_radius=1.0, slope=2.0,
                                                                       core_radius=0.3)

                defls_1 = power_law_core.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                # Foro deflection angles, a 90 degree rtation flips the x / y image_grid

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__rotation_90_ellpitical_cordinates_on_corners__flips_x_and_y_deflection_angles(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                       phi=0.0,
                                                                       einstein_radius=1.0, slope=2.2,
                                                                       core_radius=0.3)

                defls_0 = power_law_core.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                       phi=90.0,
                                                                       einstein_radius=1.0, slope=2.2,
                                                                       core_radius=0.3)

                defls_1 = power_law_core.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__compute_deflection__same_as_power_law_for_core_0(self):
                power_law_core = mass_profiles.EllipticalCoredPowerLaw(centre=(0.3, -0.1), axis_ratio=0.7,
                                                                       phi=60.0,
                                                                       einstein_radius=1.1, slope=2.1,
                                                                       core_radius=0.0)

                deflections_core = power_law_core.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                power_law = mass_profiles.EllipticalPowerLaw(centre=(0.3, -0.1), axis_ratio=0.7, phi=60.0,
                                                             einstein_radius=1.1, slope=2.1)

                deflections_power_law = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert deflections_core[0,0] == pytest.approx(deflections_power_law[0,0], 1e-6)
                assert deflections_core[0,1] == pytest.approx(deflections_power_law[0,1], 1e-6)

            def test__compute_deflection__value_via_fortran__same_value(self):
                power_law = mass_profiles.EllipticalCoredPowerLaw(centre=(0.5, -0.7), axis_ratio=0.7,
                                                                  phi=60.0,
                                                                  einstein_radius=1.3, slope=1.8,
                                                                  core_radius=0.2)

                deflections = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert deflections[0,0] == pytest.approx(-0.54883, 1e-3)
                assert deflections[0,1] == pytest.approx(0.98697, 1e-3)

            def test__compute_deflection__value_via_fortran_2__same_value(self):
                power_law = mass_profiles.EllipticalCoredPowerLaw(centre=(-0.2, 0.2), axis_ratio=0.6,
                                                                  phi=120.0,
                                                                  einstein_radius=0.5, slope=2.4,
                                                                  core_radius=0.5)

                deflections = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert deflections[0,0] == pytest.approx(0.11403, 1e-3)
                assert deflections[0,1] == pytest.approx(0.01111, 1e-3)

            def test__2_coordinates__2_sets_of_deflections(self):
                power_law = mass_profiles.EllipticalCoredPowerLaw(centre=(-0.2, 0.2), axis_ratio=0.6,
                                                                  phi=120.0,
                                                                  einstein_radius=0.5, slope=2.4,
                                                                  core_radius=0.5)

                deflections = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625],
                                                                             [0.1625, 0.1625]]))

                assert deflections[0,0] == pytest.approx(0.11403, 1e-3)
                assert deflections[0,1] == pytest.approx(0.01111, 1e-3)
                assert deflections[1,0] == pytest.approx(0.11403, 1e-3)
                assert deflections[1,1] == pytest.approx(0.01111, 1e-3)


    class TestSphericalCoredPowerLaw(object):

        class TestSurfaceDensity(object):
            def test__function__gives_correct_values(self):
                power_law = mass_profiles.SphericalCoredPowerLaw(centre=(1, 1), einstein_radius=1.0,
                                                                 slope=2.2,
                                                                 core_radius=0.1)

                kappa = power_law.surface_density_func(radius=1.0)

                assert kappa == pytest.approx(0.39761, 1e-4)

            def test__function__same_as_power_law_no_core(self):
                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(1, 1), einstein_radius=1.0,
                                                                      slope=2.2,
                                                                      core_radius=0.)

                kappa_core = power_law_core.surface_density_func(radius=3.0)

                power_law = mass_profiles.SphericalPowerLaw(centre=(1, 1), einstein_radius=1.0, slope=2.2)

                kappa = power_law.surface_density_func(radius=3.0)

                assert kappa == kappa_core

            def test__flip_coordinates_lens_center__same_value(self):
                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0,
                                                                      slope=2.0,
                                                                      core_radius=0.2)

                surface_density_0 = power_law_core.surface_density_from_grid(grid=np.array([[1.0, 1.0]]))

                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(1.0, 1.0), einstein_radius=1.0,
                                                                      slope=2.0,
                                                                      core_radius=0.2)

                surface_density_1 = power_law_core.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_coordinates_90_circular__same_value(self):
                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0,
                                                                      slope=2.0,
                                                                      core_radius=0.2)

                surface_density_0 = power_law_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0,
                                                                      slope=2.0,
                                                                      core_radius=0.2)

                surface_density_1 = power_law_core.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__simple_case__correct_value(self):
                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.0, 0.0),
                                                                      einstein_radius=1.0, slope=2.0,
                                                                      core_radius=0.2)

                surface_density = power_law_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.49029, 1e-3)

            def test__double_einr__double_value(self):
                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.0, 0.0),
                                                                      einstein_radius=2.0, slope=2.0,
                                                                      core_radius=0.2)

                surface_density = power_law_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(2.0 * 0.49029, 1e-3)

        class TestPotential(object):
            def test__flip_coordinates_lens_center__same_value(self):
                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0,
                                                                      slope=2.0,
                                                                      core_radius=0.2)

                potential_0 = power_law_core.potential_from_grid(grid=np.array([[1.0, 1.0]]))

                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(1.0, 1.0), einstein_radius=1.0,
                                                                      slope=2.0,
                                                                      core_radius=0.2)

                potential_1 = power_law_core.potential_from_grid(grid=np.array([[0.0, 0.0]]))

                assert potential_0 == potential_1

            def test__rotation_coordinates_90_circular__same_value(self):
                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0,
                                                                      slope=2.0,
                                                                      core_radius=0.2)

                potential_0 = power_law_core.potential_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0,
                                                                      slope=2.0,
                                                                      core_radius=0.2)

                potential_1 = power_law_core.potential_from_grid(grid=np.array([[0.0, 1.0]]))

                assert potential_0 == potential_1

            def test__same_as_power_law_for_no_core(self):
                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(1, 1),
                                                                      einstein_radius=1.0, slope=2.2,
                                                                      core_radius=0.)

                potential_core = power_law_core.potential_from_grid(grid=np.array([[0.1, 0.1]]))

                power_law = mass_profiles.SphericalPowerLaw(centre=(1, 1), einstein_radius=1.0, slope=2.2)

                potential = power_law.potential_from_grid(grid=np.array([[0.1, 0.1]]))

                assert potential_core == potential

            def test__same_as_power_law_for_no_core_2(self):
                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(3, 5),
                                                                      einstein_radius=15.0, slope=1.6,
                                                                      core_radius=0.)
                potential_core = power_law_core.potential_from_grid(grid=np.array([[0.1, 0.8]]))

                power_law = mass_profiles.SphericalPowerLaw(centre=(3, 5), einstein_radius=15.0, slope=1.6)

                potential = power_law.potential_from_grid(grid=np.array([[0.1, 0.8]]))

                assert potential_core == potential

            def test__value_via_fortran__same_value(self):
                power_law = mass_profiles.SphericalCoredPowerLaw(centre=(0.5, -0.7), einstein_radius=1.0,
                                                                 slope=1.8, core_radius=0.2)

                potential = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert potential == pytest.approx(0.54913, 1e-3)

            def test__value_via_fortran_2__same_value(self):
                power_law = mass_profiles.SphericalCoredPowerLaw(centre=(-0.2, 0.2), einstein_radius=0.5,
                                                                 slope=2.4, core_radius=0.5)

                potential = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert potential == pytest.approx(0.01820, 1e-3)

            def test__2_coordinates__2_potentials(self):

                power_law = mass_profiles.SphericalCoredPowerLaw(centre=(-0.2, 0.2), einstein_radius=0.5,
                                                                 slope=2.4, core_radius=0.5)

                potential = power_law.potential_from_grid(grid=np.array([[0.1625, 0.1875],
                                                                         [0.1625, 0.1875]]))

                assert potential[0] == pytest.approx(0.01820, 1e-3)
                assert potential[1] == pytest.approx(0.01820, 1e-3)

        class TestDeflections(object):

            def test__flip_coordinates_lens_center__flips_deflection_angles(self):

                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.0, 0.0),
                                                                      einstein_radius=1.0, slope=2.0,
                                                                      core_radius=0.3)

                defls_0 = power_law_core.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(1.0, 1.0),
                                                                      einstein_radius=1.0, slope=2.0,
                                                                      core_radius=0.3)

                defls_1 = power_law_core.deflections_from_grid(grid=np.array([[0.0, 0.0]]))

                assert defls_0[0,0] == pytest.approx(-defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(-defls_1[0,1], 1e-5)

            def test__rotation_coordinates_90_circular__flips_x_and_y_deflection_angles(self):
                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.0, 0.0),
                                                                      einstein_radius=1.0, slope=2.0,
                                                                      core_radius=0.3)

                defls_0 = power_law_core.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.0, 0.0),
                                                                      einstein_radius=1.0, slope=2.0,
                                                                      core_radius=0.3)

                defls_1 = power_law_core.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                # Foro deflection angles, a 90 degree rtation flips the x / y image_grid

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__compute_deflection__same_as_power_law_for_no_core(self):
                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.3, -0.1),
                                                                      einstein_radius=1.1, slope=2.2,
                                                                      core_radius=0.0)

                deflections_core = power_law_core.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                power_law = mass_profiles.SphericalPowerLaw(centre=(0.3, -0.1), einstein_radius=1.1,
                                                            slope=2.2)

                deflections_power_law = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert deflections_core[0,0] == pytest.approx(deflections_power_law[0,0], 1e-6)
                assert deflections_core[0,1] == pytest.approx(deflections_power_law[0,1], 1e-6)

            def test__compute_deflection__same_as_power_law_for_no_core_2(self):
                power_law_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.3, -0.1),
                                                                      einstein_radius=1.1, slope=1.6,
                                                                      core_radius=0.0)

                deflections_core = power_law_core.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                power_law = mass_profiles.SphericalPowerLaw(centre=(0.3, -0.1), einstein_radius=1.1,
                                                            slope=1.6)

                deflections_power_law = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert deflections_core[0,0] == pytest.approx(deflections_power_law[0,0], 1e-6)
                assert deflections_core[0,1] == pytest.approx(deflections_power_law[0,1], 1e-6)

                # TODO : Add Fortran values

            def test__compute_deflection__value_via_fortran__same_value(self):

                power_law = mass_profiles.SphericalCoredPowerLaw(centre=(0.5, -0.7), einstein_radius=1.0,
                                                                 slope=1.8, core_radius=0.2)

                deflections = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert deflections[0,0] == pytest.approx(-0.30680, 1e-3)
                assert deflections[0,1] == pytest.approx(0.80677, 1e-3)

            def test__compute_deflection__value_via_fortran_2__same_value(self):
                power_law = mass_profiles.SphericalCoredPowerLaw(centre=(-0.2, 0.2), einstein_radius=0.5,
                                                                 slope=2.4, core_radius=0.5)

                deflections = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert deflections[0,0] == pytest.approx(0.09316, 1e-3)
                assert deflections[0,1] == pytest.approx(-0.00321, 1e-3)

            def test__2_coordinates__2_sets_of_deflections(self):
                power_law = mass_profiles.SphericalCoredPowerLaw(centre=(-0.2, 0.2), einstein_radius=0.5,
                                                                 slope=2.4, core_radius=0.5)

                deflections = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1875],
                                                                             [0.1625, 0.1875]]))

                assert deflections[0,0] == pytest.approx(0.09316, 1e-3)
                assert deflections[0,1] == pytest.approx(-0.00321, 1e-3)
                assert deflections[1,0] == pytest.approx(0.09316, 1e-3)
                assert deflections[1,1] == pytest.approx(-0.00321, 1e-3)


    class TestEllipticalCoredIsothermal(object):

        class TestSurfaceDensity(object):
            def test__function__gives_correct_values(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(1, 1), axis_ratio=1.0,
                                                                          phi=45.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.1)

                kappa = isothermal_core.surface_density_func(radius=1.0)

                assert kappa == pytest.approx(0.49752, 1e-4)

            def test__function__same_as_isothermal_core_no_core(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(1, 1), axis_ratio=1.0,
                                                                          phi=45.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.0)

                kappa_core = isothermal_core.surface_density_func(radius=3.0)

                isothermal = mass_profiles.EllipticalIsothermal(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                                einstein_radius=1.0)

                kappa = isothermal.surface_density_func(radius=3.0)

                assert kappa == kappa_core

            def test__flip_coordinates_lens_center__same_value(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.2)

                surface_density_0 = isothermal_core.surface_density_from_grid(grid=np.array([[1.0, 1.0]]))

                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(1.0, 1.0), axis_ratio=1.0,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.2)

                surface_density_1 = isothermal_core.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_coordinates_90_circular__same_value(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.2)

                surface_density_0 = isothermal_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                          phi=90.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.2)

                surface_density_1 = isothermal_core.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.2)

                surface_density_0 = isothermal_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                          phi=90.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.2)

                surface_density_1 = isothermal_core.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__simple_case__correct_value(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.2)

                surface_density = isothermal_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.49029, 1e-3)

            def test__double_einr__double_value(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                          phi=0.0,
                                                                          einstein_radius=2.0,
                                                                          core_radius=0.2)

                surface_density = isothermal_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(2.0 * 0.49029, 1e-3)

            def test__different_axis_ratio__new_value(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=0.5,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.2)

                surface_density = isothermal_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # axis ratio changes only einstein_rescaled, so wwe can use the above value and times by 1.0/1.5.

                assert surface_density == pytest.approx(0.49029 * 1.33333, 1e-3)

        class TestPotential(object):
            def test__flip_coordinates_lens_center__same_value(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.2)

                potential_0 = isothermal_core.potential_from_grid(grid=np.array([[1.0, 1.0]]))

                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(1.0, 1.0), axis_ratio=1.0,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.2)

                potential_1 = isothermal_core.potential_from_grid(grid=np.array([[0.0, 0.0]]))

                assert potential_0 == potential_1

            def test__rotation_coordinates_90_circular__same_value(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.2)

                potential_0 = isothermal_core.potential_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                          phi=90.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.2)

                potential_1 = isothermal_core.potential_from_grid(grid=np.array([[0.0, 1.0]]))

                assert potential_0 == potential_1

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.2)

                potential_0 = isothermal_core.potential_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                     phi=90.0,
                                                                     einstein_radius=1.0, core_radius=0.2)

                potential_1 = isothermal.potential_from_grid(grid=np.array([[0.0, 1.0]]))

                assert potential_0 == potential_1

            def test__same_as_sie_for_no_core(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(1, 1), axis_ratio=0.9,
                                                                          phi=45.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.)

                potential_core = isothermal_core.potential_from_grid(grid=np.array([[0.1, 0.1]]))

                isothermal = mass_profiles.EllipticalIsothermal(centre=(1, 1), axis_ratio=0.9, phi=45.0,
                                                                einstein_radius=1.0)

                potential = isothermal.potential_from_grid(grid=np.array([[0.1, 0.1]]))

                assert potential_core == potential

            def test__compute_potential__ratio_via_fortran__same_ratio(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.5, -0.7), axis_ratio=0.7,
                                                                          phi=60.0,
                                                                          einstein_radius=1.3,
                                                                          core_radius=0.2)

                potential_0 = isothermal_core.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(-0.2, 0.2), axis_ratio=0.6,
                                                                          phi=120.0,
                                                                          einstein_radius=0.5,
                                                                          core_radius=0.5)

                potential_1 = isothermal_core.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                ratio = potential_0 / potential_1

                assert ratio == pytest.approx(18.47647, 1e-3)

            def test__compute_potential__value_via_fortran__same_value(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.5, -0.7), axis_ratio=0.7,
                                                                          phi=60.0,
                                                                          einstein_radius=1.3,
                                                                          core_radius=0.2)

                potential = isothermal_core.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential == pytest.approx(0.74354, 1e-3)

            def test__compute_potential__value_via_fortran_2__same_value(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(-0.2, 0.2), axis_ratio=0.6,
                                                                          phi=120.0,
                                                                          einstein_radius=0.5,
                                                                          core_radius=0.5)

                potential = isothermal_core.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential == pytest.approx(0.04024, 1e-3)

            def test__2_coordinates__2_potentials(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(-0.2, 0.2), axis_ratio=0.6,
                                                                          phi=120.0,
                                                                          einstein_radius=0.5,
                                                                          core_radius=0.5)

                potential = isothermal_core.potential_from_grid(grid=np.array([[0.1625, 0.1625],
                                                                               [0.1625, 0.1625]]))

                assert potential[0] == pytest.approx(0.04024, 1e-3)
                assert potential[1] == pytest.approx(0.04024, 1e-3)


        class TestDeflections(object):
            def test__flip_coordinates_lens_center__flips_deflection_angles(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.3)

                defls_0 = isothermal_core.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(1.0, 1.0), axis_ratio=1.0,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.3)

                defls_1 = isothermal_core.deflections_from_grid(grid=np.array([[0.0, 0.0]]))

                assert defls_0[0,0] == pytest.approx(-defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(-defls_1[0,1], 1e-5)

            def test__rotation_coordinates_90_circular__flips_x_and_y_deflection_angles(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.3)

                defls_0 = isothermal_core.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                          phi=90.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.3)

                defls_1 = isothermal_core.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                # Foro deflection angles, a 90 degree rtation flips the x / y image_grid

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__rotation_90_ellpitical_cordinates_on_corners__flips_x_and_y_deflection_angles(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                          phi=0.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.3)

                defls_0 = isothermal_core.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                          phi=90.0,
                                                                          einstein_radius=1.0,
                                                                          core_radius=0.3)

                defls_1 = isothermal_core.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__same_as_isothermal_core_for_core_0(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.3, -0.1), axis_ratio=0.7,
                                                                          phi=60.0,
                                                                          einstein_radius=1.1,
                                                                          core_radius=0.0)

                deflections_core = isothermal_core.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                isothermal_core = mass_profiles.EllipticalIsothermal(centre=(0.3, -0.1), axis_ratio=0.7,
                                                                     phi=60.0,
                                                                     einstein_radius=1.1)

                deflections_isothermal_core = isothermal_core.deflections_from_grid(
                    grid=np.array([[0.1625, 0.1625]]))

                assert deflections_core[0,0] == pytest.approx(deflections_isothermal_core[0,0], 1e-6)
                assert deflections_core[0,1] == pytest.approx(deflections_isothermal_core[0,1], 1e-6)

            def test__ratio_via_fortran__same_ratio(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.5, -0.7), axis_ratio=0.7,
                                                                          phi=60.0,
                                                                          einstein_radius=1.3,
                                                                          core_radius=0.2)

                deflections = isothermal_core.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                ratio = deflections[0,0] / deflections[0,1]

                assert ratio == pytest.approx(-0.53649 / 0.98365, 1e-3)

            def test__value_via_fortran__same_value(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(0.5, -0.7), axis_ratio=0.7,
                                                                          phi=60.0,
                                                                          einstein_radius=1.3,
                                                                          core_radius=0.2)

                deflections = isothermal_core.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert deflections[0,0] == pytest.approx(-0.52047, 1e-3)
                assert deflections[0,1] == pytest.approx(0.95429, 1e-3)

            def test__value_via_fortran_2__same_value(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(-0.2, 0.2), axis_ratio=0.6,
                                                                          phi=120.0,
                                                                          einstein_radius=0.5,
                                                                          core_radius=0.5)

                deflections = isothermal_core.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert deflections[0,0] == pytest.approx(0.20500, 1e-3)
                assert deflections[0,1] == pytest.approx(0.02097, 1e-3)

            def test__2_coordinates__2_sets_of_deflections(self):
                isothermal_core = mass_profiles.EllipticalCoredIsothermal(centre=(-0.2, 0.2), axis_ratio=0.6,
                                                                          phi=120.0,
                                                                          einstein_radius=0.5,
                                                                          core_radius=0.5)

                deflections = isothermal_core.deflections_from_grid(grid=np.array([[0.1625, 0.1625],
                                                                                   [0.1625, 0.1625]]))

                assert deflections[0,0] == pytest.approx(0.20500, 1e-3)
                assert deflections[0,1] == pytest.approx(0.02097, 1e-3)
                assert deflections[1,0] == pytest.approx(0.20500, 1e-3)
                assert deflections[1,1] == pytest.approx(0.02097, 1e-3)


    class TestSphericalCoredIsothermal(object):

        class TestSurfaceDensity(object):
            def test__function__gives_correct_values(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(1, 1),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.1)

                kappa = isothermal_core.surface_density_func(radius=1.0)

                assert kappa == pytest.approx(0.49751, 1e-4)

            def test__function__same_as_isothermal_core_no_core(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(1, 1),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.0)

                kappa_core = isothermal_core.surface_density_func(radius=3.0)

                isothermal = mass_profiles.SphericalIsothermal(centre=(1, 1), einstein_radius=1.0)

                kappa = isothermal.surface_density_func(radius=3.0)

                assert kappa == kappa_core

            def test__flip_coordinates_lens_center__same_value(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.0, 0.0),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.2)

                surface_density_0 = isothermal_core.surface_density_from_grid(grid=np.array([[1.0, 1.0]]))

                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(1.0, 1.0),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.2)

                surface_density_1 = isothermal_core.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_coordinates_90_circular__same_value(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.0, 0.0),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.2)

                surface_density_0 = isothermal_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.0, 0.0),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.2)

                surface_density_1 = isothermal_core.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__simple_case__correct_value(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.0, 0.0),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.2)

                surface_density = isothermal_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(0.49029, 1e-3)

            def test__double_einr__double_value(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.0, 0.0),
                                                                         einstein_radius=2.0,
                                                                         core_radius=0.2)

                surface_density = isothermal_core.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                # eta = 1.0
                # kappa = 0.5 * 1.0 ** 1.0

                assert surface_density == pytest.approx(2.0 * 0.49029, 1e-3)

        class TestPotential(object):

            def test__flip_coordinates_lens_center__same_value(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.0, 0.0),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.2)

                potential_0 = isothermal_core.potential_from_grid(grid=np.array([[1.0, 1.0]]))

                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(1.0, 1.0),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.2)

                potential_1 = isothermal_core.potential_from_grid(grid=np.array([[0.0, 0.0]]))

                assert potential_0 == potential_1

            def test__rotation_coordinates_90_circular__same_value(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.0, 0.0),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.2)

                potential_0 = isothermal_core.potential_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.0, 0.0),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.2)

                potential_1 = isothermal_core.potential_from_grid(grid=np.array([[0.0, 1.0]]))

                assert potential_0 == potential_1

            # def test__same_as_sie_for_no_core(self):
            #
            #     isothermal_core = profiles.SphericalCoredIsothermal(centre=(1, 1), axis_ratio=0.9, phi=45.0,
            #                                                       einstein_radius=1.0, core_radius=0.)
            #
            #     potential_core = isothermal_core.compute_potential(image_grid=np.array([[0.1, 0.1]]))
            #
            #     isothermal = profiles.SphericalIsothermal(centre=(1, 1), axis_ratio=0.9, phi=45.0,
            #                                                       einstein_radius=1.0)
            #
            #     potential = isothermal.compute_potential(image_grid=np.array([[0.1, 0.1]]))
            #
            #     assert potential_core == potential

            def test__same_as_isothermal_core_for_spherical(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.3, -0.1),
                                                                         einstein_radius=1.1,
                                                                         core_radius=0.5)

                potentials_core = isothermal_core.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                isothermal_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.3, -0.1),
                                                                       einstein_radius=1.1,
                                                                       slope=2.0, core_radius=0.5)

                potentials_isothermal_core = isothermal_core.potential_from_grid(
                    grid=np.array([[0.1625, 0.1625]]))

                assert potentials_core == pytest.approx(potentials_isothermal_core, 1e-6)

            def test__same_as_isothermal_core_for_spherical_2(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(-0.3, 0.7),
                                                                         einstein_radius=10.1,
                                                                         core_radius=1.5)

                potentials_core = isothermal_core.potential_from_grid(grid=np.array([[-0.1625, -1.1625]]))

                isothermal_core = mass_profiles.SphericalCoredPowerLaw(centre=(-0.3, 0.7),
                                                                       einstein_radius=10.1,
                                                                       slope=2.0, core_radius=1.5)

                potentials_isothermal_core = isothermal_core.potential_from_grid(
                    grid=np.array([[-0.1625, -1.1625]]))

                assert potentials_core == pytest.approx(potentials_isothermal_core, 1e-6)

            def test__compute_potential__value_via_fortran__same_value(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.5, -0.7),
                                                                         einstein_radius=1.3,
                                                                         core_radius=0.2)

                potential = isothermal_core.potential_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert potential == pytest.approx(0.72231, 1e-3)

            def test__compute_potential__value_via_fortran_2__same_value(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(-0.2, 0.2),
                                                                         einstein_radius=0.5,
                                                                         core_radius=0.5)

                potential = isothermal_core.potential_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert potential == pytest.approx(0.03103, 1e-3)

            def test__2_coordinates__2_potentials(self):

                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(-0.2, 0.2),
                                                                         einstein_radius=0.5,
                                                                         core_radius=0.5)

                potential = isothermal_core.potential_from_grid(grid=np.array([[0.1625, 0.1875],
                                                                               [0.1625, 0.1875]]))

                assert potential[0] == pytest.approx(0.03103, 1e-3)
                assert potential[1] == pytest.approx(0.03103, 1e-3)

        class TestDeflections(object):
            def test__flip_coordinates_lens_center__flips_deflection_angles(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.0, 0.0),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.3)

                defls_0 = isothermal_core.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(1.0, 1.0),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.3)

                defls_1 = isothermal_core.deflections_from_grid(grid=np.array([[0.0, 0.0]]))

                assert defls_0[0,0] == pytest.approx(-defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(-defls_1[0,1], 1e-5)

            def test__rotation_coordinates_90_circular__flips_x_and_y_deflection_angles(self):

                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.0, 0.0),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.3)

                defls_0 = isothermal_core.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.0, 0.0),
                                                                         einstein_radius=1.0,
                                                                         core_radius=0.3)

                defls_1 = isothermal_core.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                # Foro deflection angles, a 90 degree rtation flips the x / y image_grid

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__same_as_isothermal_core_for_spherical(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.3, -0.1),
                                                                         einstein_radius=1.1,
                                                                         core_radius=0.5)

                deflections_core = isothermal_core.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                isothermal_core = mass_profiles.SphericalCoredPowerLaw(centre=(0.3, -0.1),
                                                                       einstein_radius=1.1,
                                                                       slope=2.0, core_radius=0.5)

                deflections_isothermal_core = isothermal_core.deflections_from_grid(
                    grid=np.array([[0.1625, 0.1625]]))

                assert deflections_core[0,0] == pytest.approx(deflections_isothermal_core[0,0], 1e-6)
                assert deflections_core[0,1] == pytest.approx(deflections_isothermal_core[0,1], 1e-6)

            def test__same_as_isothermal_core_for_spherical_2(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(-0.3, 0.7),
                                                                         einstein_radius=10.1,
                                                                         core_radius=1.5)

                deflections_core = isothermal_core.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                isothermal_core = mass_profiles.SphericalCoredPowerLaw(centre=(-0.3, 0.7),
                                                                       einstein_radius=10.1,
                                                                       slope=2.0, core_radius=1.5)

                deflections_isothermal_core = isothermal_core.deflections_from_grid(
                    grid=np.array([[0.1625, 0.1625]]))

                assert deflections_core[0,0] == pytest.approx(deflections_isothermal_core[0,0], 1e-6)
                assert deflections_core[0,1] == pytest.approx(deflections_isothermal_core[0,1], 1e-6)

                # TODO : Add Fortran

            def test__compute_deflection_angles_via_fortran__same_value(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(0.5, -0.7),
                                                                         einstein_radius=1.3,
                                                                         core_radius=0.2)

                deflections = isothermal_core.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert deflections[0,0] == pytest.approx(-0.37489, 1e-3)
                assert deflections[0,1] == pytest.approx(0.98582, 1e-3)

            def test__compute_deflection_angles_via_fortran_2__same_value(self):
                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(-0.2, 0.2),
                                                                         einstein_radius=0.5,
                                                                         core_radius=0.5)

                deflections = isothermal_core.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert deflections[0,0] == pytest.approx(0.16216, 1e-3)
                assert deflections[0,1] == pytest.approx(-0.00559, 1e-3)

            def test__2_coordinates__2_sets_of_deflections(self):

                isothermal_core = mass_profiles.SphericalCoredIsothermal(centre=(-0.2, 0.2), einstein_radius=0.5,
                                                                         core_radius=0.5)

                deflections = isothermal_core.deflections_from_grid(grid=np.array([[0.1625, 0.1875],
                                                                                   [0.1625, 0.1875]]))

                assert deflections[0,0] == pytest.approx(0.16216, 1e-3)
                assert deflections[0,1] == pytest.approx(-0.00559, 1e-3)
                assert deflections[1,0] == pytest.approx(0.16216, 1e-3)
                assert deflections[1,1] == pytest.approx(-0.00559, 1e-3)


    class TestEllipticalNFW(object):

        class TestCoordFuncc(object):
            def test__coord_func_x_above_1(self):
                assert mass_profiles.EllipticalNFW.coord_func(2.0) == pytest.approx(0.60459, 1e-3)

            def test__coord_func_x_below_1(self):
                assert mass_profiles.EllipticalNFW.coord_func(0.5) == pytest.approx(1.5206919, 1e-3)

            def test__coord_1(self):
                assert mass_profiles.EllipticalNFW.coord_func(1.0) == 1.0

        class TestSurfaceDensity(object):

            def test__flip_coordinates_lens_center__same_value(self):
                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=10.0)

                surface_density_0 = nfw.surface_density_from_grid(grid=np.array([[1.0, 1.0]]))

                nfw = mass_profiles.EllipticalNFW(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=10.0)

                surface_density_1 = nfw.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_coordinates_90_circular__same_value(self):
                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=10.0)

                surface_density_0 = nfw.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                  kappa_s=1.0, scale_radius=10.0)

                surface_density_1 = nfw.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=10.0)

                surface_density_0 = nfw.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                  kappa_s=1.0, scale_radius=10.0)

                surface_density_1 = nfw.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__simple_case__correct_value(self):
                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                # r = 2.0 (> 1.0)
                # F(r) = (1/(sqrt(3))*atan(sqrt(3)) = 0.60459978807
                # kappa(r) = 2 * kappa_s * (1 - 0.60459978807) / (4-1) = 0.263600141

                surface_density = nfw.surface_density_from_grid(grid=np.array([[2.0, 0.0]]))

                assert surface_density == pytest.approx(0.263600141, 1e-3)

            def test__simple_case_2__correct_value(self):
                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                surface_density = nfw.surface_density_from_grid(grid=np.array([[0.5, 0.0]]))

                assert surface_density == pytest.approx(1.388511, 1e-3)

            def test__double_kappa__doubles_value(self):
                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=2.0, scale_radius=1.0)

                surface_density = nfw.surface_density_from_grid(grid=np.array([[0.5, 0.0]]))

                assert surface_density == pytest.approx(2.0 * 1.388511, 1e-3)

            def test__double_scale_radius_and_coordinate__same_value(self):
                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=2.0)

                surface_density = nfw.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx(1.388511, 1e-3)

            def test__different_axis_ratio_and_coordinate_change__new_value(self):
                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                surface_density = nfw.surface_density_from_grid(grid=np.array([[0.0, 0.25]]))

                assert surface_density == pytest.approx(1.388511, 1e-3)

            def test__different_rotate_phi_90_same_result(self):
                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                surface_density_0 = nfw.surface_density_from_grid(grid=np.array([[0.0, 2.0]]))

                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=0.5, phi=90.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                surface_density_1 = nfw.surface_density_from_grid(grid=np.array([[2.0, 0.0]]))

                assert surface_density_0 == surface_density_1

        class TestPotential(object):

            def test__flip_coordinates_lens_center__same_value(self):

                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                potential_0 = nfw.potential_from_grid(grid=np.array([[1.00001, 1.00001]]))

                nfw = mass_profiles.EllipticalNFW(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                potential_1 = nfw.potential_from_grid(grid=np.array([[0.00001, 0.00001]]))

                assert potential_0 == pytest.approx(potential_1, 1e-4)

            def test__rotation_coordinates_90_circular__same_value(self):
                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                potential_0 = nfw.potential_from_grid(grid=np.array([[1.1, 0.0]]))

                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                potential_1 = nfw.potential_from_grid(grid=np.array([[0.0, 1.1]]))

                # Foro deflection angles, a 90 degree rtation flips the x / y image_grid

                assert potential_0 == pytest.approx(potential_1, 1e-5)

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                potential_0 = nfw.potential_from_grid(grid=np.array([[1.1, 0.0]]))

                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                potential_1 = nfw.potential_from_grid(grid=np.array([[0.0, 1.1]]))

                assert potential_0 == pytest.approx(potential_1, 1e-5)

            def test__compare_to_fortran__same_potential(self):
                nfw = mass_profiles.EllipticalNFW(centre=(0.2, 0.3), axis_ratio=0.7, phi=6.0,
                                                  kappa_s=2.5, scale_radius=4.0)

                potential = nfw.potential_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert potential == pytest.approx(0.05380, 1e-3)

            def test__2_coordinates__2_potentials(self):

                nfw = mass_profiles.EllipticalNFW(centre=(0.2, 0.3), axis_ratio=0.7, phi=6.0,
                                                  kappa_s=2.5, scale_radius=4.0)

                potential = nfw.potential_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))

                assert potential[0] == pytest.approx(0.05380, 1e-3)
                assert potential[1] == pytest.approx(0.05380, 1e-3)

        class TestDeflections(object):

            def test__flip_coordinates_lens_center__same_value(self):

                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                defls_0 = nfw.deflections_from_grid(grid=np.array([[1.00001, 1.00001]]))

                nfw = mass_profiles.EllipticalNFW(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                defls_1 = nfw.deflections_from_grid(grid=np.array([[0.00001, 0.00001]]))

                assert defls_0[0,0] == pytest.approx(-defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(-defls_1[0,1], 1e-5)

            def test__rotation_coordinates_90_circular__same_value(self):

                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                defls_0 = nfw.deflections_from_grid(grid=np.array([[1.1, 0.0]]))

                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                defls_1 = nfw.deflections_from_grid(grid=np.array([[0.0, 1.1]]))

                # Foro deflection angles, a 90 degree rtation flips the x / y image_grid

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):

                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                defls_0 = nfw.deflections_from_grid(grid=np.array([[1.1, 0.0]]))

                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                defls_1 = nfw.deflections_from_grid(grid=np.array([[0.0, 1.1]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__compare_to_fortran_1(self):

                nfw = mass_profiles.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                  kappa_s=1.0, scale_radius=1.0)

                defls = nfw.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.56194, 1e-3)
                assert defls[0,1] == pytest.approx(0.56194, 1e-3)

            def test__compare_to_fortran_2(self):

                nfw = mass_profiles.EllipticalNFW(centre=(0.2, 0.3), axis_ratio=0.7, phi=6.0,
                                                  kappa_s=2.5, scale_radius=4.0)
                defls = nfw.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(-0.44204, 1e-3)
                assert defls[0,1] == pytest.approx(-2.59480, 1e-3)

            def test__2_coordinates__2_sets_of_deflections(self):

                nfw = mass_profiles.EllipticalNFW(centre=(0.2, 0.3), axis_ratio=0.7, phi=6.0,
                                                  kappa_s=2.5, scale_radius=4.0)
                defls = nfw.deflections_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))

                assert defls[0, 0] == pytest.approx(-0.44204, 1e-3)
                assert defls[0, 1] == pytest.approx(-2.59480, 1e-3)
                assert defls[1, 0] == pytest.approx(-0.44204, 1e-3)
                assert defls[1, 1] == pytest.approx(-2.59480, 1e-3)


    class TestSphericalNFW(object):

        class TestSurfaceDensity(object):

            def test__simple_case__correct_value(self):

                nfw = mass_profiles.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
                surface_density = nfw.surface_density_from_grid(grid=np.array([[2.0, 0.0]]))

                assert surface_density == pytest.approx(0.263600141, 1e-3)

        class TestPotential(object):

            def test__flip_coordinates_lens_center__same_value(self):

                nfw = mass_profiles.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=10.0)

                potential_0 = nfw.potential_from_grid(grid=np.array([[1.0, 1.0]]))

                nfw = mass_profiles.SphericalNFW(centre=(1.0, 1.0), kappa_s=1.0, scale_radius=10.0)

                potential_1 = nfw.potential_from_grid(grid=np.array([[2.0, 2.0]]))

                assert potential_0 == pytest.approx(potential_1, 1e-4)

            def test__compare_to_fortran__same_potential(self):

                nfw = mass_profiles.SphericalNFW(centre=(0.2, 0.3), kappa_s=2.5, scale_radius=4.0)
                potential = nfw.potential_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert potential == pytest.approx(0.03702, 1e-3)

            def test__2_coordinates__2_potentials(self):

                nfw = mass_profiles.SphericalNFW(centre=(0.2, 0.3), kappa_s=2.5, scale_radius=4.0)
                potential = nfw.potential_from_grid(grid=np.array([[0.1625, 0.1875], [0.1625, 0.1875]]))

                assert potential[0] == pytest.approx(0.03702, 1e-3)
                assert potential[1] == pytest.approx(0.03702, 1e-3)

        class TestDeflections(object):

            def test__flip_coordinates_lens_center__same_value(self):

                nfw = mass_profiles.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

                defls_0 = nfw.deflections_from_grid(grid=np.array([[1.00001, 1.00001]]))

                nfw = mass_profiles.SphericalNFW(centre=(1.0, 1.0), kappa_s=1.0, scale_radius=1.0)

                defls_1 = nfw.deflections_from_grid(grid=np.array([[0.00001, 0.00001]]))

                assert defls_0[0,0] == pytest.approx(-defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(-defls_1[0,1], 1e-5)

            def test__compare_to_fortran_1(self):

                nfw = mass_profiles.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

                defls = nfw.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.56194, 1e-3)
                assert defls[0,1] == pytest.approx(0.56194, 1e-3)

            def test__compare_to_fortran_2(self):

                nfw = mass_profiles.SphericalNFW(centre=(0.2, 0.3), kappa_s=2.5, scale_radius=4.0)
                defls = nfw.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert defls[0,0] == pytest.approx(-0.69636, 1e-3)
                assert defls[0,1] == pytest.approx(-2.08909, 1e-3)

            def test__compare_to_elliptical__same_value(self):

                nfw = mass_profiles.SphericalNFW(centre=(1.0, 1.0), kappa_s=10.0, scale_radius=0.1)

                defls_0 = nfw.deflections_from_grid(grid=np.array([[3.0, 3.0]]))

                nfw = mass_profiles.EllipticalNFW(centre=(1.0, 1.0), axis_ratio=1.0, phi=45.0,
                                                  kappa_s=10.0, scale_radius=0.1)

                defls_1 = nfw.deflections_from_grid(grid=np.array([[3.0, 3.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,1], 1e-5)

            def test__compare_to_elliptical_2__same_value(self):

                nfw = mass_profiles.SphericalNFW(centre=(1.5, 1.5), kappa_s=7.0, scale_radius=0.15)

                defls_0 = nfw.deflections_from_grid(grid=np.array([[-3.2, 1.0]]))

                nfw = mass_profiles.EllipticalNFW(centre=(1.5, 1.5), axis_ratio=1.0, phi=60.0,
                                                  kappa_s=7.0, scale_radius=0.15)

                defls_1 = nfw.deflections_from_grid(grid=np.array([[-3.2, 1.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,1], 1e-5)

            def test__2_coordinates__2_sets_of_delfections(self):

                nfw = mass_profiles.SphericalNFW(centre=(0.2, 0.3), kappa_s=2.5, scale_radius=4.0)
                defls = nfw.deflections_from_grid(grid=np.array([[0.1625, 0.1875], [0.1625, 0.1875]]))

                assert defls[0,0] == pytest.approx(-0.69636, 1e-3)
                assert defls[0,1] == pytest.approx(-2.08909, 1e-3)
                assert defls[1,0] == pytest.approx(-0.69636, 1e-3)
                assert defls[1,1] == pytest.approx(-2.08909, 1e-3)


    class TestEllipticalGeneralizedNFW(object):

        class TestSurfaceDensity(object):

            def test__simple_case__correct_value(self):

                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, axis_ratio=0.5,
                                                              phi=90.0, inner_slope=1.5, scale_radius=1.0)
                surface_density = gnfw.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx(0.30840, 1e-3)

            def test__double_kappa__double_value(self):
                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=2.0, axis_ratio=0.5,
                                                              phi=90.0, inner_slope=1.5, scale_radius=1.0)
                surface_density = gnfw.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx(0.30840 * 2, 1e-3)

            def test__compare_to_spherical_nfw__same_value(self):
                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(0.8, 0.2), kappa_s=1.0, axis_ratio=1.0,
                                                              phi=00.0, inner_slope=1.0, scale_radius=2.0)
                surface_density_0 = gnfw.surface_density_from_grid(grid=np.array([[2.0, 0.0]]))

                nfw = mass_profiles.SphericalNFW(centre=(0.8, 0.2), kappa_s=1.0, scale_radius=2.0)
                surface_density_1 = nfw.surface_density_from_grid(grid=np.array([[2.0, 0.0]]))

                assert surface_density_0 == pytest.approx(surface_density_1, 1e-3)

            def test__compare_to_elliptical_nfw__same_value(self):
                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(0.8, 0.2), kappa_s=1.0, axis_ratio=0.5,
                                                              phi=100.0, inner_slope=1.0, scale_radius=2.0)

                surface_density_0 = gnfw.surface_density_from_grid(grid=np.array([[12.0, 10.0]]))

                nfw = mass_profiles.EllipticalNFW(centre=(0.8, 0.2), kappa_s=1.0, axis_ratio=0.5,
                                                  phi=100.0, scale_radius=2.0)
                surface_density_1 = nfw.surface_density_from_grid(grid=np.array([[12.0, 10.0]]))

                assert surface_density_0 == pytest.approx(surface_density_1, 1e-3)

            def test__compare_to_spherical_gnfw__same_value(self):
                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(0.8, 0.2), kappa_s=1.0, axis_ratio=1.0,
                                                              phi=100.0, inner_slope=1.5, scale_radius=2.0)
                surface_density_0 = gnfw.surface_density_from_grid(grid=np.array([[12.0, 10.0]]))

                nfw = mass_profiles.SphericalGeneralizedNFW(centre=(0.8, 0.2), kappa_s=1.0, inner_slope=1.5,
                                                            scale_radius=2.0)
                surface_density_1 = nfw.surface_density_from_grid(grid=np.array([[12.0, 10.0]]))

                assert surface_density_0 == pytest.approx(surface_density_1, 1e-3)


        class TestPotential(object):

            def test__flip_coordinates_lens_center__same_value(self):
                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, axis_ratio=0.5,
                                                              phi=100.0, inner_slope=1.5, scale_radius=10.0)

                potential_0 = gnfw.potential_from_grid(grid=np.array([[1.0, 1.0]]))

                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=1.0, axis_ratio=0.5,
                                                              phi=100.0, inner_slope=1.5, scale_radius=10.0)

                potential_1 = gnfw.potential_from_grid(grid=np.array([[2.0, 2.0]]))

                assert potential_0 == pytest.approx(potential_1, 1e-4)

            def test__compare_to_elliptical_nfw__same_value(self):

                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=5.0, axis_ratio=0.5,
                                                              phi=100.0, inner_slope=1.0, scale_radius=10.0)

                potential_0 = gnfw.potential_from_grid(grid=np.array([[2.0, 2.0]]))

                nfw = mass_profiles.EllipticalNFW(centre=(1.0, 1.0), kappa_s=5.0, axis_ratio=0.5,
                                                  phi=100.0, scale_radius=10.0)

                potential_1 = nfw.potential_from_grid(grid=np.array([[2.0, 2.0]]))

                assert potential_0[0] == pytest.approx(potential_1[0], 1e-4)

            def test__compare_to_known_value(self):

                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=5.0, axis_ratio=0.5,
                                                              phi=100.0, inner_slope=1.0, scale_radius=10.0)

                potential_0 = gnfw.potential_from_grid(grid=np.array([[2.0, 2.0]]))

                assert potential_0[0] == pytest.approx(2.4718, 1e-4)


        class TestDeflections(object):

            def test__flip_coordinates_lens_center__same_value(self):

                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, axis_ratio=0.5,
                                                              phi=100.0, inner_slope=1.5, scale_radius=1.0)

                defls_0 = gnfw.deflections_from_grid(grid=np.array([[1.00001, 1.00001]]))

                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=1.0, axis_ratio=0.5,
                                                              phi=100.0, inner_slope=1.5, scale_radius=1.0)

                defls_1 = gnfw.deflections_from_grid(grid=np.array([[0.00001, 0.00001]]))

                assert defls_0[0,0] == pytest.approx(-defls_1[0,0], 1e-4)
                assert defls_0[0,1] == pytest.approx(-defls_1[0,1], 1e-4)

            def test__compare_to_fortran_1(self):

                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, axis_ratio=0.3,
                                                              phi=100.0, inner_slope=0.5, scale_radius=8.0)

                defls = gnfw.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert defls[0,0] == pytest.approx(0.58988, 1e-3)
                assert defls[0,1] == pytest.approx(0.26604, 1e-3)

            def test__compare_to_fortran_2(self):
                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(0.2, 0.3), kappa_s=2.5, axis_ratio=0.5,
                                                              phi=100.0, inner_slope=1.5, scale_radius=4.0)
                defls = gnfw.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert defls[0,0] == pytest.approx(-4.02541, 1e-3)
                assert defls[0,1] == pytest.approx(-5.99032, 1e-3)

            def test__compare_to_spherical_gnfw__same_values(self):

                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=10.0,
                                                              axis_ratio=1.0,
                                                              phi=0.0, inner_slope=1.5, scale_radius=8.0)

                defls_0 = gnfw.deflections_from_grid(grid=np.array([[3.0, 3.0]]))

                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=10.0,
                                                             inner_slope=1.5,
                                                             scale_radius=8.0)

                defls_1 = gnfw.deflections_from_grid(grid=np.array([[3.0, 3.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,0], 1e-4)
                assert defls_0[0,1] == pytest.approx(defls_1[0,1], 1e-4)

            def test__compare_to_spherical_gnfw_2__same_values(self):
                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(-1.0, -2.0), kappa_s=1.0,
                                                              axis_ratio=1.0,
                                                              phi=100.0, inner_slope=0.5, scale_radius=3.0)

                defls_0 = gnfw.deflections_from_grid(grid=np.array([[1.0, -3.0]]))

                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(-1.0, -2.0), kappa_s=1.0,
                                                             inner_slope=0.5,
                                                             scale_radius=3.0)

                defls_1 = gnfw.deflections_from_grid(grid=np.array([[1.0, -3.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,1], 1e-5)

            def test__compare_to_elliptical_nfw__same_values(self):

                gnfw = mass_profiles.EllipticalGeneralizedNFW(centre=(-5.0, -10.0), kappa_s=0.1,
                                                              axis_ratio=0.5,
                                                              phi=100.0, inner_slope=1.0, scale_radius=20.0)

                defls_0 = gnfw.deflections_from_grid(grid=np.array([[-7.0, 0.2]]))

                gnfw = mass_profiles.EllipticalNFW(centre=(-5.0, -10.0), kappa_s=0.1, axis_ratio=0.5,
                                                   phi=100.0, scale_radius=20.0)

                defls_1 = gnfw.deflections_from_grid(grid=np.array([[-7.0, 0.2]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,0], 1e-4)
                assert defls_0[0,1] == pytest.approx(defls_1[0,1], 1e-4)


    class TestSphericalGeneralizedNFW(object):

        class TestSurfaceDensity(object):

            def test__simple_case__correct_value(self):

                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5,
                                                             scale_radius=1.0)

                surface_density = gnfw.surface_density_from_grid(grid=np.array([[2.0, 0.0]]))

                assert surface_density == pytest.approx(0.30840, 1e-3)

            def test__double_kappa__double_value(self):
                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=2.0, inner_slope=1.5,
                                                             scale_radius=1.0)
                surface_density = gnfw.surface_density_from_grid(grid=np.array([[2.0, 0.0]]))

                assert surface_density == pytest.approx(0.30840 * 2, 1e-3)

            def test__compare_to_spherical_nfw__same_value(self):
                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(0.8, 0.2), kappa_s=1.0, inner_slope=1.0,
                                                             scale_radius=2.0)
                surface_density_0 = gnfw.surface_density_from_grid(grid=np.array([[2.0, 0.0]]))

                nfw = mass_profiles.SphericalNFW(centre=(0.8, 0.2), kappa_s=1.0, scale_radius=2.0)
                surface_density_1 = nfw.surface_density_from_grid(grid=np.array([[2.0, 0.0]]))

                assert surface_density_0 == pytest.approx(surface_density_1, 1e-3)


        class TestPotential(object):

            def test__flip_coordinates_lens_center__same_value(self):
                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5,
                                                             scale_radius=10.0)

                potential_0 = gnfw.potential_from_grid(grid=np.array([[1.0, 1.0]]))

                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=1.0, inner_slope=1.5,
                                                             scale_radius=10.0)

                potential_1 = gnfw.potential_from_grid(grid=np.array([[2.0, 2.0]]))

                assert potential_0 == pytest.approx(potential_1, 1e-4)

            def test__compare_to_spherical_nfw__same_value(self):
                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=5.0, inner_slope=1.0,
                                                             scale_radius=10.0)

                potential_0 = gnfw.potential_from_grid(grid=np.array([[2.0, 2.0]]))

                nfw = mass_profiles.SphericalNFW(centre=(1.0, 1.0), kappa_s=5.0, scale_radius=10.0)

                potential_1 = nfw.potential_from_grid(grid=np.array([[2.0, 2.0]]))

                assert potential_0 == pytest.approx(potential_1, 1e-4)

            def test__compare_to_fortran__same_potential(self):

                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0,
                                                             inner_slope=0.5, scale_radius=8.0)

                potential = gnfw.potential_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert potential == pytest.approx(0.00920, 1e-3)

            def test__compare_to_fortran_2__same_potential(self):

                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0,
                                                             inner_slope=1.5, scale_radius=8.0)

                potential = gnfw.potential_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert potential == pytest.approx(0.17448, 1e-3)


        class TestDeflections(object):

            def test__flip_coordinates_lens_center__same_value(self):

                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5,
                                                             scale_radius=1.0)

                defls_0 = gnfw.deflections_from_grid(grid=np.array([[1.00001, 1.00001]]))

                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=1.0, inner_slope=1.5,
                                                             scale_radius=1.0)

                defls_1 = gnfw.deflections_from_grid(grid=np.array([[0.00001, 0.00001]]))

                assert defls_0[0,0] == pytest.approx(-defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(-defls_1[0,1], 1e-5)

            def test__compare_to_fortran_1(self):

                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0,
                                                             inner_slope=0.5, scale_radius=8.0)

                defls = gnfw.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert defls[0,0] == pytest.approx(0.37701, 1e-3)
                assert defls[0,1] == pytest.approx(0.43501, 1e-3)

            def test__compare_to_fortran_2(self):
                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(0.2, 0.3), kappa_s=2.5,
                                                             inner_slope=1.5, scale_radius=4.0)
                defls = gnfw.deflections_from_grid(grid=np.array([[0.1625, 0.1875]]))

                assert defls[0,0] == pytest.approx(-3.10418, 1e-3)
                assert defls[0,1] == pytest.approx(-9.31254, 1e-3)

            def test__compare_to_spherical_nfw__same_values(self):

                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=10.0,
                                                             inner_slope=1.0,
                                                             scale_radius=0.1)

                defls_0 = gnfw.deflections_from_grid(grid=np.array([[3.0, 3.0]]))

                gnfw = mass_profiles.SphericalNFW(centre=(1.0, 1.0), kappa_s=10.0, scale_radius=0.1)

                defls_1 = gnfw.deflections_from_grid(grid=np.array([[3.0, 3.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,1], 1e-5)

            def test__2_coordinates__2_sets_of_deflections(self):

                gnfw = mass_profiles.SphericalGeneralizedNFW(centre=(0.2, 0.3), kappa_s=2.5,
                                                             inner_slope=1.5, scale_radius=4.0)
                defls = gnfw.deflections_from_grid(grid=np.array([[0.1625, 0.1875], [0.1625, 0.1875]]))

                assert defls[0,0] == pytest.approx(-3.10418, 1e-3)
                assert defls[0,1] == pytest.approx(-9.31254, 1e-3)
                assert defls[1,0] == pytest.approx(-3.10418, 1e-3)
                assert defls[1,1] == pytest.approx(-9.31254, 1e-3)


    class TestSersicMass(object):

        class TestSurfaceDensity(object):

            def test__flip_coordinates_lens_center__same_value(self):
                sersic = mass_profiles.EllipticalSersicMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                            effective_radius=1.0, sersic_index=4.0,
                                                            mass_to_light_ratio=1.0)

                surface_density_0 = sersic.surface_density_from_grid(grid=np.array([[1.0, 1.0]]))

                sersic = mass_profiles.EllipticalSersicMass(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                            effective_radius=1.0, sersic_index=4.0,
                                                            mass_to_light_ratio=1.0)

                surface_density_1 = sersic.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_coordinates_90_circular__same_value(self):
                sersic = mass_profiles.EllipticalSersicMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                            effective_radius=1.0, sersic_index=4.0,
                                                            mass_to_light_ratio=1.0)

                surface_density_0 = sersic.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                sersic = mass_profiles.EllipticalSersicMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0, intensity=1.0,
                                                            effective_radius=1.0, sersic_index=4.0,
                                                            mass_to_light_ratio=1.0)

                surface_density_1 = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                sersic = mass_profiles.EllipticalSersicMass(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, intensity=1.0,
                                                            effective_radius=1.0, sersic_index=4.0,
                                                            mass_to_light_ratio=1.0)

                surface_density_0 = sersic.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                sersic = mass_profiles.EllipticalSersicMass(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=1.0,
                                                            effective_radius=1.0, sersic_index=4.0,
                                                            mass_to_light_ratio=1.0)

                surface_density_1 = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__different_rotate_phi_90_same_result(self):
                sersic = mass_profiles.EllipticalSersicMass(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                                            effective_radius=2.0, sersic_index=2.0,
                                                            mass_to_light_ratio=1.0)

                surface_density_0 = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                sersic = mass_profiles.EllipticalSersicMass(axis_ratio=0.5, phi=90.0, intensity=3.0,
                                                            effective_radius=2.0, sersic_index=2.0,
                                                            mass_to_light_ratio=1.0)

                surface_density_1 = sersic.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__simple_case__correct_value(self):
                sersic = mass_profiles.EllipticalSersicMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                            effective_radius=0.6, sersic_index=4.0,
                                                            mass_to_light_ratio=1.0)

                surface_density = sersic.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx(0.351797, 1e-3)

            def test__simple_case_2__correct_value(self):
                sersic = mass_profiles.EllipticalSersicMass(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                            effective_radius=2.0, sersic_index=2.0,
                                                            mass_to_light_ratio=1.0)

                surface_density = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.5]]))

                assert surface_density == pytest.approx(4.90657319276, 1e-3)

            def test__double_intensity__doubles_value(self):
                sersic = mass_profiles.EllipticalSersicMass(axis_ratio=1.0, phi=0.0, intensity=6.0,
                                                            effective_radius=2.0, sersic_index=2.0,
                                                            mass_to_light_ratio=1.0)

                surface_density = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.5]]))

                assert surface_density == pytest.approx(2.0 * 4.90657319276, 1e-3)

            def test__double_mass_to_light_ratio__doubles_value(self):
                sersic = mass_profiles.EllipticalSersicMass(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                            effective_radius=2.0, sersic_index=2.0,
                                                            mass_to_light_ratio=2.0)

                surface_density = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.5]]))

                assert surface_density == pytest.approx(2.0 * 4.90657319276, 1e-3)

            def test__different_axis_ratio__new_value(self):
                sersic = mass_profiles.EllipticalSersicMass(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                                            effective_radius=2.0, sersic_index=2.0,
                                                            mass_to_light_ratio=1.0)

                surface_density = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density == pytest.approx(5.38066670129, 1e-3)

        class TestDeflections(object):

            def test__flip_coordinates_lens_center__same_value(self):

                sersic = mass_profiles.EllipticalSersicMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                            effective_radius=1.0, sersic_index=4.0,
                                                            mass_to_light_ratio=1.0)

                defls_0 = sersic.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                sersic = mass_profiles.EllipticalSersicMass(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                            effective_radius=1.0, sersic_index=4.0,
                                                            mass_to_light_ratio=1.0)

                defls_1 = sersic.deflections_from_grid(grid=np.array([[0.0, 0.0]]))

                assert defls_0[0,0] == pytest.approx(-defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(-defls_1[0,1], 1e-5)

            def test__rotation_coordinates_90_circular__same_value(self):
                sersic = mass_profiles.EllipticalSersicMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                            effective_radius=1.0, sersic_index=4.0,
                                                            mass_to_light_ratio=1.0)

                defls_0 = sersic.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                sersic = mass_profiles.EllipticalSersicMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0, intensity=1.0,
                                                            effective_radius=1.0, sersic_index=4.0,
                                                            mass_to_light_ratio=1.0)

                defls_1 = sersic.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                # Foro deflection angles, a 90 degree rtation flips the x / y image_grid

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                sersic = mass_profiles.EllipticalSersicMass(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, intensity=1.0,
                                                            effective_radius=1.0, sersic_index=4.0,
                                                            mass_to_light_ratio=1.0)

                defls_0 = sersic.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                sersic = mass_profiles.EllipticalSersicMass(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=1.0,
                                                            effective_radius=1.0, sersic_index=4.0,
                                                            mass_to_light_ratio=1.0)

                defls_1 = sersic.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__compare_to_dev_vaucauleurs(self):
                sersic = mass_profiles.EllipticalSersicMass(centre=(0.2, 0.4), axis_ratio=0.9, phi=10.0, intensity=2.0,
                                                            effective_radius=0.8, sersic_index=4.0,
                                                            mass_to_light_ratio=3.0)

                sersic_defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.2, 0.4), axis_ratio=0.9,
                                                                             phi=10.0,
                                                                             intensity=2.0,
                                                                             effective_radius=0.8,
                                                                             mass_to_light_ratio=3.0)

                dev_vaucouleurs_defls = dev_vaucouleurs.deflections_from_grid(
                    grid=np.array([[0.1625, 0.1625]]))

                assert sersic_defls[0,0] == dev_vaucouleurs_defls[0,0] == pytest.approx(-3.37605, 1e-3)
                assert sersic_defls[0,1] == dev_vaucouleurs_defls[0,1] == pytest.approx(-24.528, 1e-3)

            def test__compare_to_exponential(self):
                sersic = mass_profiles.EllipticalSersicMass(centre=(-0.2, -0.4), axis_ratio=0.8, phi=110.0,
                                                            intensity=5.0,
                                                            effective_radius=0.2, sersic_index=1.0,
                                                            mass_to_light_ratio=1.00)

                sersic_defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                exponential = mass_profiles.EllipticalExponentialMass(centre=(-0.2, -0.4), axis_ratio=0.8, phi=110.0,
                                                                      intensity=5.0,
                                                                      effective_radius=0.2, mass_to_light_ratio=1.0)

                exponential_defls = exponential.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert sersic_defls[0,0] == exponential_defls[0,0] == pytest.approx(0.62569, 1e-3)
                assert sersic_defls[0,1] == exponential_defls[0,1] == pytest.approx(0.90493, 1e-3)

            def test__compare_to_fortran_sersic_index_2(self):

                sersic = mass_profiles.EllipticalSersicMass(centre=(-0.2, -0.4), axis_ratio=0.8, phi=110.0,
                                                            intensity=5.0,
                                                            effective_radius=0.2, sersic_index=2.0,
                                                            mass_to_light_ratio=1.0)

                defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.79374, 1e-3)
                assert defls[0,1] == pytest.approx(1.1446, 1e-3)

            def test__2_coordinates__2_sets_of_deflections(self):

                sersic = mass_profiles.EllipticalSersicMass(centre=(-0.2, -0.4), axis_ratio=0.8, phi=110.0,
                                                            intensity=5.0,
                                                            effective_radius=0.2, sersic_index=2.0,
                                                            mass_to_light_ratio=1.0)

                defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.79374, 1e-3)
                assert defls[0,1] == pytest.approx(1.1446, 1e-3)
                assert defls[1,0] == pytest.approx(0.79374, 1e-3)
                assert defls[1,1] == pytest.approx(1.1446, 1e-3)

            def test__from_light_profile(self):
                light_sersic = light_profiles.EllipticalSersic(centre=(-0.2, -0.4), axis_ratio=0.8,
                                                                             phi=110.0,
                                                                             intensity=5.0, effective_radius=0.2,
                                                                             sersic_index=2.0)
                mass_sersic = mass_profiles.EllipticalSersicMass.from_profile(light_sersic,
                                                                              mass_to_light_ratio=1.)

                defls = mass_sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.79374, 1e-3)
                assert defls[0,1] == pytest.approx(1.1446, 1e-3)


    class TestExponentialMass(object):
        class TestSurfaceDensity(object):

            def test__flip_coordinates_lens_center__same_value(self):
                exponential = mass_profiles.EllipticalExponentialMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                      intensity=1.0,
                                                                      effective_radius=1.0, mass_to_light_ratio=1.0)

                surface_density_0 = exponential.surface_density_from_grid(grid=np.array([[1.0, 1.0]]))

                exponential = mass_profiles.EllipticalExponentialMass(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                      intensity=1.0,
                                                                      effective_radius=1.0, mass_to_light_ratio=1.0)

                surface_density_1 = exponential.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_coordinates_90_circular__same_value(self):
                exponential = mass_profiles.EllipticalExponentialMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                      intensity=1.0,
                                                                      effective_radius=1.0, mass_to_light_ratio=1.0)

                surface_density_0 = exponential.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                exponential = mass_profiles.EllipticalExponentialMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                                      intensity=1.0,
                                                                      effective_radius=1.0, mass_to_light_ratio=1.0)

                surface_density_1 = exponential.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                exponential = mass_profiles.EllipticalExponentialMass(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                      intensity=1.0,
                                                                      effective_radius=1.0, mass_to_light_ratio=1.0)

                surface_density_0 = exponential.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                exponential = mass_profiles.EllipticalExponentialMass(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                                      intensity=1.0,
                                                                      effective_radius=1.0, mass_to_light_ratio=1.0)

                surface_density_1 = exponential.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__different_rotate_phi_90_same_result(self):
                exponential = mass_profiles.EllipticalExponentialMass(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                                                      effective_radius=2.0, mass_to_light_ratio=1.0)

                surface_density_0 = exponential.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                exponential = mass_profiles.EllipticalExponentialMass(axis_ratio=0.5, phi=90.0, intensity=3.0,
                                                                      effective_radius=2.0, mass_to_light_ratio=1.0)

                surface_density_1 = exponential.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__simple_case_1__correct_value(self):
                exponential = mass_profiles.EllipticalExponentialMass(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                                                      effective_radius=2.0, mass_to_light_ratio=1.0)

                surface_density = exponential.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density == pytest.approx(4.9047, 1e-3)

            def test__simple_case_2__correct_value(self):
                exponential = mass_profiles.EllipticalExponentialMass(axis_ratio=0.5, phi=90.0, intensity=2.0,
                                                                      effective_radius=3.0, mass_to_light_ratio=1.0)

                surface_density = exponential.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx(4.8566, 1e-3)

            def test__double_intensity__doubles_value(self):
                exponential = mass_profiles.EllipticalExponentialMass(axis_ratio=0.5, phi=90.0, intensity=4.0,
                                                                      effective_radius=3.0, mass_to_light_ratio=1.0)

                surface_density = exponential.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx(2.0 * 4.8566, 1e-3)

            def test__double_mass_to_light_ratio__doubles_value(self):
                exponential = mass_profiles.EllipticalExponentialMass(axis_ratio=0.5, phi=90.0, intensity=2.0,
                                                                      effective_radius=3.0, mass_to_light_ratio=2.0)

                surface_density = exponential.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx(2.0 * 4.8566, 1e-3)

            def test__different_axis_ratio__new_value(self):
                exponential = mass_profiles.EllipticalExponentialMass(axis_ratio=0.5, phi=90.0, intensity=2.0,
                                                                      effective_radius=3.0, mass_to_light_ratio=1.0)

                surface_density = exponential.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx(4.8566, 1e-3)

        class TestDeflections(object):
            def test__flip_coordinates_lens_center__same_value(self):
                exponential = mass_profiles.EllipticalExponentialMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                      intensity=1.0,
                                                                      effective_radius=1.0, mass_to_light_ratio=1.0)

                defls_0 = exponential.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                exponential = mass_profiles.EllipticalExponentialMass(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                      intensity=1.0,
                                                                      effective_radius=1.0, mass_to_light_ratio=1.0)

                defls_1 = exponential.deflections_from_grid(grid=np.array([[0.0, 0.0]]))

                assert defls_0[0,0] == pytest.approx(-defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(-defls_1[0,1], 1e-5)

            def test__rotation_coordinates_90_circular__same_value(self):

                exponential = mass_profiles.EllipticalExponentialMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                      intensity=1.0,
                                                                      effective_radius=1.0, mass_to_light_ratio=1.0)

                defls_0 = exponential.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                exponential = mass_profiles.EllipticalExponentialMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                                      intensity=1.0,
                                                                      effective_radius=1.0, mass_to_light_ratio=1.0)

                defls_1 = exponential.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                # Foro deflection angles, a 90 degree rtation flips the x / y image_grid

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                exponential = mass_profiles.EllipticalExponentialMass(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                      intensity=1.0,
                                                                      effective_radius=1.0, mass_to_light_ratio=1.0)

                defls_0 = exponential.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                exponential = mass_profiles.EllipticalExponentialMass(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                                      intensity=1.0,
                                                                      effective_radius=1.0, mass_to_light_ratio=1.0)

                defls_1 = exponential.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__compare_to_fortran(self):

                exponential = mass_profiles.EllipticalExponentialMass(centre=(-0.2, -0.4), axis_ratio=0.8, phi=110.0,
                                                                      intensity=5.0,
                                                                      effective_radius=0.2, mass_to_light_ratio=1.0)

                defls = exponential.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.62569, 1e-3)
                assert defls[0,1] == pytest.approx(0.90493, 1e-3)

            def test__double_intensity__double_defls(self):
                exponential = mass_profiles.EllipticalExponentialMass(centre=(-0.2, -0.4), axis_ratio=0.8, phi=110.0,
                                                                      intensity=5.0,
                                                                      effective_radius=0.2, mass_to_light_ratio=1.0)

                defls = exponential.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.62569, 1e-3)
                assert defls[0,1] == pytest.approx(0.90493, 1e-3)

            def test__2_coordinates__2_sets_of_deflections(self):

                exponential = mass_profiles.EllipticalExponentialMass(centre=(-0.2, -0.4), axis_ratio=0.8, phi=110.0,
                                                                      intensity=5.0,
                                                                      effective_radius=0.2, mass_to_light_ratio=1.0)

                defls = exponential.deflections_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.62569, 1e-3)
                assert defls[0,1] == pytest.approx(0.90493, 1e-3)
                assert defls[1,0] == pytest.approx(0.62569, 1e-3)
                assert defls[1,1] == pytest.approx(0.90493, 1e-3)

            def test__from_light_profile(self):
                light_exponential = light_profiles.EllipticalExponential(
                    centre=(-0.2, -0.4), axis_ratio=0.8, phi=110.0, intensity=5.0, effective_radius=0.2)

                mass_exponential = mass_profiles.EllipticalExponentialMass.from_exponential_light_profile(
                    light_exponential,
                    mass_to_light_ratio=1.)

                defls = mass_exponential.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.62569, 1e-3)
                assert defls[0,1] == pytest.approx(0.90493, 1e-3)


    class TestDevVaucouleursMass(object):

        class TestSurfaceDensity(object):

            def test__flip_coordinates_lens_center__same_value(self):
                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                             intensity=1.0,
                                                                             effective_radius=1.0,
                                                                             mass_to_light_ratio=1.0)

                surface_density_0 = dev_vaucouleurs.surface_density_from_grid(grid=np.array([[1.0, 1.0]]))

                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                             intensity=1.0,
                                                                             effective_radius=1.0,
                                                                             mass_to_light_ratio=1.0)

                surface_density_1 = dev_vaucouleurs.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_coordinates_90_circular__same_value(self):
                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                             intensity=1.0,
                                                                             effective_radius=1.0,
                                                                             mass_to_light_ratio=1.0)

                surface_density_0 = dev_vaucouleurs.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                             phi=90.0,
                                                                             intensity=1.0,
                                                                             effective_radius=1.0,
                                                                             mass_to_light_ratio=1.0)

                surface_density_1 = dev_vaucouleurs.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                             intensity=1.0,
                                                                             effective_radius=1.0,
                                                                             mass_to_light_ratio=1.0)

                surface_density_0 = dev_vaucouleurs.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                             phi=90.0,
                                                                             intensity=1.0,
                                                                             effective_radius=1.0,
                                                                             mass_to_light_ratio=1.0)

                surface_density_1 = dev_vaucouleurs.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__different_rotate_phi_90_same_result(self):
                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                                                             effective_radius=2.0,
                                                                             mass_to_light_ratio=1.0)

                surface_density_0 = dev_vaucouleurs.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(axis_ratio=0.5, phi=90.0, intensity=3.0,
                                                                             effective_radius=2.0,
                                                                             mass_to_light_ratio=1.0)

                surface_density_1 = dev_vaucouleurs.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__simple_case_1__correct_value(self):
                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                                                             effective_radius=2.0,
                                                                             mass_to_light_ratio=1.0)

                surface_density = dev_vaucouleurs.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density == pytest.approx(5.6697, 1e-3)

            def test__simple_case_2__correct_value(self):
                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(axis_ratio=0.5, phi=90.0, intensity=2.0,
                                                                             effective_radius=3.0,
                                                                             mass_to_light_ratio=1.0)

                surface_density = dev_vaucouleurs.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx(7.4455, 1e-3)

            def test__double_intensity__doubles_value(self):
                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(axis_ratio=0.5, phi=90.0, intensity=4.0,
                                                                             effective_radius=3.0,
                                                                             mass_to_light_ratio=1.0)

                surface_density = dev_vaucouleurs.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx(2.0 * 7.4455, 1e-3)

            def test__double_mass_to_light_ratio__doubles_value(self):
                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(axis_ratio=0.5, phi=90.0, intensity=2.0,
                                                                             effective_radius=3.0,
                                                                             mass_to_light_ratio=2.0)

                surface_density = dev_vaucouleurs.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx(2.0 * 7.4455, 1e-3)

        class TestDeflections(object):

            def test__flip_coordinates_lens_center__same_value(self):
                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                             intensity=1.0,
                                                                             effective_radius=1.0,
                                                                             mass_to_light_ratio=1.0)

                defls_0 = dev_vaucouleurs.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                             intensity=1.0,
                                                                             effective_radius=1.0,
                                                                             mass_to_light_ratio=1.0)

                defls_1 = dev_vaucouleurs.deflections_from_grid(grid=np.array([[0.0, 0.0]]))

                assert defls_0[0,0] == pytest.approx(-defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(-defls_1[0,1], 1e-5)

            def test__rotation_coordinates_90_circular__same_value(self):
                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                             intensity=1.0,
                                                                             effective_radius=1.0,
                                                                             mass_to_light_ratio=1.0)

                defls_0 = dev_vaucouleurs.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                             phi=90.0,
                                                                             intensity=1.0,
                                                                             effective_radius=1.0,
                                                                             mass_to_light_ratio=1.0)

                defls_1 = dev_vaucouleurs.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                # Foro deflection angles, a 90 degree rtation flips the x / y image_grid

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                             intensity=1.0,
                                                                             effective_radius=1.0,
                                                                             mass_to_light_ratio=1.0)

                defls_0 = dev_vaucouleurs.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.0, 0.0), axis_ratio=0.8,
                                                                             phi=90.0,
                                                                             intensity=1.0,
                                                                             effective_radius=1.0,
                                                                             mass_to_light_ratio=1.0)

                defls_1 = dev_vaucouleurs.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__compare_to_fortran(self):
                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.2, 0.4), axis_ratio=0.9,
                                                                             phi=10.0,
                                                                             intensity=2.0,
                                                                             effective_radius=0.8,
                                                                             mass_to_light_ratio=3.0)

                defls = dev_vaucouleurs.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] / defls[0,1] == pytest.approx(0.1376, 1e-3)
                assert defls[0,0] == pytest.approx(-3.37605, 1e-3)
                assert defls[0,1] == pytest.approx(-24.528, 1e-3)

            def test__2_coordinates__2_sets_of_deflections(self):

                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.2, 0.4), axis_ratio=0.9,
                                                                             phi=10.0,
                                                                             intensity=2.0,
                                                                             effective_radius=0.8,
                                                                             mass_to_light_ratio=3.0)

                defls = dev_vaucouleurs.deflections_from_grid(grid=np.array([[0.1625, 0.1625],
                                                                             [0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(-3.37605, 1e-3)
                assert defls[0,1] == pytest.approx(-24.528, 1e-3)
                assert defls[1,0] == pytest.approx(-3.37605, 1e-3)
                assert defls[1,1] == pytest.approx(-24.528, 1e-3)


            def test__from_light_profile(self):
                light_dev_vaucouleurs = light_profiles.EllipticalDevVaucouleurs(
                    centre=(0.2, 0.4), axis_ratio=0.9, phi=10.0, intensity=2.0, effective_radius=0.8)

                mass_dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass.from_dev_vaucouleurs_light_profile(
                    light_dev_vaucouleurs,
                    mass_to_light_ratio=3.)

                defls = mass_dev_vaucouleurs.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(-3.37605, 1e-3)
                assert defls[0,1] == pytest.approx(-24.528, 1e-3)


    class TestSersicMassRadialGradient(object):

        class TestSurfaceDensity(object):

            def test__flip_coordinates_lens_center__same_value(self):

                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                          intensity=1.0,
                                                                          effective_radius=1.0, sersic_index=4.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                surface_density_0 = sersic.surface_density_from_grid(grid=np.array([[1.0, 1.0]]))

                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                          intensity=1.0,
                                                                          effective_radius=1.0, sersic_index=4.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                surface_density_1 = sersic.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_coordinates_90_circular__same_value(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                          intensity=1.0,
                                                                          effective_radius=1.0, sersic_index=4.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                surface_density_0 = sersic.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                                          intensity=1.0,
                                                                          effective_radius=1.0, sersic_index=4.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                surface_density_1 = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                          intensity=1.0,
                                                                          effective_radius=1.0, sersic_index=4.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                surface_density_0 = sersic.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                                          intensity=1.0,
                                                                          effective_radius=1.0, sersic_index=4.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                surface_density_1 = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density_0 == surface_density_1

            def test__different_rotate_phi_90_same_result(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                                                          effective_radius=2.0, sersic_index=2.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                surface_density_0 = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                sersic = mass_profiles.EllipticalSersicMassRadialGradient(axis_ratio=0.5, phi=90.0, intensity=3.0,
                                                                          effective_radius=2.0, sersic_index=2.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                surface_density_1 = sersic.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density_0 == surface_density_1

            def test__simple_case__correct_value(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                          intensity=1.0,
                                                                          effective_radius=0.6, sersic_index=4.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1/0.6)**-1.0 = 0.6

                surface_density = sersic.surface_density_from_grid(grid=np.array([[1.0, 0.0]]))

                assert surface_density == pytest.approx((0.6) * 0.351797, 1e-3)

            def test__simple_case_2__correct_value(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                                          effective_radius=2.0, sersic_index=2.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=-1.0)

                # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1.5/2.0)**1.0 = 0.75

                surface_density = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.5]]))

                assert surface_density == pytest.approx((0.75) * 4.90657319276, 1e-3)

            def test__double_intensity__doubles_value(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(axis_ratio=1.0, phi=0.0, intensity=6.0,
                                                                          effective_radius=2.0, sersic_index=2.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=-1.0)

                surface_density = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.5]]))

                assert surface_density == pytest.approx(2.0 * (0.75) * 4.90657319276, 1e-3)

            def test__double_mass_to_light_ratio__doubles_value(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                                          effective_radius=2.0, sersic_index=2.0,
                                                                          mass_to_light_ratio=2.0,
                                                                          mass_to_light_gradient=-1.0)

                surface_density = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.5]]))

                assert surface_density == pytest.approx(2.0 * (0.75) * 4.90657319276, 1e-3)

            def test__different_axis_ratio__new_value(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                                                          effective_radius=2.0, sersic_index=2.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = ((0.5*1.41)/2.0)**-1.0 = 2.836

                surface_density = sersic.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

                assert surface_density == pytest.approx((2.836879) * 5.38066670129, 1e-2)

        class TestDeflections(object):

            def test__flip_coordinates_lens_center__same_value(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                          intensity=1.0,
                                                                          effective_radius=1.0, sersic_index=4.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                defls_0 = sersic.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                          intensity=1.0,
                                                                          effective_radius=1.0, sersic_index=4.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                defls_1 = sersic.deflections_from_grid(grid=np.array([[0.0, 0.0]]))

                assert defls_0[0,0] == pytest.approx(-defls_1[0,0], 1e-5)
                assert defls_0[0,1] == pytest.approx(-defls_1[0,1], 1e-5)

            def test__rotation_coordinates_90_circular__same_value(self):

                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                          intensity=1.0,
                                                                          effective_radius=1.0, sersic_index=4.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                defls_0 = sersic.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                                          intensity=1.0,
                                                                          effective_radius=1.0, sersic_index=4.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                defls_1 = sersic.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                # Foro deflection angles, a 90 degree rtation flips the x / y image_grid

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):

                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                          intensity=1.0,
                                                                          effective_radius=1.0, sersic_index=4.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                defls_0 = sersic.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                                          intensity=1.0,
                                                                          effective_radius=1.0, sersic_index=4.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                defls_1 = sersic.deflections_from_grid(grid=np.array([[0.0, 1.0]]))

                assert defls_0[0,0] == pytest.approx(defls_1[0,1], 1e-5)
                assert defls_0[0,1] == pytest.approx(defls_1[0,0], 1e-5)

            def test__compare_to_dev_vaucauleurs_without_radial_gradient(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(0.2, 0.4), axis_ratio=0.9, phi=10.0,
                                                                          intensity=2.0,
                                                                          effective_radius=0.8, sersic_index=4.0,
                                                                          mass_to_light_ratio=3.0,
                                                                          mass_to_light_gradient=0.0)

                sersic_defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                dev_vaucouleurs = mass_profiles.EllipticalDevVaucouleursMass(centre=(0.2, 0.4), axis_ratio=0.9,
                                                                             phi=10.0,
                                                                             intensity=2.0,
                                                                             effective_radius=0.8,
                                                                             mass_to_light_ratio=3.0)

                dev_vaucouleurs_defls = dev_vaucouleurs.deflections_from_grid(
                    grid=np.array([[0.1625, 0.1625]]))

                assert sersic_defls[0,0] == dev_vaucouleurs_defls[0,0] == pytest.approx(-3.37605, 1e-3)
                assert sersic_defls[0,1] == dev_vaucouleurs_defls[0,1] == pytest.approx(-24.528, 1e-3)

            def test__compare_to_exponential_without_radial_gradient(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(-0.2, -0.4), axis_ratio=0.8,
                                                                          phi=110.0, intensity=5.0,
                                                                          effective_radius=0.2, sersic_index=1.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=0.0)

                sersic_defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                exponential = mass_profiles.EllipticalExponentialMass(centre=(-0.2, -0.4), axis_ratio=0.8, phi=110.0,
                                                                      intensity=5.0,
                                                                      effective_radius=0.2, mass_to_light_ratio=1.0)

                exponential_defls = exponential.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert sersic_defls[0,0] == exponential_defls[0,0] == pytest.approx(0.62569, 1e-3)
                assert sersic_defls[0,1] == exponential_defls[0,1] == pytest.approx(0.90493, 1e-3)

            def test__compare_to_fortran_sersic_index_no_gradient(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(-0.2, -0.4), axis_ratio=0.8,
                                                                          phi=110.0, intensity=5.0,
                                                                          effective_radius=0.2, sersic_index=2.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=0.0)

                defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.79374, 1e-3)
                assert defls[0,1] == pytest.approx(1.1446, 1e-3)

            def test__compare_to_fortran_with_gradient_positive(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(-0.2, -0.4), axis_ratio=0.8,
                                                                          phi=110.0, intensity=5.0,
                                                                          effective_radius=0.2, sersic_index=2.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=1.0)

                defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(2.3638898009652, 1e-3)
                assert defls[0,1] == pytest.approx(3.60324873535244, 1e-3)

            def test__compare_to_fortran_with_gradient_negative(self):
                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(-0.2, -0.4), axis_ratio=0.8,
                                                                          phi=110.0, intensity=5.0,
                                                                          effective_radius=0.2, sersic_index=2.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=-1.0)

                defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.725459334118341, 1e-3)
                assert defls[0,1] == pytest.approx(0.97806399756448, 1e-3)

            def test__2_coordinates__2_sets_of_deflections(self):

                sersic = mass_profiles.EllipticalSersicMassRadialGradient(centre=(-0.2, -0.4), axis_ratio=0.8,
                                                                          phi=110.0, intensity=5.0,
                                                                          effective_radius=0.2, sersic_index=2.0,
                                                                          mass_to_light_ratio=1.0,
                                                                          mass_to_light_gradient=-1.0)

                defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.725459334118341, 1e-3)
                assert defls[0,1] == pytest.approx(0.97806399756448, 1e-3)
                assert defls[1,0] == pytest.approx(0.725459334118341, 1e-3)
                assert defls[1,1] == pytest.approx(0.97806399756448, 1e-3)

            def test__from_light_profile__deflection_angles_unchanged(self):
                light_sersic = light_profiles.EllipticalSersic(centre=(-0.2, -0.4), axis_ratio=0.8,
                                                                             phi=110.0,
                                                                             intensity=5.0, effective_radius=0.2,
                                                                             sersic_index=2.0)

                mass_sersic = mass_profiles.EllipticalSersicMassRadialGradient.from_profile(light_sersic,
                                                                                            mass_to_light_ratio=1.0,
                                                                                            mass_to_light_gradient=0.0)

                defls = mass_sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.79374, 1e-3)
                assert defls[0,1] == pytest.approx(1.1446, 1e-3)


    class TestExternalShear(object):

        class TestDeflections(object):

            def test__compare_to_fortran_1(self):

                shear = mass_profiles.ExternalShear(magnitude=0.1, phi=45.0)
                defls = shear.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(0.01625, 1e-3)
                assert defls[0,1] == pytest.approx(0.01625, 1e-3)

            def test__compare_to_fortran_2(self):

                shear = mass_profiles.ExternalShear(magnitude=0.2, phi=75.0)
                defls = shear.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(-0.011895, 1e-3)
                assert defls[0,1] == pytest.approx(0.04439, 1e-3)

            def test__2_coordinates__2_sets_of_deflections(self):

                shear = mass_profiles.ExternalShear(magnitude=0.2, phi=75.0)
                defls = shear.deflections_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))

                assert defls[0,0] == pytest.approx(-0.011895, 1e-3)
                assert defls[0,1] == pytest.approx(0.04439, 1e-3)
                assert defls[1,0] == pytest.approx(-0.011895, 1e-3)
                assert defls[1,1] == pytest.approx(0.04439, 1e-3)


class TestMassIntegral(object):
    class TestWithinCircle(object):

        def test__singular_isothermal_sphere__compare_to_analytic1(self):

            import math

            sis = mass_profiles.SphericalIsothermal(einstein_radius=2.0)

            integral_radius = 2.0

            dimensionless_mass_integral = sis.dimensionless_mass_within_circle(radius=integral_radius)

            assert math.pi * sis.einstein_radius * integral_radius == pytest.approx(dimensionless_mass_integral, 1e-3)

        def test__singular_isothermal_sphere__compare_to_analytic2(self):

            import math

            sis = mass_profiles.SphericalIsothermal(einstein_radius=4.0)

            integral_radius = 4.0

            dimensionless_mass_integral = sis.dimensionless_mass_within_circle(radius=integral_radius)

            assert math.pi * sis.einstein_radius * integral_radius == pytest.approx(dimensionless_mass_integral, 1e-3)

        def test__singular_isothermal__compare_to_grid(self):

            sis = mass_profiles.SphericalIsothermal(einstein_radius=2.0)

            import math
            import numpy as np

            integral_radius = 1.0
            dimensionless_mass_total = 0.0

            xs = np.linspace(-1.5, 1.5, 40)
            ys = np.linspace(-1.5, 1.5, 40)

            edge = xs[1] - xs[0]
            area = edge ** 2

            for x in xs:
                for y in ys:

                    eta = math.sqrt(x ** 2 + y ** 2)

                    if eta < integral_radius:
                        dimensionless_mass_total += sis.surface_density_func(eta) * area

            dimensionless_mass_integral = sis.dimensionless_mass_within_circle(radius=integral_radius)

            assert dimensionless_mass_total == pytest.approx(dimensionless_mass_integral, 0.02)

        def test__elliptical_isothermal__compare_to_grid(self):

            sie = mass_profiles.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.2, phi=0.0)

            import math
            import numpy as np

            integral_radius = 1.0
            dimensionless_mass_total = 0.0

            xs = np.linspace(-1.5, 1.5, 40)
            ys = np.linspace(-1.5, 1.5, 40)

            edge = xs[1] - xs[0]
            area = edge ** 2

            for x in xs:
                for y in ys:

                    eta = math.sqrt(x ** 2 + y ** 2)

                    if eta < integral_radius:
                        dimensionless_mass_total += sie.surface_density_func(eta) * area

            dimensionless_mass_integral = sie.dimensionless_mass_within_circle(radius=integral_radius)

            assert dimensionless_mass_total == pytest.approx(dimensionless_mass_integral, 0.02)

        def test__cored_elliptical_isothermal__compare_to_grid(self):

            cored_sie = mass_profiles.EllipticalCoredIsothermal(einstein_radius=2.0, axis_ratio=0.2, phi=0.0,
                                                                core_radius=0.6)

            import math
            import numpy as np

            integral_radius = 1.0
            dimensionless_mass_total = 0.0

            xs = np.linspace(-1.5, 1.5, 40)
            ys = np.linspace(-1.5, 1.5, 40)

            edge = xs[1] - xs[0]
            area = edge ** 2

            for x in xs:
                for y in ys:

                    eta = math.sqrt(x ** 2 + y ** 2)

                    if eta < integral_radius:
                        dimensionless_mass_total += cored_sie.surface_density_func(eta) * area

            dimensionless_mass_integral = cored_sie.dimensionless_mass_within_circle(radius=integral_radius)

            assert dimensionless_mass_total == pytest.approx(dimensionless_mass_integral, 0.02)

        def test__cored_power_law_isothermal__compare_to_grid1(self):

            cored_power_law = mass_profiles.EllipticalCoredPowerLaw(einstein_radius=2.0, axis_ratio=0.2, phi=0.0,
                                                                    slope=2.7, core_radius=0.6)

            import math
            import numpy as np

            integral_radius = 1.0
            dimensionless_mass_total = 0.0

            xs = np.linspace(-1.5, 1.5, 40)
            ys = np.linspace(-1.5, 1.5, 40)

            edge = xs[1] - xs[0]
            area = edge ** 2

            for x in xs:
                for y in ys:

                    eta = math.sqrt(x ** 2 + y ** 2)

                    if eta < integral_radius:
                        dimensionless_mass_total += cored_power_law.surface_density_func(eta) * area

            dimensionless_mass_integral = cored_power_law.dimensionless_mass_within_circle(radius=integral_radius)

            assert dimensionless_mass_total == pytest.approx(dimensionless_mass_integral, 0.02)

        def test__cored_power_law_isothermal__compare_to_grid2(self):

            cored_power_law = mass_profiles.EllipticalCoredPowerLaw(einstein_radius=2.0, axis_ratio=0.2, phi=0.0,
                                                                    slope=1.3, core_radius=0.6)

            import math
            import numpy as np

            integral_radius = 1.0
            dimensionless_mass_total = 0.0

            xs = np.linspace(-1.5, 1.5, 40)
            ys = np.linspace(-1.5, 1.5, 40)

            edge = xs[1] - xs[0]
            area = edge ** 2

            for x in xs:
                for y in ys:

                    eta = math.sqrt(x ** 2 + y ** 2)

                    if eta < integral_radius:
                        dimensionless_mass_total += cored_power_law.surface_density_func(eta) * area

            dimensionless_mass_integral = cored_power_law.dimensionless_mass_within_circle(radius=integral_radius)

            assert dimensionless_mass_total == pytest.approx(dimensionless_mass_integral, 0.02)

        def test__elliptical_nfw_profile__compare_to_grid(self):

            nfw = mass_profiles.EllipticalNFW(kappa_s=2.0, axis_ratio=0.2, phi=0.0, scale_radius=5.0)

            import math
            import numpy as np

            integral_radius = 1.0
            dimensionless_mass_total = 0.0

            xs = np.linspace(-1.5, 1.5, 40)
            ys = np.linspace(-1.5, 1.5, 40)

            edge = xs[1] - xs[0]
            area = edge ** 2

            for x in xs:
                for y in ys:

                    eta = math.sqrt(x ** 2 + y ** 2)

                    if eta < integral_radius:
                        dimensionless_mass_total += nfw.surface_density_func(eta) * area

            dimensionless_mass_integral = nfw.dimensionless_mass_within_circle(radius=integral_radius)

            assert dimensionless_mass_total == pytest.approx(dimensionless_mass_integral, 0.02)

        def test__elliptical_gnfw_profile__compare_to_grid1(self):

            nfw = mass_profiles.EllipticalGeneralizedNFW(kappa_s=2.0, axis_ratio=0.2, phi=0.0, inner_slope=0.2,
                                                         scale_radius=5.0)

            import math
            import numpy as np

            integral_radius = 1.0
            dimensionless_mass_total = 0.0

            xs = np.linspace(-1.5, 1.5, 40)
            ys = np.linspace(-1.5, 1.5, 40)

            edge = xs[1] - xs[0]
            area = edge ** 2

            for x in xs:
                for y in ys:

                    eta = math.sqrt(x ** 2 + y ** 2)

                    if eta < integral_radius:
                        dimensionless_mass_total += nfw.surface_density_func(eta) * area

            dimensionless_mass_integral = nfw.dimensionless_mass_within_circle(radius=integral_radius)

            assert dimensionless_mass_total == pytest.approx(dimensionless_mass_integral, 0.02)

        def test__elliptical_gnfw_profile__compare_to_grid2(self):

            nfw = mass_profiles.EllipticalGeneralizedNFW(kappa_s=2.0, axis_ratio=0.2, phi=0.0, inner_slope=1.8,
                                                         scale_radius=5.0)

            import math
            import numpy as np

            integral_radius = 1.0
            dimensionless_mass_total = 0.0

            xs = np.linspace(-1.5, 1.5, 40)
            ys = np.linspace(-1.5, 1.5, 40)

            edge = xs[1] - xs[0]
            area = edge ** 2

            for x in xs:
                for y in ys:

                    eta = math.sqrt(x ** 2 + y ** 2)

                    if eta < integral_radius:
                        dimensionless_mass_total += nfw.surface_density_func(eta) * area

            dimensionless_mass_integral = nfw.dimensionless_mass_within_circle(radius=integral_radius)

            assert dimensionless_mass_total == pytest.approx(dimensionless_mass_integral, 0.02)

    class TestWithinEllipse(object):

        def test__singular_isothermal_sphere__compare_circle_and_ellipse(self):

            sis = mass_profiles.SphericalIsothermal(einstein_radius=2.0)

            integral_radius = 2.0

            dimensionless_mass_integral_circle = sis.dimensionless_mass_within_circle(radius=integral_radius)

            dimensionless_mass_integral_ellipse = sis.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            assert dimensionless_mass_integral_circle == dimensionless_mass_integral_ellipse

        def test__singular_isothermal_ellipsoid__compare_circle_and_ellipse(self):

            sie = mass_profiles.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.5, phi=0.0)

            integral_radius = 2.0

            dimensionless_mass_integral_circle = sie.dimensionless_mass_within_circle(radius=integral_radius)

            dimensionless_mass_integral_ellipse = sie.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            assert dimensionless_mass_integral_circle == dimensionless_mass_integral_ellipse * 2.0

        def test__singular_isothermal_ellipsoid__compare_to_grid(self):

            import numpy as np

            sie = mass_profiles.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.5, phi=0.0)

            integral_radius = 0.5
            dimensionless_mass_tot = 0.0

            xs = np.linspace(-1.0, 1.0, 40)
            ys = np.linspace(-1.0, 1.0, 40)

            edge = xs[1] - xs[0]
            area = edge ** 2

            for x in xs:
                for y in ys:

                    eta = sie.grid_to_elliptical_radius(np.array([[x, y]]))

                    if eta < integral_radius:
                        dimensionless_mass_tot += sie.surface_density_func(eta) * area

            intensity_integral = sie.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            # Large errors required due to cusp at center of SIE - can get to errors of 0.01 for a 400 x 400 grid.
            assert dimensionless_mass_tot == pytest.approx(intensity_integral, 0.1)

        def test__singular_power_law_ellipsoid__compare_to_grid(self):

            import numpy as np

            sple = mass_profiles.EllipticalPowerLaw(einstein_radius=2.0, slope=1.5, axis_ratio=0.5, phi=0.0)

            integral_radius = 0.5
            dimensionless_mass_tot = 0.0

            xs = np.linspace(-1.0, 1.0, 40)
            ys = np.linspace(-1.0, 1.0, 40)

            edge = xs[1] - xs[0]
            area = edge ** 2

            for x in xs:
                for y in ys:

                    eta = sple.grid_to_elliptical_radius(np.array([[x, y]]))

                    if eta < integral_radius:
                        dimensionless_mass_tot += sple.surface_density_func(eta) * area

            intensity_integral = sple.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            assert dimensionless_mass_tot == pytest.approx(intensity_integral, 0.01)

        def test__cored_singular_power_law_ellipsoid__compare_to_grid(self):

            import numpy as np

            sple_core = mass_profiles.EllipticalCoredPowerLaw(einstein_radius=2.0, slope=1.8, core_radius=0.5,
                                                              axis_ratio=0.5, phi=0.0)

            integral_radius = 0.5
            dimensionless_mass_tot = 0.0

            xs = np.linspace(-1.0, 1.0, 40)
            ys = np.linspace(-1.0, 1.0, 40)

            edge = xs[1] - xs[0]
            area = edge ** 2

            for x in xs:
                for y in ys:

                    eta = sple_core.grid_to_elliptical_radius(np.array([[x, y]]))

                    if eta < integral_radius:
                        dimensionless_mass_tot += sple_core.surface_density_func(eta) * area

            intensity_integral = sple_core.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            assert dimensionless_mass_tot == pytest.approx(intensity_integral, 0.02)

        def test__elliptical_nfw__compare_to_grid(self):

            import numpy as np

            nfw = mass_profiles.EllipticalNFW(kappa_s=1.0, scale_radius=5.0, axis_ratio=0.5, phi=0.0)

            integral_radius = 0.5
            dimensionless_mass_tot = 0.0

            xs = np.linspace(-1.0, 1.0, 40)
            ys = np.linspace(-1.0, 1.0, 40)

            edge = xs[1] - xs[0]
            area = edge ** 2

            for x in xs:
                for y in ys:

                    eta = nfw.grid_to_elliptical_radius(np.array([[x, y]]))

                    if eta < integral_radius:
                        dimensionless_mass_tot += nfw.surface_density_func(eta) * area

            intensity_integral = nfw.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            assert dimensionless_mass_tot == pytest.approx(intensity_integral, 0.02)

        def test__generalized_elliptical_nfw__compare_to_grid(self):

            import numpy as np

            gnfw = mass_profiles.EllipticalGeneralizedNFW(kappa_s=1.0, scale_radius=5.0, inner_slope=1.0,
                                                          axis_ratio=0.5, phi=0.0)

            integral_radius = 0.5
            dimensionless_mass_tot = 0.0

            xs = np.linspace(-1.0, 1.0, 40)
            ys = np.linspace(-1.0, 1.0, 40)

            edge = xs[1] - xs[0]
            area = edge ** 2

            for x in xs:
                for y in ys:

                    eta = gnfw.grid_to_elliptical_radius(np.array([[x, y]]))

                    if eta < integral_radius:
                        dimensionless_mass_tot += gnfw.surface_density_func(eta) * area

            intensity_integral = gnfw.dimensionless_mass_within_ellipse(major_axis=integral_radius)

            assert dimensionless_mass_tot == pytest.approx(intensity_integral, 0.04)

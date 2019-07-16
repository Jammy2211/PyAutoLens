import numpy as np
import pytest
from astropy import cosmology as cosmo

from autolens import dimensions as dim
from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.model.profiles import mass_profiles as mp

from test.unit.mock.model import mock_cosmology

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestAbstractNFW(object):
    def test__coord_function_f__correct_values(self):
        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
        )

        # r > 1

        assert truncated_nfw.coord_func_f(
            grid_radius=np.array([2.0, 3.0])
        ) == pytest.approx(np.array([0.604599, 0.435209]), 1.0e-4)

        # r < 1
        assert truncated_nfw.coord_func_f(
            grid_radius=np.array([0.5, 1.0 / 3.0])
        ) == pytest.approx(1.52069, 1.86967, 1.0e-4)
        #
        # r == 1
        assert (
            truncated_nfw.coord_func_f(grid_radius=np.array([1.0, 1.0]))
            == np.array([1.0, 1.0])
        ).all()

    def test__coord_function_g__correct_values(self):
        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
        )

        # r > 1

        assert truncated_nfw.coord_func_g(
            grid_radius=np.array([2.0, 3.0])
        ) == pytest.approx(np.array([0.13180, 0.070598]), 1.0e-4)

        # r < 1

        assert truncated_nfw.coord_func_g(
            grid_radius=np.array([0.5, 1.0 / 3.0])
        ) == pytest.approx(np.array([0.69425, 0.97838]), 1.0e-4)

        # r == 1
        assert (
            truncated_nfw.coord_func_g(grid_radius=np.array([1.0, 1.0]))
            == np.array([1.0 / 3.0, 1.0 / 3.0])
        ).all()

    def test__coord_function_h__correct_values(self):
        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
        )

        assert truncated_nfw.coord_func_h(
            grid_radius=np.array([0.5, 3.0])
        ) == pytest.approx(np.array([0.134395, 0.840674]), 1.0e-4)

    def test__coord_function_k__correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=2.0
        )

        assert truncated_nfw.coord_func_k(
            grid_radius=np.array([2.0, 3.0])
        ) == pytest.approx(np.array([-0.09983408, -0.06661738]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=4.0
        )

        assert truncated_nfw.coord_func_k(
            grid_radius=np.array([2.0, 3.0])
        ) == pytest.approx(np.array([-0.19869011, -0.1329414]), 1.0e-4)

    def test__coord_function_l__correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=2.0
        )

        assert truncated_nfw.coord_func_l(
            grid_radius=np.array([2.0, 2.0])
        ) == pytest.approx(np.array([0.00080191, 0.00080191]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
        )

        assert truncated_nfw.coord_func_l(
            grid_radius=np.array([2.0, 2.0])
        ) == pytest.approx(np.array([0.00178711, 0.00178711]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
        )
        assert truncated_nfw.coord_func_l(
            grid_radius=np.array([3.0, 3.0])
        ) == pytest.approx(np.array([0.00044044, 0.00044044]), 1.0e-4)

    def test__coord_function_m__correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=2.0
        )

        assert truncated_nfw.coord_func_m(
            grid_radius=np.array([2.0, 2.0])
        ) == pytest.approx(np.array([0.0398826, 0.0398826]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
        )

        assert truncated_nfw.coord_func_m(
            grid_radius=np.array([2.0, 2.0])
        ) == pytest.approx(np.array([0.06726646, 0.06726646]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
        )

        assert truncated_nfw.coord_func_m(
            grid_radius=np.array([3.0, 3.0])
        ) == pytest.approx(np.array([0.06946888, 0.06946888]), 1.0e-4)

    def test__rho_at_scale_radius__numerical_values_in_default_units(self):
        cosmology = mock_cosmology.MockCosmology(
            kpc_per_arcsec=2.0, critical_surface_density=2.0
        )

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        rho = nfw.rho_at_scale_radius_for_units(
            redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology
        )
        assert rho == pytest.approx(1.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0)
        rho = nfw.rho_at_scale_radius_for_units(
            redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology
        )
        assert rho == pytest.approx(3.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0)
        rho = nfw.rho_at_scale_radius_for_units(
            redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology
        )
        assert rho == pytest.approx(0.25, 1e-3)

    def test__rho_at_scale_radius__unit_conversions(self):
        cosmology = mock_cosmology.MockCosmology(
            arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0
        )

        nfw_arcsec = mp.SphericalNFW(
            centre=(dim.Length(0.0, "arcsec"), dim.Length(0.0, "arcsec")),
            kappa_s=1.0,
            scale_radius=dim.Length(1.0, "arcsec"),
        )

        rho = nfw_arcsec.rho_at_scale_radius_for_units(
            unit_length="arcsec",
            unit_mass="solMass",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )
        assert rho == pytest.approx(2.0, 1e-3)

        # When converting to kpc, the critical surface density is divided by kpc_per_arcsec**2.0 = 2.0**2.0
        # The scale radius also becomes scale_radius*kpc_per_arcsec = 2.0

        rho = nfw_arcsec.rho_at_scale_radius_for_units(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )
        assert rho == pytest.approx(0.5 / 2.0, 1e-3)

        rho = nfw_arcsec.rho_at_scale_radius_for_units(
            unit_length="arcsec",
            unit_mass="angular",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )
        assert rho == pytest.approx(1.0, 1e-3)

        # This will make the critical surface density 4.0, with the same conversions as above

        cosmology = mock_cosmology.MockCosmology(
            arcsec_per_kpc=0.25, kpc_per_arcsec=4.0, critical_surface_density=2.0
        )

        rho = nfw_arcsec.rho_at_scale_radius_for_units(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )
        assert rho == pytest.approx(0.5 / 4.0, 1e-3)

        cosmology = mock_cosmology.MockCosmology(
            arcsec_per_kpc=0.25, kpc_per_arcsec=4.0, critical_surface_density=4.0
        )

        rho = nfw_arcsec.rho_at_scale_radius_for_units(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )
        assert rho == pytest.approx(0.25 / 4.0, 1e-3)

        nfw_kpc = mp.SphericalNFW(
            centre=(dim.Length(0.0, "kpc"), dim.Length(0.0, "kpc")),
            kappa_s=1.0,
            scale_radius=dim.Length(1.0, "kpc"),
        )

        cosmology = mock_cosmology.MockCosmology(
            arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0
        )
        rho = nfw_kpc.rho_at_scale_radius_for_units(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )
        assert rho == pytest.approx(0.5, 1e-3)

        rho = nfw_kpc.rho_at_scale_radius_for_units(
            unit_length="arcsec",
            unit_mass="solMass",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )
        assert rho == pytest.approx(2.0 / 0.5, 1e-3)

        rho = nfw_kpc.rho_at_scale_radius_for_units(
            unit_length="kpc",
            unit_mass="angular",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )
        assert rho == pytest.approx(1.0, 1e-3)

    def test__delta_concentration_value_in_default_units(self):
        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

        cosmology = mock_cosmology.MockCosmology(
            arcsec_per_kpc=1.0,
            kpc_per_arcsec=1.0,
            critical_surface_density=1.0,
            cosmic_average_density=1.0,
        )

        delta_concentration = nfw.delta_concentration_for_units(
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="solMass",
            cosmology=cosmology,
        )
        assert delta_concentration == pytest.approx(1.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0)
        delta_concentration = nfw.delta_concentration_for_units(
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="solMass",
            cosmology=cosmology,
        )
        assert delta_concentration == pytest.approx(3.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0)
        delta_concentration = nfw.delta_concentration_for_units(
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_mass="solMass",
            cosmology=cosmology,
        )
        assert delta_concentration == pytest.approx(0.25, 1e-3)

        # cosmology = mock_cosmology.MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0,
        #                           cosmic_average_density=1.0)
        #
        # nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        # delta_concentration = nfw.delta_concentration(unit_length='kpc', unit_mass='solMass', redshift_lens=0.5,
        #                                               redshift_source=1.0, cosmology=cosmology)
        # assert delta_concentration == pytest.approx(0.5, 1e-3)
        #
        # nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0)
        # delta_concentration = nfw.delta_concentration(unit_length='kpc', unit_mass='solMass', redshift_lens=0.5, redshift_source=1.0,
        #                                               cosmology=cosmology)
        # assert delta_concentration == pytest.approx(1.0, 1e-3)
        #
        # nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=20.0)
        # delta_concentration = nfw.delta_concentration(unit_length='kpc',  unit_mass='solMass', redshift_lens=0.5, redshift_source=1.0,
        #                                               cosmology=cosmology)
        # assert delta_concentration == pytest.approx(0.05, 1e-3)
        #
        # cosmology = mock_cosmology.MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0,
        #                           cosmic_average_density=2.0)
        #
        # nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        # delta_concentration = nfw.delta_concentration(unit_length='kpc',  unit_mass='solMass', redshift_lens=0.5, redshift_source=1.0,
        #                                               cosmology=cosmology)
        # assert delta_concentration == pytest.approx(0.25, 1e-3)

    def test__solve_concentration(self):
        cosmology = mock_cosmology.MockCosmology(
            arcsec_per_kpc=1.0,
            kpc_per_arcsec=1.0,
            critical_surface_density=1.0,
            cosmic_average_density=1.0,
        )

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

        concentration = nfw.concentration_for_units(
            redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology
        )

        assert concentration == pytest.approx(0.0074263, 1.0e-4)

    def test__radius_at_200__different_length_units_include_conversions(self):
        nfw_arcsec = mp.SphericalNFW(
            centre=(dim.Length(0.0, "arcsec"), dim.Length(0.0, "arcsec")),
            kappa_s=1.0,
            scale_radius=dim.Length(1.0, "arcsec"),
        )

        cosmology = mock_cosmology.MockCosmology(arcsec_per_kpc=0.2, kpc_per_arcsec=5.0)

        concentration = nfw_arcsec.concentration_for_units(
            cosmology=cosmology, redshift_profile=0.5, redshift_source=1.0
        )

        radius_200 = nfw_arcsec.radius_at_200_for_units(
            unit_length="arcsec",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )

        assert radius_200 == concentration * 1.0

        cosmology = mock_cosmology.MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0)

        concentration = nfw_arcsec.concentration_for_units(
            unit_length="kpc",
            cosmology=cosmology,
            redshift_profile=0.5,
            redshift_source=1.0,
        )

        radius_200 = nfw_arcsec.radius_at_200_for_units(
            unit_length="kpc",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )

        assert radius_200 == 2.0 * concentration * 1.0

        nfw_kpc = mp.SphericalNFW(
            centre=(dim.Length(0.0, "kpc"), dim.Length(0.0, "kpc")),
            kappa_s=1.0,
            scale_radius=dim.Length(1.0, "kpc"),
        )

        cosmology = mock_cosmology.MockCosmology(arcsec_per_kpc=0.2, kpc_per_arcsec=5.0)

        concentration = nfw_kpc.concentration_for_units(
            unit_length="kpc",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )

        radius_200 = nfw_kpc.radius_at_200_for_units(
            unit_length="kpc",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )

        assert radius_200 == concentration * 1.0

        cosmology = mock_cosmology.MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0)

        concentration = nfw_kpc.concentration_for_units(
            unit_length="arcsec",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )

        radius_200 = nfw_kpc.radius_at_200_for_units(
            unit_length="arcsec",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )

        assert radius_200 == concentration * 1.0 / 2.0

    def test__mass_at_200__unit_conversions_work(self):
        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

        cosmology = mock_cosmology.MockCosmology(
            arcsec_per_kpc=1.0,
            kpc_per_arcsec=1.0,
            critical_surface_density=1.0,
            cosmic_average_density=1.0,
        )

        radius_at_200 = nfw.radius_at_200_for_units(
            unit_length="arcsec",
            redshift_profile=0.5,
            redshift_source=1.0,
            cosmology=cosmology,
        )

        mass_at_200 = nfw.mass_at_200_for_units(
            cosmology=cosmology,
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_length="arcsec",
            unit_mass="solMass",
        )

        mass_calc = (
            200.0
            * ((4.0 / 3.0) * np.pi)
            * cosmology.cosmic_average_density
            * (radius_at_200 ** 3.0)
        )
        assert mass_at_200 == pytest.approx(mass_calc, 1.0e-5)

        # cosmology = mock_cosmology.MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0,
        #                           cosmic_average_density=1.0)
        #
        # radius_at_200 = nfw.radius_at_200_for_units(unit_length='arcsec', redshift_lens=0.5, redshift_source=1.0,
        #                                             cosmology=cosmology)
        #
        # mass_at_200 = nfw.mass_at_200(cosmology=cosmology, redshift_lens=0.5, redshift_source=1.0, unit_length='arcsec',
        #                               unit_mass='solMass')
        #
        # mass_calc = 200.0 * ((4.0 / 3.0) * np.pi) * cosmology.cosmic_average_density * (radius_at_200 ** 3.0)
        # assert mass_at_200 == pytest.approx(mass_calc, 1.0e-5)

    def test__values_of_quantities_for_real_cosmology__include_unit_conversions(self):

        cosmology = cosmo.LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7)

        nfw = mp.SphericalTruncatedNFW(
            kappa_s=0.5, scale_radius=5.0, truncation_radius=10.0
        )

        rho = nfw.rho_at_scale_radius_for_units(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            cosmology=cosmology,
        )

        delta_concentration = nfw.delta_concentration_for_units(
            redshift_profile=0.6,
            redshift_source=2.5,
            unit_length="kpc",
            unit_mass="solMass",
            redshift_of_cosmic_average_density="local",
            cosmology=cosmology,
        )

        concentration = nfw.concentration_for_units(
            redshift_profile=0.6,
            redshift_source=2.5,
            unit_length="kpc",
            unit_mass="solMass",
            redshift_of_cosmic_average_density="local",
            cosmology=cosmology,
        )

        radius_at_200 = nfw.radius_at_200_for_units(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            redshift_of_cosmic_average_density="local",
            cosmology=cosmology,
        )

        mass_at_200 = nfw.mass_at_200_for_units(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            redshift_of_cosmic_average_density="local",
            cosmology=cosmology,
        )

        mass_at_truncation_radius = nfw.mass_at_truncation_radius(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            redshift_of_cosmic_average_density="local",
            cosmology=cosmology,
        )

        assert rho == pytest.approx(29027857.01622403, 1.0e-4)
        assert delta_concentration == pytest.approx(213451.19421263796, 1.0e-4)
        assert concentration == pytest.approx(18.6605624462417, 1.0e-4)
        assert radius_at_200 == pytest.approx(623.7751567997697, 1.0e-4)
        assert mass_at_200 == pytest.approx(27651532986258.375, 1.0e-4)
        assert mass_at_truncation_radius == pytest.approx(14877085957074.299, 1.0e-4)

        rho = nfw.rho_at_scale_radius_for_units(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            cosmology=cosmology,
        )

        delta_concentration = nfw.delta_concentration_for_units(
            redshift_profile=0.6,
            redshift_source=2.5,
            unit_length="kpc",
            unit_mass="solMass",
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        concentration = nfw.concentration_for_units(
            redshift_profile=0.6,
            redshift_source=2.5,
            unit_length="kpc",
            unit_mass="solMass",
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        radius_at_200 = nfw.radius_at_200_for_units(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        mass_at_200 = nfw.mass_at_200_for_units(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        mass_at_truncation_radius = nfw.mass_at_truncation_radius(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        assert rho == pytest.approx(29027857.01622403, 1.0e-4)
        assert delta_concentration == pytest.approx(110665.28111397651, 1.0e-4)
        assert concentration == pytest.approx(14.401574489517804, 1.0e-4)
        assert radius_at_200 == pytest.approx(481.40801817963467, 1.0e-4)
        assert mass_at_200 == pytest.approx(24516707575366.09, 1.0e-4)
        assert mass_at_truncation_radius == pytest.approx(13190486262169.797, 1.0e-4)

        nfw = mp.SphericalTruncatedNFW(
            kappa_s=0.5,
            scale_radius=dim.Length(3.0, "kpc"),
            truncation_radius=dim.Length(7.0, "kpc"),
        )

        rho = nfw.rho_at_scale_radius_for_units(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            cosmology=cosmology,
        )

        delta_concentration = nfw.delta_concentration_for_units(
            redshift_profile=0.6,
            redshift_source=2.5,
            unit_length="kpc",
            unit_mass="solMass",
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        concentration = nfw.concentration_for_units(
            redshift_profile=0.6,
            redshift_source=2.5,
            unit_length="kpc",
            unit_mass="solMass",
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        radius_at_200 = nfw.radius_at_200_for_units(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        mass_at_200 = nfw.mass_at_200_for_units(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        mass_at_truncation_radius = nfw.mass_at_truncation_radius(
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        assert rho == pytest.approx(323442484.90222085, 1.0e-4)
        assert delta_concentration == pytest.approx(1233086.3244882922, 1.0e-4)
        assert concentration == pytest.approx(36.61521013005619, 1.0e-4)
        assert radius_at_200 == pytest.approx(109.84563039016857, 1.0e-4)
        assert mass_at_200 == pytest.approx(291253092446.923, 1.0e-4)
        assert mass_at_truncation_radius == pytest.approx(177609204745.61484, 1.0e-4)

        rho = nfw.rho_at_scale_radius_for_units(
            unit_length="arcsec",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            cosmology=cosmology,
        )

        delta_concentration = nfw.delta_concentration_for_units(
            redshift_profile=0.6,
            redshift_source=2.5,
            unit_length="arcsec",
            unit_mass="solMass",
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        concentration = nfw.concentration_for_units(
            redshift_profile=0.6,
            redshift_source=2.5,
            unit_length="arcsec",
            unit_mass="solMass",
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        radius_at_200 = nfw.radius_at_200_for_units(
            unit_length="arcsec",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        mass_at_200 = nfw.mass_at_200_for_units(
            unit_length="arcsec",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        mass_at_truncation_radius = nfw.mass_at_truncation_radius(
            unit_length="arcsec",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            redshift_of_cosmic_average_density="profile",
            cosmology=cosmology,
        )

        kpc_per_arcsec = 1.0 / cosmology.arcsec_per_kpc_proper(z=0.6).value

        assert rho == pytest.approx(323442484.90222085 * kpc_per_arcsec ** 3.0, 1.0e-4)
        assert delta_concentration == pytest.approx(1233086.3244882922, 1.0e-4)
        assert concentration == pytest.approx(36.61521013005619, 1.0e-4)
        assert radius_at_200 == pytest.approx(
            109.84563039016857 / kpc_per_arcsec, 1.0e-4
        )
        assert mass_at_200 == pytest.approx(291253092446.923, 1.0e-4)
        assert mass_at_truncation_radius == pytest.approx(177609204745.61484, 1.0e-4)


class TestGeneralizedNFW(object):
    def test__constructor_and_units(self):
        # gnfw = mp.EllipticalGeneralizedNFW(centre=(0.7, 1.0), axis_ratio=0.7, phi=45.0,
        #                                    kappa_s=2.0, inner_slope=1.5, scale_radius=10.0)
        #
        # assert gnfw.centre == (0.7, 1.0)
        # assert gnfw.axis_ratio == 0.7
        # assert gnfw.phi == 45.0
        # assert gnfw.kappa_s == 2.0
        # assert gnfw.inner_slope == 1.5
        # assert gnfw.scale_radius == 10.0

        gnfw = mp.SphericalGeneralizedNFW(
            centre=(1.0, 2.0), kappa_s=2.0, inner_slope=1.5, scale_radius=10.0
        )

        assert gnfw.centre == (1.0, 2.0)
        assert isinstance(gnfw.centre[0], dim.Length)
        assert isinstance(gnfw.centre[1], dim.Length)
        assert gnfw.centre[0].unit == "arcsec"
        assert gnfw.centre[1].unit == "arcsec"

        assert gnfw.axis_ratio == 1.0
        assert isinstance(gnfw.axis_ratio, float)

        assert gnfw.phi == 0.0
        assert isinstance(gnfw.phi, float)

        assert gnfw.kappa_s == 2.0
        assert isinstance(gnfw.kappa_s, float)

        assert gnfw.inner_slope == 1.5
        assert isinstance(gnfw.inner_slope, float)

        assert gnfw.scale_radius == 10.0
        assert isinstance(gnfw.scale_radius, dim.Length)
        assert gnfw.scale_radius.unit_length == "arcsec"

    # def test__coord_func_x_above_1(self):
    #     assert mp.EllipticalNFW.coord_func(2.0) == pytest.approx(0.60459, 1e-3)
    #
    #     assert mp.EllipticalNFW.coord_func(0.5) == pytest.approx(1.5206919, 1e-3)
    #
    #     assert mp.EllipticalNFW.coord_func(1.0) == 1.0

    def test__convergence_correct_values(self):
        gnfw = mp.SphericalGeneralizedNFW(
            centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0
        )
        assert gnfw.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == pytest.approx(
            0.30840, 1e-3
        )

        gnfw = mp.SphericalGeneralizedNFW(
            centre=(0.0, 0.0), kappa_s=2.0, inner_slope=1.5, scale_radius=1.0
        )
        assert gnfw.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == pytest.approx(
            0.30840 * 2, 1e-3
        )

        # gnfw = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, axis_ratio=0.5,
        #                                    phi=90.0, inner_slope=1.5, scale_radius=1.0)
        # assert gnfw.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.30840, 1e-3)
        #
        # gnfw = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=2.0, axis_ratio=0.5,
        #                                    phi=90.0, inner_slope=1.5, scale_radius=1.0)
        # assert gnfw.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.30840 * 2, 1e-3)

    def test__potential_correct_values(self):
        gnfw = mp.SphericalGeneralizedNFW(
            centre=(0.0, 0.0), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0
        )
        assert gnfw.potential_from_grid(
            grid=np.array([[0.1625, 0.1875]])
        ) == pytest.approx(0.00920, 1e-3)

        gnfw = mp.SphericalGeneralizedNFW(
            centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=8.0
        )
        assert gnfw.potential_from_grid(
            grid=np.array([[0.1625, 0.1875]])
        ) == pytest.approx(0.17448, 1e-3)

        # gnfw = mp.EllipticalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=5.0, axis_ratio=0.5,
        #                                    phi=100.0, inner_slope=1.0, scale_radius=10.0)
        # assert gnfw.potential_from_grid(grid=np.array([[2.0, 2.0]])) == pytest.approx(2.4718, 1e-4)

    def test__deflections_correct_values(self):
        gnfw = mp.SphericalGeneralizedNFW(
            centre=(0.0, 0.0), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0
        )
        deflections = gnfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(0.43501, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.37701, 1e-3)

        gnfw = mp.SphericalGeneralizedNFW(
            centre=(0.3, 0.2), kappa_s=2.5, inner_slope=1.5, scale_radius=4.0
        )
        deflections = gnfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(-9.31254, 1e-3)
        assert deflections[0, 1] == pytest.approx(-3.10418, 1e-3)

        # gnfw = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, axis_ratio=0.3,
        #                                    phi=100.0, inner_slope=0.5, scale_radius=8.0)
        # deflections = gnfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        # assert deflections[0, 0] == pytest.approx(0.26604, 1e-3)
        # assert deflections[0, 1] == pytest.approx(0.58988, 1e-3)
        #
        # gnfw = mp.EllipticalGeneralizedNFW(centre=(0.3, 0.2), kappa_s=2.5, axis_ratio=0.5,
        #                                    phi=100.0, inner_slope=1.5, scale_radius=4.0)
        # deflections = gnfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        # assert deflections[0, 0] == pytest.approx(-5.99032, 1e-3)
        # assert deflections[0, 1] == pytest.approx(-4.02541, 1e-3)

    def test__surfce_density__change_geometry(self):
        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(1.0, 1.0))
        assert gnfw_0.convergence_from_grid(
            grid=np.array([[1.0, 1.0]])
        ) == gnfw_1.convergence_from_grid(grid=np.array([[0.0, 0.0]]))

        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        assert gnfw_0.convergence_from_grid(
            grid=np.array([[1.0, 0.0]])
        ) == gnfw_1.convergence_from_grid(grid=np.array([[0.0, 1.0]]))

        # gnfw_0 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        # gnfw_1 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        # assert gnfw_0.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == gnfw_1.convergence_from_grid(
        #     grid=np.array([[0.0, 1.0]]))

    def test__potential__change_geometry(self):
        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(1.0, 1.0))
        assert gnfw_0.potential_from_grid(
            grid=np.array([[1.0, 1.0]])
        ) == gnfw_1.potential_from_grid(grid=np.array([[0.0, 0.0]]))

        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        assert gnfw_0.potential_from_grid(
            grid=np.array([[1.0, 0.0]])
        ) == gnfw_1.potential_from_grid(grid=np.array([[0.0, 1.0]]))

        # gnfw_0 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        # gnfw_1 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        # assert gnfw_0.potential_from_grid(grid=np.array([[1.0, 0.0]])) == gnfw_1.potential_from_grid(
        #     grid=np.array([[0.0, 1.0]]))

    def test__deflections__change_geometry(self):
        gnfw_0 = mp.SphericalGeneralizedNFW(
            centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0
        )
        gnfw_1 = mp.SphericalGeneralizedNFW(
            centre=(1.0, 1.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0
        )
        deflections_0 = gnfw_0.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        deflections_1 = gnfw_1.deflections_from_grid(grid=np.array([[0.0, 0.0]]))
        assert deflections_0[0, 0] == pytest.approx(-deflections_1[0, 0], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(-deflections_1[0, 1], 1e-5)

        gnfw_0 = mp.SphericalGeneralizedNFW(
            centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0
        )
        gnfw_1 = mp.SphericalGeneralizedNFW(
            centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0
        )
        deflections_0 = gnfw_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        deflections_1 = gnfw_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        assert deflections_0[0, 0] == pytest.approx(deflections_1[0, 1], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(deflections_1[0, 0], 1e-5)

        # gnfw_0 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, kappa_s=1.0,
        #                                      inner_slope=1.5, scale_radius=1.0)
        # gnfw_1 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, kappa_s=1.0,
        #                                      inner_slope=1.5, scale_radius=1.0)
        # deflections_0 = gnfw_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        # deflections_1 = gnfw_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        # assert deflections_0[0, 0] == pytest.approx(deflections_1[0, 1], 1e-5)
        # assert deflections_0[0, 1] == pytest.approx(deflections_1[0, 0], 1e-5)

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        gNFW = mp.SphericalGeneralizedNFW(
            centre=(-0.7, 0.5), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = gNFW.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5
        )

        interp_deflections_values = gNFW.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    # def test__compare_to_nfw(self):
    #     nfw = mp.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, kappa_s=1.0, scale_radius=5.0)
    #     gnfw = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, kappa_s=1.0,
    #                                        inner_slope=1.0, scale_radius=5.0)
    #
    #     assert nfw.potential_from_grid(grid) == pytest.approx(gnfw.potential_from_grid(grid), 1e-3)
    #     assert nfw.potential_from_grid(grid) == pytest.approx(gnfw.potential_from_grid(grid), 1e-3)
    #     assert nfw.deflections_from_grid(grid) == pytest.approx(gnfw.deflections_from_grid(grid), 1e-3)
    #     assert nfw.deflections_from_grid(grid) == pytest.approx(gnfw.deflections_from_grid(grid), 1e-3)

    # def test__spherical_and_elliptical_match(self):
    #     elliptical = mp.EllipticalGeneralizedNFW(centre=(0.1, 0.2), axis_ratio=1.0, phi=0.0, kappa_s=2.0,
    #                                              inner_slope=1.5, scale_radius=3.0)
    #     spherical = mp.SphericalGeneralizedNFW(centre=(0.1, 0.2), kappa_s=2.0, inner_slope=1.5, scale_radius=3.0)
    #
    #     assert elliptical.convergence_from_grid(grid) == pytest.approx(spherical.convergence_from_grid(grid),
    #                                                                        1e-4)
    #     assert elliptical.potential_from_grid(grid) == pytest.approx(spherical.potential_from_grid(grid), 1e-4)
    #     assert elliptical.deflections_from_grid(grid) == pytest.approx(spherical.deflections_from_grid(grid), 1e-4)

    def test__reshape_decorators(self):

        regular_grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=1.0
        )

        # gnfw = mp.EllipticalGeneralizedNFW()
        #
        # convergence = gnfw.convergence_from_grid(
        #     grid=regular_grid, return_in_2d=True, return_binned=False)
        #
        # assert convergence.shape == (2, 2)
        #
        # potential = gnfw.potential_from_grid(
        #     grid=regular_grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)
        #
        # deflections = gnfw.deflections_from_grid(
        #     grid=regular_grid, return_in_2d=True, return_binned=False)
        #
        # assert deflections.shape == (2, 2, 2)

        gnfw = mp.SphericalGeneralizedNFW()

        convergence = gnfw.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        potential = gnfw.potential_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert potential.shape == (2, 2)

        deflections = gnfw.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)


class TestTruncatedNFW(object):
    def test__constructor_and_units(self):

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(1.0, 2.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=2.0
        )

        assert truncated_nfw.centre == (1.0, 2.0)
        assert isinstance(truncated_nfw.centre[0], dim.Length)
        assert isinstance(truncated_nfw.centre[1], dim.Length)
        assert truncated_nfw.centre[0].unit == "arcsec"
        assert truncated_nfw.centre[1].unit == "arcsec"

        assert truncated_nfw.axis_ratio == 1.0
        assert isinstance(truncated_nfw.axis_ratio, float)

        assert truncated_nfw.phi == 0.0
        assert isinstance(truncated_nfw.phi, float)

        assert truncated_nfw.kappa_s == 2.0
        assert isinstance(truncated_nfw.kappa_s, float)

        assert truncated_nfw.inner_slope == 1.0
        assert isinstance(truncated_nfw.inner_slope, float)

        assert truncated_nfw.scale_radius == 10.0
        assert isinstance(truncated_nfw.scale_radius, dim.Length)
        assert truncated_nfw.scale_radius.unit_length == "arcsec"

        assert truncated_nfw.truncation_radius == 2.0
        assert isinstance(truncated_nfw.truncation_radius, dim.Length)
        assert truncated_nfw.truncation_radius.unit_length == "arcsec"

    def test__convergence_correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0, truncation_radius=2.0
        )

        assert truncated_nfw.convergence_from_grid(
            grid=np.array([[2.0, 0.0]])
        ) == pytest.approx(2.0 * 0.046409642, 1.0e-4)

        assert truncated_nfw.convergence_from_grid(
            grid=np.array([[1.0, 1.0]])
        ) == pytest.approx(2.0 * 0.10549515, 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0, truncation_radius=2.0
        )

        assert truncated_nfw.convergence_from_grid(
            grid=np.array([[2.0, 0.0]])
        ) == pytest.approx(6.0 * 0.046409642, 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=3.0, scale_radius=5.0, truncation_radius=2.0
        )

        assert truncated_nfw.convergence_from_grid(
            grid=np.array([[2.0, 0.0]])
        ) == pytest.approx(1.51047026, 1.0e-4)

    def test__deflections_correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0, truncation_radius=2.0
        )

        # factor = (4.0 * kappa_s * scale_radius / (r / scale_radius))

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        factor = (4.0 * 1.0 * 1.0) / (2.0 / 1.0)
        assert deflections[0, 0] == pytest.approx(factor * 0.38209715, 1.0e-4)
        assert deflections[0, 1] == pytest.approx(0.0, 1.0e-4)

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[0.0, 2.0]]))

        assert deflections[0, 0] == pytest.approx(0.0, 1.0e-4)
        assert deflections[0, 1] == pytest.approx(factor * 0.38209715, 1.0e-4)

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

        factor = (4.0 * 1.0 * 1.0) / (np.sqrt(2) / 1.0)
        assert deflections[0, 0] == pytest.approx(
            (1.0 / np.sqrt(2)) * factor * 0.3125838, 1.0e-4
        )
        assert deflections[0, 1] == pytest.approx(
            (1.0 / np.sqrt(2)) * factor * 0.3125838, 1.0e-4
        )

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0, truncation_radius=2.0
        )

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        factor = (4.0 * 2.0 * 1.0) / (2.0 / 1.0)
        assert deflections[0, 0] == pytest.approx(factor * 0.38209715, 1.0e-4)
        assert deflections[0, 1] == pytest.approx(0.0, 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0, truncation_radius=2.0
        )

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        assert deflections[0, 0] == pytest.approx(2.1702661386, 1.0e-4)
        assert deflections[0, 1] == pytest.approx(0.0, 1.0e-4)

    def test__compare_nfw_and_truncated_nfw_with_large_truncation_radius__convergence_and_deflections_identical(
        self
    ):

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0, truncation_radius=50000.0
        )

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0)

        truncated_nfw_convergence = truncated_nfw.convergence_from_grid(
            grid=np.array([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
        )
        nfw_convergence = nfw.convergence_from_grid(
            grid=np.array([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
        )

        assert truncated_nfw_convergence == pytest.approx(nfw_convergence, 1.0e-4)

        truncated_nfw_deflections = truncated_nfw.deflections_from_grid(
            grid=np.array([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
        )
        nfw_deflections = nfw.deflections_from_grid(
            grid=np.array([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
        )

        assert truncated_nfw_deflections == pytest.approx(nfw_deflections, 1.0e-4)

    def test__deflections_of_spherical_profile__dont_use_interpolate_and_cache_decorators(
        self
    ):

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(-0.7, 0.5), kappa_s=1.0, scale_radius=8.0, truncation_radius=2.0
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = truncated_nfw.deflections_from_grid(
            grid=regular_with_interp
        )

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5
        )

        interp_deflections_values = truncated_nfw.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y != interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:, 1]).all()

    def test__mass_at_truncation_radius__values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(
            centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0, truncation_radius=1.0
        )

        cosmology = mock_cosmology.MockCosmology(
            arcsec_per_kpc=1.0,
            kpc_per_arcsec=1.0,
            critical_surface_density=1.0,
            cosmic_average_density=1.0,
        )

        mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(
            redshift_profile=0.5,
            redshift_source=1.0,
            unit_length="arcsec",
            unit_mass="solMass",
            cosmology=cosmology,
        )

        assert mass_at_truncation_radius == pytest.approx(0.00009792581, 1.0e-5)

        # truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0,
        #                                          truncation_radius=1.0)
        #
        # cosmology = mock_cosmology.MockCosmology(arcsec_per_kpc=1.0, kpc_per_arcsec=1.0, critical_surface_density=2.0,
        #                           cosmic_average_density=3.0)
        #
        # mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(redshift_lens=0.5, redshift_source=1.0,
        #     unit_length='arcsec', unit_mass='solMass', cosmology=cosmology)
        #
        # assert mass_at_truncation_radius == pytest.approx(0.00008789978, 1.0e-5)
        #
        # truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=2.0,
        #                                          truncation_radius=1.0)
        #
        # mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(redshift_lens=0.5, redshift_source=1.0,
        #     unit_length='arcsec', unit_mass='solMass', cosmology=cosmology)
        #
        # assert mass_at_truncation_radius == pytest.approx(0.0000418378, 1.0e-5)
        #
        # truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=8.0,
        #                                          truncation_radius=4.0)
        #
        # mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(redshift_lens=0.5, redshift_source=1.0,
        #     unit_length='arcsec', unit_mass='solMass', cosmology=cosmology)
        #
        # assert mass_at_truncation_radius == pytest.approx(0.0000421512, 1.0e-4)
        #
        # truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=8.0,
        #                                          truncation_radius=4.0)
        #
        # mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(redshift_lens=0.5, redshift_source=1.0,
        #     unit_length='arcsec', unit_mass='solMass', cosmology=cosmology)
        #
        # assert mass_at_truncation_radius == pytest.approx(0.00033636625, 1.0e-4)

    def test_summarize_in_units(self):

        cosmology = cosmo.LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7)

        nfw = mp.SphericalTruncatedNFW(
            kappa_s=0.5, scale_radius=5.0, truncation_radius=10.0
        )

        summary_text = nfw.summarize_in_units(
            radii=[dim.Length(10.0), dim.Length(500.0)],
            prefix="nfw_",
            unit_length="kpc",
            unit_mass="solMass",
            redshift_profile=0.6,
            redshift_source=2.5,
            redshift_of_cosmic_average_density="profile",
            whitespace=50,
            cosmology=cosmology,
        )

        i = 0

        assert summary_text[i] == "Mass Profile = SphericalTruncatedNFW\n"
        i += 1
        assert (
            summary_text[i]
            == "nfw_einstein_radius                               15.38 kpc"
        )
        i += 1
        assert (
            summary_text[i]
            == "nfw_einstein_mass                                 1.4418e+12 solMass"
        )
        i += 1
        assert (
            summary_text[i]
            == "nfw_mass_within_10.00_kpc                         5.2061e+12 solMass"
        )
        i += 1
        assert (
            summary_text[i]
            == "nfw_mass_within_500.00_kpc                        7.3287e+12 solMass"
        )
        i += 1
        assert (
            summary_text[i]
            == "nfw_rho_at_scale_radius                           29027857.02 solMass/kpc3"
        )
        i += 1
        assert (
            summary_text[i]
            == "nfw_delta_concentration                           110665.28"
        )
        i += 1
        assert (
            summary_text[i] == "nfw_concentration                                 14.40"
        )
        i += 1
        assert (
            summary_text[i]
            == "nfw_radius_at_200x_cosmic_density                 481.41 kpc"
        )
        i += 1
        assert (
            summary_text[i]
            == "nfw_mass_at_200x_cosmic_density                   2.4517e+13 solMass"
        )
        i += 1
        assert (
            summary_text[i]
            == "nfw_mass_at_truncation_radius                     1.3190e+13 solMass"
        )
        i += 1

    def test__reshape_decorators(self):

        regular_grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=1.0
        )

        truncated_nfw = mp.SphericalTruncatedNFW()

        convergence = truncated_nfw.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        potential = truncated_nfw.potential_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert potential.shape == (2, 2)

        deflections = truncated_nfw.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)


class TestTruncatedNFWChallenge(object):
    def test__constructor_and_units(self):

        truncated_nfw = mp.SphericalTruncatedNFWChallenge(
            centre=(1.0, 2.0), kappa_s=2.0, scale_radius=10.0
        )

        assert truncated_nfw.centre == (1.0, 2.0)
        assert isinstance(truncated_nfw.centre[0], dim.Length)
        assert isinstance(truncated_nfw.centre[1], dim.Length)
        assert truncated_nfw.centre[0].unit == "arcsec"
        assert truncated_nfw.centre[1].unit == "arcsec"

        assert truncated_nfw.axis_ratio == 1.0
        assert isinstance(truncated_nfw.axis_ratio, float)

        assert truncated_nfw.phi == 0.0
        assert isinstance(truncated_nfw.phi, float)

        assert truncated_nfw.kappa_s == 2.0
        assert isinstance(truncated_nfw.kappa_s, float)

        assert truncated_nfw.inner_slope == 1.0
        assert isinstance(truncated_nfw.inner_slope, float)

        assert truncated_nfw.scale_radius == 10.0
        assert isinstance(truncated_nfw.scale_radius, dim.Length)
        assert truncated_nfw.scale_radius.unit_length == "arcsec"

        assert truncated_nfw.truncation_radius == pytest.approx(
            2.0 * 189.26967095554755, 1.0e-4
        )
        assert isinstance(truncated_nfw.truncation_radius, dim.Length)
        assert truncated_nfw.truncation_radius.unit_length == "arcsec"


class TestNFW(object):
    def test__constructor_and_units(self):

        nfw = mp.EllipticalNFW(
            centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, kappa_s=2.0, scale_radius=10.0
        )

        assert nfw.centre == (1.0, 2.0)
        assert isinstance(nfw.centre[0], dim.Length)
        assert isinstance(nfw.centre[1], dim.Length)
        assert nfw.centre[0].unit == "arcsec"
        assert nfw.centre[1].unit == "arcsec"

        assert nfw.axis_ratio == 0.5
        assert isinstance(nfw.axis_ratio, float)

        assert nfw.phi == 45.0
        assert isinstance(nfw.phi, float)

        assert nfw.kappa_s == 2.0
        assert isinstance(nfw.kappa_s, float)

        assert nfw.inner_slope == 1.0
        assert isinstance(nfw.inner_slope, float)

        assert nfw.scale_radius == 10.0
        assert isinstance(nfw.scale_radius, dim.Length)
        assert nfw.scale_radius.unit_length == "arcsec"

        nfw = mp.SphericalNFW(centre=(1.0, 2.0), kappa_s=2.0, scale_radius=10.0)

        assert nfw.centre == (1.0, 2.0)
        assert isinstance(nfw.centre[0], dim.Length)
        assert isinstance(nfw.centre[1], dim.Length)
        assert nfw.centre[0].unit == "arcsec"
        assert nfw.centre[1].unit == "arcsec"

        assert nfw.axis_ratio == 1.0
        assert isinstance(nfw.axis_ratio, float)

        assert nfw.phi == 0.0
        assert isinstance(nfw.phi, float)

        assert nfw.kappa_s == 2.0
        assert isinstance(nfw.kappa_s, float)

        assert nfw.inner_slope == 1.0
        assert isinstance(nfw.inner_slope, float)

        assert nfw.scale_radius == 10.0
        assert isinstance(nfw.scale_radius, dim.Length)
        assert nfw.scale_radius.unit_length == "arcsec"

    def test__convergence_correct_values(self):
        # r = 2.0 (> 1.0)
        # F(r) = (1/(sqrt(3))*atan(sqrt(3)) = 0.60459978807
        # kappa(r) = 2 * kappa_s * (1 - 0.60459978807) / (4-1) = 0.263600141
        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        assert nfw.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == pytest.approx(
            0.263600141, 1e-3
        )

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        assert nfw.convergence_from_grid(grid=np.array([[0.5, 0.0]])) == pytest.approx(
            1.388511, 1e-3
        )

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0)
        assert nfw.convergence_from_grid(grid=np.array([[0.5, 0.0]])) == pytest.approx(
            2.0 * 1.388511, 1e-3
        )

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=2.0)
        assert nfw.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(
            1.388511, 1e-3
        )

        nfw = mp.EllipticalNFW(
            centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, kappa_s=1.0, scale_radius=1.0
        )
        assert nfw.convergence_from_grid(grid=np.array([[0.25, 0.0]])) == pytest.approx(
            1.388511, 1e-3
        )

    def test__potential_correct_values(self):
        nfw = mp.SphericalNFW(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)
        assert nfw.potential_from_grid(
            grid=np.array([[0.1875, 0.1625]])
        ) == pytest.approx(0.03702, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)
        assert nfw.potential_from_grid(
            grid=np.array([[0.1875, 0.1625]])
        ) == pytest.approx(0.03702, 1e-3)

        nfw = mp.EllipticalNFW(
            centre=(0.3, 0.2), axis_ratio=0.7, phi=6.0, kappa_s=2.5, scale_radius=4.0
        )
        assert nfw.potential_from_grid(
            grid=np.array([[0.1625, 0.1625]])
        ) == pytest.approx(0.05380, 1e-3)

    def test__potential__spherical_and_elliptical_are_same(self):
        nfw_spherical = mp.SphericalNFW(
            centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0
        )
        nfw_elliptical = mp.EllipticalNFW(
            centre=(0.3, 0.2), axis_ratio=1.0, phi=0.0, kappa_s=2.5, scale_radius=4.0
        )

        potential_spherical = nfw_spherical.potential_from_grid(
            grid=np.array([[0.1875, 0.1625]])
        )
        potential_elliptical = nfw_elliptical.potential_from_grid(
            grid=np.array([[0.1875, 0.1625]])
        )

        assert potential_spherical == pytest.approx(potential_elliptical, 1e-3)

        potential_spherical = nfw_spherical.potential_from_grid(
            grid=np.array([[50.0, 50.0]])
        )
        potential_elliptical = nfw_elliptical.potential_from_grid(
            grid=np.array([[50.0, 50.0]])
        )

        assert potential_spherical == pytest.approx(potential_elliptical, 1e-3)

        potential_spherical = nfw_spherical.potential_from_grid(
            grid=np.array([[-50.0, -50.0]])
        )
        potential_elliptical = nfw_elliptical.potential_from_grid(
            grid=np.array([[-50.0, -50.0]])
        )

        assert potential_spherical == pytest.approx(potential_elliptical, 1e-3)

    def test__deflections_correct_values(self):
        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        deflections = nfw.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(0.56194, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.56194, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)
        deflections = nfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(-2.08909, 1e-3)
        assert deflections[0, 1] == pytest.approx(-0.69636, 1e-3)

        nfw = mp.EllipticalNFW(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, kappa_s=1.0, scale_radius=1.0
        )
        deflections = nfw.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(0.56194, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.56194, 1e-3)

        nfw = mp.EllipticalNFW(
            centre=(0.3, 0.2), axis_ratio=0.7, phi=6.0, kappa_s=2.5, scale_radius=4.0
        )
        deflections = nfw.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(-2.59480, 1e-3)
        assert deflections[0, 1] == pytest.approx(-0.44204, 1e-3)

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        nfw = mp.EllipticalNFW(
            centre=(-0.7, 0.5), axis_ratio=0.9, phi=45.0, kappa_s=1.0, scale_radius=8.0
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = nfw.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5
        )

        interp_deflections_values = nfw.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__dont_use_interpolate_and_cache_decorators(
        self
    ):
        nfw = mp.SphericalNFW(centre=(-0.7, 0.5), kappa_s=1.0, scale_radius=8.0)

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = nfw.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5
        )

        interp_deflections_values = nfw.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y != interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:, 1]).all()

    def test__reshape_decorators(self):

        regular_grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=1.0
        )

        nfw = mp.EllipticalNFW()

        convergence = nfw.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        potential = nfw.potential_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert potential.shape == (2, 2)

        deflections = nfw.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)

        nfw = mp.SphericalNFW()

        convergence = nfw.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        potential = nfw.potential_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert potential.shape == (2, 2)

        deflections = nfw.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)

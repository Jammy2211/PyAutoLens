from autolens.model import cosmology_util
from astropy import cosmology as cosmo

import pytest

planck = cosmo.Planck15

class TestCosmology:

    def test__arcsec_to_kpc_conversion(self):

        arcsec_per_kpc_proper = cosmology_util.arcsec_per_kpc_proper_from_redshift_and_cosmology(
            redshift=0.1, cosmology=planck)

        assert arcsec_per_kpc_proper.value == pytest.approx(0.525060, 1e-5)

        kpc_per_arcsec_proper = cosmology_util.kpc_per_arcsec_proper_from_redshift_and_cosmology(
            redshift=0.1, cosmology=planck)

        assert kpc_per_arcsec_proper.value == pytest.approx(1.904544, 1e-5)

        arcsec_per_kpc_proper = cosmology_util.arcsec_per_kpc_proper_from_redshift_and_cosmology(
            redshift=1.0, cosmology=planck)

        assert arcsec_per_kpc_proper.value == pytest.approx(0.1214785, 1e-5)

        kpc_per_arcsec_proper = cosmology_util.kpc_per_arcsec_proper_from_redshift_and_cosmology(
            redshift=1.0, cosmology=planck)

        assert kpc_per_arcsec_proper.value == pytest.approx(8.231907, 1e-5)

    def test__angular_diameter_distances(self):

        angular_diameter_distance_to_earth_kpc = \
            cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
                redshift=0.1, cosmology=planck, units_distance='kpc')
            
        assert angular_diameter_distance_to_earth_kpc.value == pytest.approx(392840, 1e-5)

        angular_diameter_distance_to_earth_arcsec = \
            cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
                redshift=0.1, cosmology=planck, units_distance='arcsec')

        arcsec_per_kpc = cosmology_util.arcsec_per_kpc_proper_from_redshift_and_cosmology(redshift=0.1,
                                                                                          cosmology=planck)

        assert (arcsec_per_kpc * angular_diameter_distance_to_earth_kpc).value == \
                pytest.approx(angular_diameter_distance_to_earth_arcsec.value, 1e-5)

        angular_diameter_distance_between_redshifts_kpc = \
            cosmology_util.angular_diameter_distance_between_redshifts_from_redshifts_and_cosmlology(
                redshift_0=0.1, redshift_1=1.0, cosmology=planck, units_distance='kpc')

        assert angular_diameter_distance_between_redshifts_kpc.value == pytest.approx(1481890.4, 1e-5)

    def test__critical_surface_mass_densities(self):

        critical_surface_mass_density = \
            cosmology_util.critical_surface_mass_density_between_redshifts_from_redshifts_and_cosmology(
            redshift_0=0.1, redshift_1=1.0, cosmology=planck, units_distance='kpc')

        assert critical_surface_mass_density.value == pytest.approx(4.85e9, 1e-2)

        critical_surface_mass_density = \
            cosmology_util.critical_surface_mass_density_between_redshifts_from_redshifts_and_cosmology(
            redshift_0=0.1, redshift_1=1.0, cosmology=planck, units_mass='solMass', units_distance='arcsec')

        assert critical_surface_mass_density.value == pytest.approx(17593241668, 1e-2)

    def test__cosmic_average_mass_density(self):

        cosmic_average_mass_density = cosmology_util.cosmic_average_mass_density_from_redshift_and_cosmology(
            redshift=0.6, cosmology=planck, units_mass='solMass', units_distance='kpc')

        assert cosmic_average_mass_density.value == pytest.approx(249.20874, 1.0e-4)

        cosmic_average_mass_density = cosmology_util.cosmic_average_mass_density_from_redshift_and_cosmology(
            redshift=0.6, cosmology=planck, units_mass='solMass', units_distance='arcsec')

        assert cosmic_average_mass_density.value == pytest.approx(81280.09116133313, 1.0e-4)
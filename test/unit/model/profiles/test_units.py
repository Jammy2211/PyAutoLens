import pytest

from autolens import exc
from autolens.model.profiles import units


class TestFloatDistance(object):

    def test__conversions_from_arcsec_to_kpc_and_back__errors_raised_if_no_kpc_per_arcsec(self):
        unit_arcsec = units.Distance(value=2.0)

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == 'arcsec'

        unit_arcsec = unit_arcsec.convert(unit_distance='arcsec')

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == 'arcsec'

        unit_kpc = unit_arcsec.convert(unit_distance='kpc', kpc_per_arcsec=2.0)

        assert unit_kpc == 4.0
        assert unit_kpc.unit == 'kpc'

        unit_kpc = unit_kpc.convert(unit_distance='kpc')

        assert unit_kpc == 4.0
        assert unit_kpc.unit == 'kpc'

        unit_arcsec = unit_kpc.convert(unit_distance='arcsec', kpc_per_arcsec=2.0)

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == 'arcsec'

        with pytest.raises(exc.UnitsException):
            unit_arcsec.convert(unit_distance='kpc')
            unit_kpc.convert(unit_distance='arcsec')
            unit_arcsec.convert(unit_distance='lol')


class TestFloatLuminosity(object):

    def test__conversions_from_electrons_per_second_and_counts_and_back__errors_raised_if_no_exposure_time(self):
        unit_electrons_per_second = units.Luminosity(value=2.0)

        assert unit_electrons_per_second == 2.0
        assert unit_electrons_per_second.unit == 'electrons_per_second'

        unit_electrons_per_second = unit_electrons_per_second.convert(unit_luminosity='electrons_per_second')

        assert unit_electrons_per_second == 2.0
        assert unit_electrons_per_second.unit == 'electrons_per_second'

        unit_counts = unit_electrons_per_second.convert(unit_luminosity='counts', exposure_time=2.0)

        assert unit_counts == 4.0
        assert unit_counts.unit == 'counts'

        unit_counts = unit_counts.convert(unit_luminosity='counts')

        assert unit_counts == 4.0
        assert unit_counts.unit == 'counts'

        unit_electrons_per_second = unit_counts.convert(unit_luminosity='electrons_per_second', exposure_time=2.0)

        assert unit_electrons_per_second == 2.0
        assert unit_electrons_per_second.unit == 'electrons_per_second'

        with pytest.raises(exc.UnitsException):
            unit_electrons_per_second.convert(unit_luminosity='counts')
            unit_counts.convert(unit_luminosity='electrons_per_second')
            unit_electrons_per_second.convert(unit_luminosity='lol')


class TestFloatMass(object):

    def test__conversions_from_angular_and_sol_mass_and_back__errors_raised_if_no_exposure_time(self):
        unit_angular = units.Mass(value=2.0)

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular'

        unit_angular = unit_angular.convert(unit_mass='angular')

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular'

        unit_sol_mass = unit_angular.convert(unit_mass='solMass', critical_surface_mass_density=2.0)

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == 'solMass'

        unit_sol_mass = unit_sol_mass.convert(unit_mass='solMass')

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == 'solMass'

        unit_angular = unit_sol_mass.convert(unit_mass='angular', critical_surface_mass_density=2.0)

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular'

        with pytest.raises(exc.UnitsException):
            unit_angular.convert(unit_mass='solMass')
            unit_sol_mass.convert(unit_mass='angular')
            unit_angular.convert(unit_mass='lol')

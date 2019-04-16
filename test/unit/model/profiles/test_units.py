from autolens import exc
from autolens.model.profiles import units

import pytest


class TestFloatNone(object):

    def test__constructor_and_convert(self):

        unit_none = units.FloatNone(value=2.0)

        assert unit_none == 2.0
        assert unit_none.unit_type == None
        assert unit_none.unit == None

        with pytest.raises(exc.UnitsException):
            unit_none.convert()


class TestFloatDistance(object):

    def test__conversions_from_arcsec_to_kpc_and_back__errors_raised_if_no_kpc_per_arcsec(self):

        unit_arcsec = units.FloatDistance(value=2.0, unit_distance='arcsec')

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit_type == 'distance'
        assert unit_arcsec.unit_distance == 'arcsec'
        assert unit_arcsec.unit == 'arcsec'

        unit_arcsec = unit_arcsec.convert(unit_distance='arcsec')

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit_type == 'distance'
        assert unit_arcsec.unit_distance == 'arcsec'
        assert unit_arcsec.unit == 'arcsec'

        unit_kpc = unit_arcsec.convert(unit_distance='kpc', kpc_per_arcsec=2.0)

        assert unit_kpc == 4.0
        assert unit_kpc.unit_type == 'distance'
        assert unit_kpc.unit_distance == 'kpc'
        assert unit_kpc.unit == 'kpc'

        unit_kpc = unit_kpc.convert(unit_distance='kpc')

        assert unit_kpc == 4.0
        assert unit_kpc.unit_type == 'distance'
        assert unit_kpc.unit_distance == 'kpc'
        assert unit_kpc.unit == 'kpc'

        unit_arcsec = unit_kpc.convert(unit_distance='arcsec', kpc_per_arcsec=2.0)

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit_type == 'distance'
        assert unit_arcsec.unit_distance == 'arcsec'
        assert unit_arcsec.unit == 'arcsec'

        with pytest.raises(exc.UnitsException):
            unit_arcsec.convert(unit_distance='kpc')
            unit_kpc.convert(unit_distance='arcsec')
            unit_arcsec.convert(unit_distance='lol')


class TestFloatLuminosity(object):

    def test__conversions_from_electrons_per_second_and_counts_and_back__errors_raised_if_no_exposure_time(self):

        unit_electrons_per_second = units.FloatLuminosity(value=2.0, unit_luminosity='electrons_per_second')

        assert unit_electrons_per_second == 2.0
        assert unit_electrons_per_second.unit_type == 'luminosity'
        assert unit_electrons_per_second.unit_luminosity == 'electrons_per_second'
        assert unit_electrons_per_second.unit == 'electrons_per_second'

        unit_electrons_per_second = unit_electrons_per_second.convert(unit_luminosity='electrons_per_second')

        assert unit_electrons_per_second == 2.0
        assert unit_electrons_per_second.unit_type == 'luminosity'
        assert unit_electrons_per_second.unit_luminosity == 'electrons_per_second'
        assert unit_electrons_per_second.unit == 'electrons_per_second'

        unit_counts = unit_electrons_per_second.convert(unit_luminosity='counts', exposure_time=2.0)

        assert unit_counts == 4.0
        assert unit_counts.unit_type == 'luminosity'
        assert unit_counts.unit_luminosity == 'counts'
        assert unit_counts.unit == 'counts'

        unit_counts = unit_counts.convert(unit_luminosity='counts')

        assert unit_counts == 4.0
        assert unit_counts.unit_type == 'luminosity'
        assert unit_counts.unit_luminosity == 'counts'
        assert unit_counts.unit == 'counts'

        unit_electrons_per_second = unit_counts.convert(unit_luminosity='electrons_per_second', exposure_time=2.0)

        assert unit_electrons_per_second == 2.0
        assert unit_electrons_per_second.unit_type == 'luminosity'
        assert unit_electrons_per_second.unit_luminosity == 'electrons_per_second'
        assert unit_electrons_per_second.unit == 'electrons_per_second'

        with pytest.raises(exc.UnitsException):
            unit_electrons_per_second.convert(unit_luminosity='counts')
            unit_counts.convert(unit_luminosity='electrons_per_second')
            unit_electrons_per_second.convert(unit_luminosity='lol')


class TestFloatMass(object):

    def test__conversions_from_angular_and_solMass_and_back__errors_raised_if_no_exposure_time(self):

        unit_angular = units.FloatMass(value=2.0, unit_mass='angular')

        assert unit_angular == 2.0
        assert unit_angular.unit_type == 'mass'
        assert unit_angular.unit_mass == 'angular'
        assert unit_angular.unit == 'angular'

        unit_angular = unit_angular.convert(unit_mass='angular')

        assert unit_angular == 2.0
        assert unit_angular.unit_type == 'mass'
        assert unit_angular.unit_mass == 'angular'
        assert unit_angular.unit == 'angular'

        unit_solMass = unit_angular.convert(unit_mass='solMass', critical_surface_mass_density=2.0)

        assert unit_solMass == 4.0
        assert unit_solMass.unit_type == 'mass'
        assert unit_solMass.unit_mass == 'solMass'
        assert unit_solMass.unit == 'solMass'

        unit_solMass = unit_solMass.convert(unit_mass='solMass')

        assert unit_solMass == 4.0
        assert unit_solMass.unit_type == 'mass'
        assert unit_solMass.unit_mass == 'solMass'
        assert unit_solMass.unit == 'solMass'

        unit_angular = unit_solMass.convert(unit_mass='angular', critical_surface_mass_density=2.0)

        assert unit_angular == 2.0
        assert unit_angular.unit_type == 'mass'
        assert unit_angular.unit_mass == 'angular'
        assert unit_angular.unit == 'angular'

        with pytest.raises(exc.UnitsException):
            unit_angular.convert(unit_mass='solMass')
            unit_solMass.convert(unit_mass='angular')
            unit_angular.convert(unit_mass='lol')


class TestFloatMassOverLuminosity(object):

    def test__conversions_from_angular_and_solMass_and_back__errors_raised_if_no_exposure_time(self):

        unit_angular_eps = units.FloatMassOverLuminosity(value=2.0, unit_mass='angular',
                                                     unit_luminosity='electrons_per_second')

        assert unit_angular_eps == 2.0
        assert unit_angular_eps.unit_type == 'mass / luminosity'
        assert unit_angular_eps.unit_mass == 'angular'
        assert unit_angular_eps.unit == 'angular / electrons_per_second'

        unit_angular_eps = unit_angular_eps.convert(unit_mass='angular', unit_luminosity='electrons_per_second')

        assert unit_angular_eps == 2.0
        assert unit_angular_eps.unit_type == 'mass / luminosity'
        assert unit_angular_eps.unit_mass == 'angular'
        assert unit_angular_eps.unit == 'angular / electrons_per_second'

        unit_solMass_eps = unit_angular_eps.convert(unit_mass='solMass', critical_surface_mass_density=2.0,
                                            unit_luminosity='electrons_per_second')

        assert unit_solMass_eps == 4.0
        assert unit_solMass_eps.unit_type == 'mass / luminosity'
        assert unit_solMass_eps.unit_mass == 'solMass'
        assert unit_solMass_eps.unit == 'solMass / electrons_per_second'

        unit_solMass_counts = unit_solMass_eps.convert(unit_mass='solMass', unit_luminosity='counts',
                                                       exposure_time=4.0)

        assert unit_solMass_counts == 1.0
        assert unit_solMass_counts.unit_type == 'mass / luminosity'
        assert unit_solMass_counts.unit_mass == 'solMass'
        assert unit_solMass_counts.unit == 'solMass / counts'

        unit_angular_counts = unit_solMass_counts.convert(unit_mass='angular', critical_surface_mass_density=2.0,
                                                       unit_luminosity='counts')

        assert unit_angular_counts == 0.5
        assert unit_angular_counts.unit_type == 'mass / luminosity'
        assert unit_angular_counts.unit_mass == 'angular'
        assert unit_angular_counts.unit == 'angular / counts'

        with pytest.raises(exc.UnitsException):
            unit_angular_eps.convert(unit_mass='solMass', unit_luminosity='counts')
            unit_solMass_eps.convert(unit_mass='angular', unit_luminosity='electrons_per_second')
            unit_angular_eps.convert(unit_mass='lol')


class TestTupleDistance(object):

    def test__conversions_from_arcsec_to_kpc_and_back__errors_raised_if_no_kpc_per_arcsec(self):

        unit_arcsec = units.TupleDistance(value=(1.0, 2.0), unit_distance='arcsec')

        assert unit_arcsec == (1.0, 2.0)
        assert unit_arcsec.unit_type == 'distance'
        assert unit_arcsec.unit == 'arcsec'

        unit_arcsec = unit_arcsec.convert(unit_distance='arcsec')

        assert unit_arcsec == (1.0, 2.0)
        assert unit_arcsec.unit_type == 'distance'
        assert unit_arcsec.unit == 'arcsec'

        unit_kpc = unit_arcsec.convert(unit_distance='kpc', kpc_per_arcsec=2.0)

        assert unit_kpc == (2.0, 4.0)
        assert unit_kpc.unit_type == 'distance'
        assert unit_kpc.unit == 'kpc'

        unit_kpc = unit_kpc.convert(unit_distance='kpc')

        assert unit_kpc == (2.0, 4.0)
        assert unit_kpc.unit_type == 'distance'
        assert unit_kpc.unit == 'kpc'

        unit_arcsec = unit_kpc.convert(unit_distance='arcsec', kpc_per_arcsec=2.0)

        assert unit_arcsec == (1.0, 2.0)
        assert unit_arcsec.unit_type == 'distance'
        assert unit_arcsec.unit == 'arcsec'

        with pytest.raises(exc.UnitsException):
            unit_arcsec.convert(unit_distance='kpc')
            unit_kpc.convert(unit_distance='arcsec')
            unit_arcsec.convert(unit_distance='lol')
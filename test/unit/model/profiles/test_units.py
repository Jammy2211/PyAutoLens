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

        unit_arcsec = units.FloatDistance(value=2.0, unit='arcsec')

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit_type == 'distance'
        assert unit_arcsec.unit == 'arcsec'

        unit_arcsec = unit_arcsec.convert(unit='arcsec')

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit_type == 'distance'
        assert unit_arcsec.unit == 'arcsec'

        unit_kpc = unit_arcsec.convert(unit='kpc', kpc_per_arcsec=2.0)

        assert unit_kpc == 4.0
        assert unit_kpc.unit_type == 'distance'
        assert unit_kpc.unit == 'kpc'

        unit_kpc = unit_kpc.convert(unit='kpc')

        assert unit_kpc == 4.0
        assert unit_kpc.unit_type == 'distance'
        assert unit_kpc.unit == 'kpc'

        unit_arcsec = unit_kpc.convert(unit='arcsec', kpc_per_arcsec=2.0)

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit_type == 'distance'
        assert unit_arcsec.unit == 'arcsec'

        with pytest.raises(exc.UnitsException):
            unit_arcsec.convert(unit='kpc')
            unit_kpc.convert(unit='arcsec')
            unit_arcsec.convert(unit='lol')


class TestFloatLuminosity(object):

    def test__conversions_from_electrons_per_second_and_counts_and_back__errors_raised_if_no_exposure_time(self):

        unit_electrons_per_second = units.FloatLuminosity(value=2.0, unit='electrons_per_second')

        assert unit_electrons_per_second == 2.0
        assert unit_electrons_per_second.unit_type == 'luminosity'
        assert unit_electrons_per_second.unit == 'electrons_per_second'

        unit_electrons_per_second = unit_electrons_per_second.convert(unit='electrons_per_second')

        assert unit_electrons_per_second == 2.0
        assert unit_electrons_per_second.unit_type == 'luminosity'
        assert unit_electrons_per_second.unit == 'electrons_per_second'

        unit_counts = unit_electrons_per_second.convert(unit='counts', exposure_time=2.0)

        assert unit_counts == 4.0
        assert unit_counts.unit_type == 'luminosity'
        assert unit_counts.unit == 'counts'

        unit_counts = unit_counts.convert(unit='counts')

        assert unit_counts == 4.0
        assert unit_counts.unit_type == 'luminosity'
        assert unit_counts.unit == 'counts'

        unit_electrons_per_second = unit_counts.convert(unit='electrons_per_second', exposure_time=2.0)

        assert unit_electrons_per_second == 2.0
        assert unit_electrons_per_second.unit_type == 'luminosity'
        assert unit_electrons_per_second.unit == 'electrons_per_second'

        with pytest.raises(exc.UnitsException):
            unit_electrons_per_second.convert(unit='counts')
            unit_counts.convert(unit='electrons_per_second')
            unit_electrons_per_second.convert(unit='lol')


class TestTupleDistance(object):

    def test__conversions_from_arcsec_to_kpc_and_back__errors_raised_if_no_kpc_per_arcsec(self):

        unit_arcsec = units.TupleDistance(value=(1.0, 2.0), unit='arcsec')

        assert unit_arcsec == (1.0, 2.0)
        assert unit_arcsec.unit_type == 'distance'
        assert unit_arcsec.unit == 'arcsec'

        unit_arcsec = unit_arcsec.convert(unit='arcsec')

        assert unit_arcsec == (1.0, 2.0)
        assert unit_arcsec.unit_type == 'distance'
        assert unit_arcsec.unit == 'arcsec'

        unit_kpc = unit_arcsec.convert(unit='kpc', kpc_per_arcsec=2.0)

        assert unit_kpc == (2.0, 4.0)
        assert unit_kpc.unit_type == 'distance'
        assert unit_kpc.unit == 'kpc'

        unit_kpc = unit_kpc.convert(unit='kpc')

        assert unit_kpc == (2.0, 4.0)
        assert unit_kpc.unit_type == 'distance'
        assert unit_kpc.unit == 'kpc'

        unit_arcsec = unit_kpc.convert(unit='arcsec', kpc_per_arcsec=2.0)

        assert unit_arcsec == (1.0, 2.0)
        assert unit_arcsec.unit_type == 'distance'
        assert unit_arcsec.unit == 'arcsec'

        with pytest.raises(exc.UnitsException):
            unit_arcsec.convert(unit='kpc')
            unit_kpc.convert(unit='arcsec')
            unit_arcsec.convert(unit='lol')
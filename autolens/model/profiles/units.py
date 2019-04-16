from autolens import exc


class FloatUnit(float):

    def __new__(self, value, unit):
        return float.__new__(self, value)

    def __init__(self, value, unit):
        float.__init__(value)
        self.unit = unit


class FloatNone(FloatUnit):

    def __new__(self, value):
        return FloatUnit.__new__(self, value, unit=None)

    def __init__(self, value):
        FloatUnit.__init__(self, value, unit=None)
        self.unit_type = None

    def convert(self):
        return exc.UnitsException('Cannot convert units for a dimensionless value')

class FloatDistance(FloatUnit):

    def __new__(self, value, unit):
        return FloatUnit.__new__(self, value, unit)

    def __init__(self, value, unit):
        FloatUnit.__init__(self, value, unit)
        self.unit_type = 'distance'

    def convert(self, unit, kpc_per_arcsec=None):

        if self.unit is not unit and kpc_per_arcsec is None:
            raise exc.UnitsException('The distance for a obj has been requested in new units without a '
                                     'kpc_per_arcsec conversion factor.')

        if self.unit is unit:
            obj = self
        elif self.unit is 'arcsec' and unit is 'kpc':
            obj = kpc_per_arcsec * self
        elif self.unit is 'kpc' and unit is 'arcsec':
            obj = self / kpc_per_arcsec
        else:
            raise exc.UnitsException('The unit specified for the distance of a obj was an invalid string, you '
                                     'must use (arcsec | kpc)')

        return FloatDistance(value=obj, unit=unit)


class FloatLuminosity(FloatUnit):

    def __new__(self, value, unit):
        return FloatUnit.__new__(self, value, unit)

    def __init__(self, value, unit):
        FloatUnit.__init__(self, value, unit)
        self.unit_type = 'luminosity'

    def convert(self, unit, exposure_time=None):

        if self.unit is not unit and exposure_time is None:
            raise exc.UnitsException('The luminosity for a obj has been requested in new units '
                                     'without an  exposure time conversion factor.')

        if self.unit is unit:
            obj = self
        elif self.unit is 'electrons_per_second' and unit is 'counts':
            obj = exposure_time * self
        elif self.unit is 'counts' and unit is 'electrons_per_second':
            obj = self / exposure_time
        else:
            raise exc.UnitsException('The unit specified for the luminosity of a obj was an invalid string, you '
                                     'must use (electrons per second | counts)')

        return FloatLuminosity(value=obj, unit=unit)


class TupleUnit(tuple):

    def __new__ (cls, value, unit):

        obj = super(TupleUnit, cls).__new__(cls, value)
        obj.unit = unit
        return obj


class TupleDistance(TupleUnit):

    def __new__ (cls, value, unit):

        obj = super(TupleDistance, cls).__new__(cls, value, unit)
        obj.unit = unit
        obj.unit_type = 'distance'
        return obj

    def convert(self, unit, kpc_per_arcsec=None):

        if unit is not self.unit and kpc_per_arcsec is None:
            raise exc.UnitsException('The distance for a obj has been requested in new units without a '
                                     'kpc_per_arcsec conversion factor.')

        if self.unit is unit:
            obj = self
        elif self.unit is 'arcsec' and unit is 'kpc':
            obj = (kpc_per_arcsec * self[0], kpc_per_arcsec * self[1])
        elif self.unit is 'kpc' and unit is 'arcsec':
            obj = (self[0] / kpc_per_arcsec, self[1] / kpc_per_arcsec)
        else:
            raise exc.UnitsException('The unit specified for the distance of a obj was an invalid string, you '
                                     'must use (arcsec | kpc)')

        return TupleDistance(value=obj, unit=unit)
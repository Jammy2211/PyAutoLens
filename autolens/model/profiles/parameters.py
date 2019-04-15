from autolens import exc

class ParameterTuple(tuple):

    def __new__ (cls, value, unit):

        parameter = super(ParameterTuple, cls).__new__(cls, value)
        parameter.unit = unit
        return parameter

class ParameterTupleDistance(ParameterTuple):

    def __new__ (cls, value, unit):

        parameter = super(ParameterTupleDistance, cls).__new__(cls, value, unit)
        parameter.unit = unit
        return parameter

    @property
    def unit_type(self):
        return 'distance'

    def convert(self, unit, kpc_per_arcsec=None):

        if unit is not self.unit and kpc_per_arcsec is None:
            raise exc.UnitsException('The distance for a parameter has been requested in new units without a '
                                     'kpc_per_arcsec conversion factor.')

        if self.unit is unit:
            parameter = self
        elif self.unit is 'arcsec' and unit is 'kpc':
            parameter = (kpc_per_arcsec * self[0], kpc_per_arcsec * self[1])
        elif self.unit is 'kpc' and unit is 'arcsec':
            parameter = (self[0] / kpc_per_arcsec, self[1] / kpc_per_arcsec)
        else:
            raise exc.UnitsException('The unit specified for the distance of a parameter was an invalid string, you '
                                     'must use (arcsec | kpc)')

        return ParameterTupleDistance(value=parameter, unit=unit)

class Parameter(float):

    def __new__(self, value, unit):
        return float.__new__(self, value)

    def __init__(self, value, unit):
        float.__init__(value)
        self.unit = unit

class ParameterNoUnit(Parameter):

    def __new__(self, value):
        return Parameter.__new__(self, value, unit=None)

    def __init__(self, value):
        Parameter.__init__(self, value, unit=None)

    @property
    def unit_type(self):
        return None

class ParameterDistance(Parameter):

    def __new__(self, value, unit):
        return Parameter.__new__(self, value, unit)

    def __init__(self, value, unit):
        Parameter.__init__(self, value, unit)

    @property
    def unit_type(self):
        return 'distance'

    def convert(self, unit, kpc_per_arcsec=None):

        if self.unit is not unit and kpc_per_arcsec is None:
            raise exc.UnitsException('The distance for a parameter has been requested in new units without a '
                                     'kpc_per_arcsec conversion factor.')

        if self.unit is unit:
            parameter = self
        elif self.unit is 'arcsec' and unit is 'kpc':
            parameter = kpc_per_arcsec * self
        elif self.unit is 'kpc' and unit is 'arcsec':
            parameter = self / kpc_per_arcsec
        else:
            raise exc.UnitsException('The unit specified for the distance of a parameter was an invalid string, you '
                                     'must use (arcsec | kpc)')

        return ParameterDistance(value=parameter, unit=unit)


class ParameterLuminosity(Parameter):

    def __new__(self, value, unit):
        return Parameter.__new__(self, value, unit)

    def __init__(self, value, unit):
        Parameter.__init__(self, value, unit)

    @property
    def unit_type(self):
        return 'luminosity'

    def convert(self, unit, exposure_time=None):

        print(self.unit, unit)

        if self.unit is not unit and exposure_time is None:
            raise exc.UnitsException('The luminosity for a parameter has been requested in new units '
                                     'without an  exposure time conversion factor.')

        if self.unit is unit:
            parameter = self
        elif self.unit is 'electrons_per_second' and unit is 'counts':
            parameter = exposure_time * self
        elif self.unit is 'counts' and unit is 'electrons_per_second':
            parameter = self / exposure_time
        else:
            raise exc.UnitsException('The unit specified for the luminosity of a parameter was an invalid string, you '
                                     'must use (electrons per second | counts)')

        return ParameterLuminosity(value=parameter, unit=unit)

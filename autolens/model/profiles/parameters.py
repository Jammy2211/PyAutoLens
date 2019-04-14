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
            raise exc.UnitsException('The units for a profile distance conversion has been input in different units '
                                     'to the profile but a kpc per arcsec was not supplied.')

        if self.unit is unit:
            return self
        elif self.unit is 'arcsec' and unit is 'kpc':
            parameter = (kpc_per_arcsec * self[0], kpc_per_arcsec * self[1])
            return ParameterTupleDistance(value=parameter, unit=unit)
        elif self.unit is 'kpc' and unit is 'arcsec':
            parameter = (self[0] / kpc_per_arcsec, self[1] / kpc_per_arcsec)
            return ParameterTupleDistance(value=parameter, unit=unit)

class Parameter(float):

    def __new__(self, value, unit):
        return float.__new__(self, value)

    def __init__(self, value, unit):
        float.__init__(value)
        self.unit = unit

class ParameterNoUnit(Parameter):

    def __new__(self, value, unit):
        return Parameter.__new__(self, value, unit)

    def __init__(self, value, unit):
        Parameter.__init__(self, value, unit)

    @property
    def unit_type(self):
        return None

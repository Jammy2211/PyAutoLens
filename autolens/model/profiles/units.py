from autolens import exc


class FloatNone(float):

    def __new__(self, value):
        return float.__new__(self, value)

    def __init__(self, value):
        float.__init__(value)
        self.unit = None
        self.unit_type = None

    def convert(self):
        raise exc.UnitsException('Cannot convert units for a dimensionless value')


class FloatDistance(float):

    def __new__(self, value, unit_distance):
        return float.__new__(self, value)

    def __init__(self, value, unit_distance):
        float.__init__(value)
        self.unit_distance = unit_distance
        self.unit_type = 'distance'

    @property
    def unit(self):
        return self.unit_distance

    def convert(self, unit_distance, kpc_per_arcsec=None):

        if self.unit_distance is not unit_distance and kpc_per_arcsec is None:
            raise exc.UnitsException('The distance for a obj has been requested in new units without a '
                                     'kpc_per_arcsec conversion factor.')

        if self.unit_distance is unit_distance:
            obj = self
        elif self.unit_distance is 'arcsec' and unit_distance is 'kpc':
            obj = kpc_per_arcsec * self
        elif self.unit_distance is 'kpc' and unit_distance is 'arcsec':
            obj = self / kpc_per_arcsec
        else:
            raise exc.UnitsException('The unit specified for the distance of a obj was an invalid string, you '
                                     'must use (arcsec | kpc)')

        return FloatDistance(value=obj, unit_distance=unit_distance)


class FloatLuminosity(float):

    def __new__(self, value, unit_luminosity):
        return float.__new__(self, value)

    def __init__(self, value, unit_luminosity):
        float.__init__(value)
        self.unit_luminosity = unit_luminosity
        self.unit_type = 'luminosity'

    @property
    def unit(self):
        return self.unit_luminosity

    def convert(self, unit_luminosity, exposure_time=None):

        if self.unit_luminosity is not unit_luminosity and exposure_time is None:
            raise exc.UnitsException('The luminosity for a obj has been requested in new units '
                                     'without an  exposure time conversion factor.')

        if self.unit_luminosity is unit_luminosity:
            obj = self
        elif self.unit_luminosity is 'electrons_per_second' and unit_luminosity is 'counts':
            obj = exposure_time * self
        elif self.unit_luminosity is 'counts' and unit_luminosity is 'electrons_per_second':
            obj = self / exposure_time
        else:
            raise exc.UnitsException('The unit specified for the luminosity of a obj was an invalid string, you '
                                     'must use (electrons per second | counts)')

        return FloatLuminosity(value=obj, unit_luminosity=unit_luminosity)


class FloatMass(float):

    def __new__(self, value, unit_mass):
        return float.__new__(self, value)

    def __init__(self, value, unit_mass):
        float.__init__(value)
        self.unit_mass = unit_mass
        self.unit_type = 'mass'

    @property
    def unit(self):
        return self.unit_mass

    def convert(self, unit_mass, critical_surface_mass_density=None):

        if self.unit is not unit_mass and critical_surface_mass_density is None:
            raise exc.UnitsException('The mass for a obj has been requested in new units '
                                     'without a critical surface mass density conversion factor.')

        if self.unit_mass is unit_mass:
            obj = self
        elif self.unit_mass is 'angular' and unit_mass is 'solMass':
            obj = critical_surface_mass_density * self
        elif self.unit_mass is 'solMass' and unit_mass is 'angular':
            obj = self / critical_surface_mass_density
        else:
            raise exc.UnitsException('The unit specified for the mass of a obj was an invalid string, you '
                                     'must use (angular | solMass)')

        return FloatMass(value=obj, unit_mass=unit_mass)


class FloatMassOverLuminosity(float):

    def __new__(self, value, unit_mass, unit_luminosity):
        return float.__new__(self, value)

    def __init__(self, value, unit_mass, unit_luminosity):
        float.__init__(value)
        self.unit_mass = unit_mass
        self.unit_luminosity = unit_luminosity
        self.unit_type = 'mass / luminosity'

    @property
    def unit(self):
        return self.unit_mass + ' / ' + self.unit_luminosity

    def convert(self, unit_mass, unit_luminosity, critical_surface_mass_density=None, exposure_time=None):

        if self.unit_mass is not unit_mass and critical_surface_mass_density is None:
            raise exc.UnitsException('The mass for a obj has been requested in new units '
                                     'without a critical surface mass density conversion factor.')

        if self.unit_mass is unit_mass:
            obj = self
        elif self.unit_mass is 'angular' and unit_mass is 'solMass':
            obj = critical_surface_mass_density * self
        elif self.unit_mass is 'solMass' and unit_mass is 'angular':
            obj = self / critical_surface_mass_density
        else:
            raise exc.UnitsException('The unit specified for the mass of a obj was an invalid string, you '
                                     'must use (electrons per second | counts)')

        if self.unit_luminosity is not unit_luminosity and exposure_time is None:
            raise exc.UnitsException('The luminosity for a obj has been requested in new units '
                                     'without an  exposure time conversion factor.')

        if self.unit_luminosity is unit_luminosity:
            obj = obj
        elif self.unit_luminosity is 'electrons_per_second' and unit_luminosity is 'counts':
            obj = obj / exposure_time
        elif self.unit_luminosity is 'counts' and unit_luminosity is 'electrons_per_second':
            obj = obj * exposure_time
        else:
            raise exc.UnitsException('The unit specified for the luminosity of a obj was an invalid string, you '
                                     'must use (electrons per second | counts)')

        return FloatMassOverLuminosity(value=obj, unit_mass=unit_mass, unit_luminosity=unit_luminosity)


class TupleUnit(tuple):

    def __new__ (cls, value, unit):

        obj = super(TupleUnit, cls).__new__(cls, value)
        obj.unit = unit
        return obj


class TupleDistance(TupleUnit):

    def __new__ (cls, value, unit_distance):

        obj = super(TupleDistance, cls).__new__(cls, value, unit_distance)
        obj.unit = unit_distance
        obj.unit_type = 'distance'
        return obj

    def convert(self, unit_distance, kpc_per_arcsec=None):

        if unit_distance is not self.unit and kpc_per_arcsec is None:
            raise exc.UnitsException('The distance for a obj has been requested in new units without a '
                                     'kpc_per_arcsec conversion factor.')

        if self.unit is unit_distance:
            obj = self
        elif self.unit is 'arcsec' and unit_distance is 'kpc':
            obj = (kpc_per_arcsec * self[0], kpc_per_arcsec * self[1])
        elif self.unit is 'kpc' and unit_distance is 'arcsec':
            obj = (self[0] / kpc_per_arcsec, self[1] / kpc_per_arcsec)
        else:
            raise exc.UnitsException('The unit specified for the distance of a obj was an invalid string, you '
                                     'must use (arcsec | kpc)')

        return TupleDistance(value=obj, unit_distance=unit_distance)
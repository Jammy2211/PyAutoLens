import inspect
import typing

from autofit.tools import dimension_type
from autolens import exc


class DimensionsProfile(object):

    def __init__(self):

        pass

    def new_profile_with_units_converted(self, unit_length=None, unit_luminosity=None, unit_mass=None,
                                         kpc_per_arcsec=None, exposure_time=None, critical_surface_density=None):

        constructor_args = inspect.getfullargspec(self.__init__).args

        def convert(value):
            if unit_length is not None:
                if isinstance(value, Length):
                    return value.convert(unit_length, kpc_per_arcsec)
                if isinstance(value, tuple):
                    return tuple(convert(item) for item in value)
            if unit_luminosity is not None and isinstance(value, Luminosity):
                return value.convert(unit_luminosity, exposure_time)
            if unit_mass is not None and isinstance(value, Mass):
                return value.convert(unit_mass, critical_surface_density)
            if (unit_mass is not None or unit_luminosity is not None) and isinstance(value, MassOverLuminosity):
                return value.convert(unit_luminosity, unit_mass, exposure_time, critical_surface_density)
            return value

        return self.__class__(
            **{key: convert(value) for key, value in self.__dict__.items() if key in constructor_args})


class Length(dimension_type.DimensionType):

    def __init__(self, value, unit_length="arcsec"):
        super().__init__(value)
        self.unit_length = unit_length
        self.unit_length_power = 1.0

    @property
    def unit(self):
        return self.unit_length

    def convert(self, unit_length, kpc_per_arcsec=None):

        value = self

        value = convert_length(value=value, unit_current=self.unit_length, unit_new=unit_length,
                               power=self.unit_length_power, kpc_per_arcsec=kpc_per_arcsec)

        return Length(value=value, unit_length=unit_length)


class Luminosity(dimension_type.DimensionType):

    def __init__(self, value, unit_luminosity='eps'):
        super().__init__(value)
        self.unit_luminosity = unit_luminosity
        self.unit_luminosity_power = 1.0

    @property
    def unit(self):
        return self.unit_luminosity

    def convert(self, unit_luminosity, exposure_time=None):

        value = self

        value = convert_luminosity(value=value, unit_current=self.unit_luminosity, unit_new=unit_luminosity,
                                   power=self.unit_luminosity_power, exposure_time=exposure_time)

        return Luminosity(value=value, unit_luminosity=unit_luminosity)


class Mass(dimension_type.DimensionType):

    def __init__(self, value, unit_mass="angular"):
        super().__init__(value)
        self.unit_mass = unit_mass
        self.unit_mass_power = 1.0

    @property
    def unit(self):
        return self.unit_mass

    def convert(self, unit_mass, critical_surface_density=None):

        value = self

        value = convert_mass(value=value, unit_current=self.unit_mass, unit_new=unit_mass,
                             critical_surface_density=critical_surface_density)

        return Mass(value=value, unit_mass=unit_mass)


class MassOverLuminosity(dimension_type.DimensionType):

    def __init__(self, value, unit_luminosity="eps", unit_mass="angular"):
        super().__init__(value)
        self.unit_luminosity = unit_luminosity
        self.unit_luminosity_power = -1.0
        self.unit_mass = unit_mass
        self.unit_mass_power = 1.0

    @property
    def unit(self):
        return self.unit_mass + ' / ' + self.unit_luminosity

    def convert(self, unit_luminosity, unit_mass,  exposure_time=None, critical_surface_density=None):
        
        value = self
        if unit_luminosity is not None:
            value = convert_luminosity(value=value, unit_current=self.unit_luminosity, unit_new=unit_luminosity,
                                       power=self.unit_luminosity_power, exposure_time=exposure_time)
        else:
            unit_luminosity = self.unit_luminosity

        if unit_mass is not None:
            value = convert_mass(value=value, unit_current=self.unit_mass, unit_new=unit_mass,
                                 critical_surface_density=critical_surface_density)
        else:
            unit_mass = self.unit_mass

        return MassOverLuminosity(value=value, unit_mass=unit_mass, unit_luminosity=unit_luminosity)


class MassOverLength2(dimension_type.DimensionType):

    def __init__(self, value, unit_length="arcsec", unit_mass="angular"):
        super().__init__(value)
        self.unit_length = unit_length
        self.unit_length_power = -2.0
        self.unit_mass = unit_mass
        self.unit_mass_power = 1.0

    @property
    def unit(self):
        return self.unit_mass + ' / ' + self.unit_length + '^2'

    def convert(self, unit_length, unit_mass, kpc_per_arcsec=None, critical_surface_density=None,):

        value = self

        if unit_length is not None:
            value = convert_length(value=value, unit_current=self.unit_length, unit_new=unit_length,
                                       power=self.unit_length_power, kpc_per_arcsec=kpc_per_arcsec)
        else:
            unit_length = value.unit_length

        if unit_mass is not None:
            value = convert_mass(value=value, unit_current=self.unit_mass, unit_new=unit_mass,
                                 critical_surface_density=critical_surface_density)
        else:
            unit_mass = value.unit_mass

        return MassOverLength2(value=value, unit_mass=unit_mass, unit_length=unit_length)


class MassOverLength3(dimension_type.DimensionType):

    def __init__(self, value, unit_length="arcsec", unit_mass="angular"):
        super().__init__(value)
        self.unit_length = unit_length
        self.unit_length_power = -3.0
        self.unit_mass = unit_mass
        self.unit_mass_power = 1.0

    @property
    def unit(self):
        return self.unit_mass + ' / ' + self.unit_length + '^3'

    def convert(self, unit_length, unit_mass, kpc_per_arcsec=None, critical_surface_density=None):

        value = self

        if unit_length is not None:
            value = convert_length(value=value, unit_current=self.unit_length, unit_new=unit_length,
                                       power=self.unit_length_power, kpc_per_arcsec=kpc_per_arcsec)
        else:
            unit_length = value.unit_length

        if unit_mass is not None:
            value = convert_mass(value=value, unit_current=self.unit_mass, unit_new=unit_mass,
                                 critical_surface_density=critical_surface_density)
        else:
            unit_mass = value.unit_mass

        return MassOverLength3(value=value, unit_mass=unit_mass, unit_length=unit_length)

Position = typing.Tuple[Length, Length]

def convert_length(value, unit_current, unit_new, power, kpc_per_arcsec):
    
    if unit_current is not unit_new and kpc_per_arcsec is None:
        raise exc.UnitsException('The length for a value has been requested in new units without a '
                                 'kpc_per_arcsec conversion factor.')

    if unit_current is unit_new:
        return value
    elif unit_current is 'arcsec' and unit_new is 'kpc':
        return (kpc_per_arcsec**power) * value
    elif unit_current is 'kpc' and unit_new is 'arcsec':
        return value / (kpc_per_arcsec**power)
    else:
        raise exc.UnitsException('The unit specified for the length of a value was an invalid string, you '
                                 'must use (arcsec | kpc)')

def convert_luminosity(value, unit_current, unit_new, power, exposure_time):
    
    if unit_current is not unit_new and exposure_time is None:
        raise exc.UnitsException('The luminosity for a value has been requested in new units '
                                 'without an  exposure time conversion factor.')

    if unit_current is unit_new:
        return value
    elif unit_current is 'eps' and unit_new is 'counts':
        return (exposure_time**power) * value
    elif unit_current is 'counts' and unit_new is 'eps':
        return value / (exposure_time**power)
    else:
        raise exc.UnitsException('The unit specified for the luminosity of a value was an invalid string, you '
                                 'must use (electrons per second | counts)')

def convert_mass(value, unit_current, unit_new, critical_surface_density):
    
    if unit_current is not unit_new and critical_surface_density is None:
        raise exc.UnitsException('The mass for a value has been requested in new units '
                                 'without a critical surface mass density conversion factor.')

    if unit_current is unit_new:
        return value
    elif unit_current is 'angular' and unit_new is 'solMass':
        return critical_surface_density * value
    elif unit_current is 'solMass' and unit_new is 'angular':
        return value / critical_surface_density
    else:
        raise exc.UnitsException('The unit specified for the mass of a value was an invalid string, you '
                                 'must use (angular | solMass)')

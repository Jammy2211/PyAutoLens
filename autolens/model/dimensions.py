import inspect
import typing

from autofit.tools import dimension_type
from autolens import exc


class DimensionsProfile(object):

    def __init__(self):

        pass

    def new_profile_with_units_converted(self, unit_length=None, unit_luminosity=None, unit_mass=None,
                                         kpc_per_arcsec=None, exposure_time=None, critical_surface_mass_density=None):

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
                return value.convert(unit_mass, critical_surface_mass_density)
            if (unit_mass is not None or unit_luminosity is not None) and isinstance(value, MassOverLuminosity):
                return value.convert(unit_mass, unit_luminosity, critical_surface_mass_density, exposure_time)
            return value

        return self.__class__(
            **{key: convert(value) for key, value in self.__dict__.items() if key in constructor_args})


class Length(dimension_type.DimensionType):

    def __init__(self, value, unit="arcsec"):
        super().__init__(value)
        self.unit = unit

    def convert(self, unit_length, kpc_per_arcsec=None):

        if self.unit is not unit_length and kpc_per_arcsec is None:
            raise exc.UnitsException('The length for a value has been requested in new units without a '
                                     'kpc_per_arcsec conversion factor.')

        if self.unit is unit_length:
            value = self
        elif self.unit is 'arcsec' and unit_length is 'kpc':
            value = kpc_per_arcsec * self
        elif self.unit is 'kpc' and unit_length is 'arcsec':
            value = self / kpc_per_arcsec
        else:
            raise exc.UnitsException('The unit specified for the length of a value was an invalid string, you '
                                     'must use (arcsec | kpc)')

        return Length(value=value, unit=unit_length)


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

    def convert(self, unit_mass, critical_surface_mass_density=None):

        value = self

        value = convert_mass(value=value, unit_current=self.unit_mass, unit_new=unit_mass,
                             critical_surface_mass_density=critical_surface_mass_density)

        return Mass(value=value, unit_mass=unit_mass)


class MassOverLuminosity(dimension_type.DimensionType):

    def __init__(self, value, unit_mass="angular", unit_luminosity="eps"):
        super().__init__(value)
        self.unit_mass = unit_mass
        self.unit_mass_power = 1.0
        self.unit_luminosity = unit_luminosity
        self.unit_luminosity_power = -1.0

    @property
    def unit(self):
        return self.unit_mass + ' / ' + self.unit_luminosity

    def convert(self, unit_mass, unit_luminosity, critical_surface_mass_density=None, exposure_time=None):
        
        value = self
        
        if unit_mass is not None:
            value = convert_mass(value=value, unit_current=self.unit_mass, unit_new=unit_mass,
                                 critical_surface_mass_density=critical_surface_mass_density)
        else:
            unit_mass = value.unit_mass

        if unit_luminosity is not None:
            value = convert_luminosity(value=value, unit_current=self.unit_luminosity, unit_new=unit_luminosity,
                                       power=self.unit_luminosity_power, exposure_time=exposure_time)
        else:
            unit_luminosity = value.unit_luminosity

        return MassOverLuminosity(value=value, unit_mass=unit_mass, unit_luminosity=unit_luminosity)


Position = typing.Tuple[Length, Length]


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

def convert_mass(value, unit_current, unit_new, critical_surface_mass_density):
    
    if unit_current is not unit_new and critical_surface_mass_density is None:
        raise exc.UnitsException('The mass for a value has been requested in new units '
                                 'without a critical surface mass density conversion factor.')

    if unit_current is unit_new:
        return value
    elif unit_current is 'angular' and unit_new is 'solMass':
        return critical_surface_mass_density * value
    elif unit_current is 'solMass' and unit_new is 'angular':
        return value / critical_surface_mass_density
    else:
        raise exc.UnitsException('The unit specified for the mass of a value was an invalid string, you '
                                 'must use (angular | solMass)')

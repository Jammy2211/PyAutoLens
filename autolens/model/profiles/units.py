import typing

from autofit.tools import dimension_type
from autolens import exc


class Distance(dimension_type.DimensionType):

    def __init__(self, value, unit="arcsec"):
        super().__init__(value)
        self.unit = unit

    def convert(self, unit_distance, kpc_per_arcsec=None):

        if self.unit is not unit_distance and kpc_per_arcsec is None:
            raise exc.UnitsException('The distance for a obj has been requested in new units without a '
                                     'kpc_per_arcsec conversion factor.')

        if self.unit is unit_distance:
            obj = self
        elif self.unit is 'arcsec' and unit_distance is 'kpc':
            obj = kpc_per_arcsec * self
        elif self.unit is 'kpc' and unit_distance is 'arcsec':
            obj = self / kpc_per_arcsec
        else:
            raise exc.UnitsException('The unit specified for the distance of a obj was an invalid string, you '
                                     'must use (arcsec | kpc)')

        return Distance(value=obj, unit=unit_distance)


class Luminosity(dimension_type.DimensionType):

    def __init__(self, value, unit='electrons_per_second'):
        super().__init__(value)
        self.unit = unit

    def convert(self, unit_luminosity, exposure_time=None):

        if self.unit is not unit_luminosity and exposure_time is None:
            raise exc.UnitsException('The luminosity for a obj has been requested in new units '
                                     'without an  exposure time conversion factor.')

        if self.unit is unit_luminosity:
            obj = self
        elif self.unit is 'electrons_per_second' and unit_luminosity is 'counts':
            obj = exposure_time * self
        elif self.unit is 'counts' and unit_luminosity is 'electrons_per_second':
            obj = self / exposure_time
        else:
            raise exc.UnitsException('The unit specified for the luminosity of a obj was an invalid string, you '
                                     'must use (electrons per second | counts)')

        return Luminosity(value=obj, unit=unit_luminosity)


class Mass(dimension_type.DimensionType):

    def __init__(self, value, unit="angular"):
        super().__init__(value)
        self.unit = unit

    def convert(self, unit_mass, critical_surface_mass_density=None):

        if self.unit is not unit_mass and critical_surface_mass_density is None:
            raise exc.UnitsException('The mass for a obj has been requested in new units '
                                     'without a critical surface mass density conversion factor.')

        if self.unit is unit_mass:
            obj = self
        elif self.unit is 'angular' and unit_mass is 'solMass':
            obj = critical_surface_mass_density * self
        elif self.unit is 'solMass' and unit_mass is 'angular':
            obj = self / critical_surface_mass_density
        else:
            raise exc.UnitsException('The unit specified for the mass of a obj was an invalid string, you '
                                     'must use (angular | solMass)')

        return Mass(value=obj, unit=unit_mass)


class MassOverLuminosity(dimension_type.DimensionType):

    def __init__(self, value, unit_mass="angular", unit_luminosity="electrons_per_second"):
        float.__init__(value)
        self.unit_mass = unit_mass
        self.unit_luminosity = unit_luminosity

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

        return MassOverLuminosity(value=obj, unit_mass=unit_mass, unit_luminosity=unit_luminosity)


Position = typing.Tuple[Distance, Distance]

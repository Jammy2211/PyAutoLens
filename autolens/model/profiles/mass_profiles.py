import inspect
from pyquad import quad_grid

import numpy as np
from astropy import cosmology as cosmo
from numba import cfunc
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable
from scipy import special
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import root_scalar

import autofit as af
from autolens import decorator_util, dimensions as dim
from autolens import text_util
from autolens.data.array import grids
from autolens.model.profiles import geometry_profiles


def jit_integrand(integrand_function):
    jitted_function = decorator_util.jit(nopython=True, cache=True)(integrand_function)
    no_args = len(inspect.getfullargspec(integrand_function).args)

    wrapped = None

    if no_args == 4:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3])
    elif no_args == 5:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4])
    elif no_args == 6:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5])
    elif no_args == 7:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6])
    elif no_args == 8:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6],
                                   xx[7])
    elif no_args == 9:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6],
                                   xx[7], xx[8])
    elif no_args == 10:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6],
                                   xx[7], xx[8], xx[9])
    elif no_args == 11:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6],
                                   xx[7], xx[8], xx[9], xx[10])

    cf = cfunc(float64(intc, CPointer(float64)))

    return LowLevelCallable(cf(wrapped).ctypes)


class MassProfile(object):

    def convergence_func(self, eta):
        raise NotImplementedError("surface_density_func should be overridden")

    def convergence_from_grid(self, grid):
        pass
        # raise NotImplementedError("surface_density_from_grid should be overridden")

    def potential_from_grid(self, grid):
        pass
        # raise NotImplementedError("potential_from_grid should be overridden")

    def deflections_from_grid(self, grid):
        raise NotImplementedError("deflections_from_grid should be overridden")

    def mass_within_circle_in_units(self, radius: dim.Length, redshift_profile=None,
                                    redshift_source=None,
                                    unit_mass='solMass', cosmology=cosmo.Planck15,
                                    **kwargs):
        raise NotImplementedError()

    def mass_within_ellipse_in_units(self, major_axis: dim.Length,
                                     redshift_profile=None, redshift_source=None,
                                     unit_mass='solMass', cosmology=cosmo.Planck15,
                                     **kwargs):
        raise NotImplementedError()

    def einstein_radius_in_units(self, unit_length='arcsec', redshift_profile=None,
                                 cosmology=cosmo.Planck15):
        return NotImplementedError()

    def einstein_mass_in_units(self, unit_mass='solMass',
                               redshift_profile=None, redshift_source=None,
                               cosmology=cosmo.Planck15, **kwargs):
        return NotImplementedError()

    def summarize_in_units(self, radii, prefix='',
                           unit_length='arcsec', unit_mass='solMass',
                           redshift_profile=None, redshift_source=None,
                           cosmology=cosmo.Planck15,
                           whitespace=80, **kwargs):
        return ["Mass Profile = {}\n".format(self.__class__.__name__)]

    @property
    def unit_mass(self):
        return NotImplementedError()


class PointMass(geometry_profiles.SphericalProfile, MassProfile):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 einstein_radius: dim.Length = 1.0):
        """
        Represents a point-mass.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius of the point-mass.
        """
        super(PointMass, self).__init__(centre=centre)
        self.einstein_radius = einstein_radius

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return self.grid_to_grid_cartesian(grid=grid,
                                           radius=self.einstein_radius / grid_radii)

    # @property
    # def mass(self):
    #     return (206265 * self.einstein_radius * (constants.c**2.0) / (4.0 * constants.G)) / 1.988e30


# noinspection PyAbstractClass
class EllipticalMassProfile(geometry_profiles.EllipticalProfile, MassProfile):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ellipse's minor-to-major axis ratio (b/a)
        phi : float
            Rotation angle of profile's ellipse counter-clockwise from positive x-axis
        """
        super(EllipticalMassProfile, self).__init__(centre=centre,
                                                    axis_ratio=axis_ratio, phi=phi)
        self.axis_ratio = axis_ratio
        self.phi = phi

    @dim.convert_units_to_input_units
    def mass_within_circle_in_units(self, radius: dim.Length, unit_mass='solMass',
                                    redshift_profile=None, redshift_source=None,
                                    cosmology=cosmo.Planck15, **kwargs):
        """ Integrate the mass profiles's convergence profile to compute the total mass within a circle of \
        specified radius. This is centred on the mass profile.

        The following units for mass can be specified and output:

        - Dimensionless angular units (default) - 'solMass'.
        - Solar masses - 'solMass' (multiplies the angular mass by the critical surface mass density).

        Parameters
        ----------
        radius : dim.Length
            The radius of the circle to compute the dimensionless mass within.
        unit_mass : str
            The units the mass is returned in (angular | angular).
        critical_surface_density : float or None
            The critical surface mass density of the strong lens configuration, which converts mass from angulalr \
            units to phsical units (e.g. solar masses).
        """

        critical_surface_density = kwargs[
            'critical_surface_density'] if 'critical_surface_density' in kwargs else None

        mass = dim.Mass(value=quad(self.mass_integral, a=0.0, b=radius, args=(1.0,))[0],
                        unit_mass=self.unit_mass)

        return mass.convert(unit_mass=unit_mass,
                            critical_surface_density=critical_surface_density)

    @dim.convert_units_to_input_units
    def mass_within_ellipse_in_units(self, major_axis: dim.Length, unit_mass='solMass',
                                     redshift_profile=None, redshift_source=None,
                                     cosmology=cosmo.Planck15, **kwargs):
        """ Integrate the mass profiles's convergence profile to compute the total angular mass within an ellipse of \
        specified major axis. This is centred on the mass profile.

        The following units for mass can be specified and output:

        - Dimensionless angular units (default) - 'solMass'.
        - Solar masses - 'solMass' (multiplies the angular mass by the critical surface mass density)

        Parameters
        ----------
        major_axis : float
            The major-axis radius of the ellipse.
        unit_mass : str
            The units the mass is returned in (angular | angular).
        critical_surface_density : float or None
            The critical surface mass density of the strong lens configuration, which converts mass from angular \
            units to phsical units (e.g. solar masses).
        """

        critical_surface_density = kwargs[
            'critical_surface_density'] if 'critical_surface_density' in kwargs else None

        mass = dim.Mass(value=quad(self.mass_integral, a=0.0, b=major_axis,
                                   args=(self.axis_ratio,))[0],
                        unit_mass=self.unit_mass)

        return mass.convert(unit_mass=unit_mass,
                            critical_surface_density=critical_surface_density)

    def mass_integral(self, x, axis_ratio):
        """Routine to integrate an elliptical light profiles - set axis ratio to 1 to compute the luminosity within a \
        circle"""
        r = x * axis_ratio
        return 2 * np.pi * r * self.convergence_func(x)

    def density_between_circular_annuli_in_angular_units(
            self, inner_annuli_radius: dim.Length, outer_annuli_radius: dim.Length,
            unit_length='arcsec', unit_mass='solMass',
            redshift_profile=None, redshift_source=None, cosmology=cosmo.Planck15,
            **kwargs):
        """Calculate the mass between two circular annuli and compute the density by dividing by the annuli surface
        area.

        The value returned by the mass integral is dimensionless, therefore the density between annuli is returned in \
        units of inverse radius squared. A conversion factor can be specified to convert this to a physical value \
        (e.g. the critical surface mass density).

        Parameters
        -----------
        inner_annuli_radius : float
            The radius of the inner annulus outside of which the density are estimated.
        outer_annuli_radius : float
            The radius of the outer annulus inside of which the density is estimated.
        """
        annuli_area = (np.pi * outer_annuli_radius ** 2.0) - (
                np.pi * inner_annuli_radius ** 2.0)

        outer_mass = self.mass_within_circle_in_units(radius=outer_annuli_radius,
                                                      redshift_profile=redshift_profile,
                                                      redshift_source=redshift_source,
                                                      unit_mass=unit_mass,
                                                      cosmology=cosmology,
                                                      kwargs=kwargs)

        inner_mass = self.mass_within_circle_in_units(radius=inner_annuli_radius,
                                                      redshift_profile=redshift_profile,
                                                      redshift_source=redshift_source,
                                                      unit_mass=unit_mass,
                                                      cosmology=cosmology,
                                                      kwargs=kwargs)

        return dim.MassOverLength2(value=(outer_mass - inner_mass) / annuli_area,
                                   unit_length=unit_length, unit_mass=unit_mass)

    @dim.convert_units_to_input_units
    def average_convergence_of_1_radius_in_units(self, unit_length='arcsec',
                                                 redshift_profile=None,
                                                 cosmology=cosmo.Planck15, **kwargs):
        """The radius a critical curve forms for this mass profile, e.g. where the mean convergence is equal to 1.0.

         In case of ellipitical mass profiles, the 'average' critical curve is used, whereby the convergence is \
         rescaled into a circle using the axis ratio.

         This radius corresponds to the Einstein radius of the mass profile, and is a property of a number of \
         mass profiles below.
         """

        kpc_per_arcsec = kwargs[
            'kpc_per_arcsec'] if 'kpc_per_arcsec' in kwargs else None

        def func(radius, redshift_profile, cosmology):
            radius = dim.Length(radius, unit_length=unit_length)
            return self.mass_within_circle_in_units(unit_mass='angular', radius=radius,
                                                    redshift_profile=redshift_profile,
                                                    cosmology=cosmology) - \
                   np.pi * radius ** 2.0

        radius = self.ellipticity_rescale * root_scalar(func, bracket=[1e-4, 1000.0],
                                                        args=(redshift_profile,
                                                              cosmology)).root
        radius = dim.Length(radius, unit_length)
        return radius.convert(unit_length=unit_length, kpc_per_arcsec=kpc_per_arcsec)

    @dim.convert_units_to_input_units
    def einstein_radius_in_units(self, unit_length='arcsec',
                                 redshift_profile=None, cosmology=cosmo.Planck15,
                                 **kwargs):
        einstein_radius = self.average_convergence_of_1_radius_in_units(
            unit_length=unit_length, redshift_profile=redshift_profile,
            cosmology=cosmology, kwargs=kwargs)

        return dim.Length(einstein_radius, unit_length)

    @dim.convert_units_to_input_units
    def einstein_mass_in_units(self, unit_mass='solMass',
                               redshift_profile=None, redshift_source=None,
                               cosmology=cosmo.Planck15, **kwargs):
        einstein_radius = self.einstein_radius_in_units(unit_length=self.unit_length,
                                                        redshift_profile=redshift_profile,
                                                        cosmology=cosmology,
                                                        kwargs=kwargs)

        return self.mass_within_circle_in_units(radius=einstein_radius,
                                                unit_mass=unit_mass,
                                                redshift_profile=redshift_profile,
                                                redshift_source=redshift_source,
                                                cosmology=cosmology, kwargs=kwargs)

    @dim.convert_units_to_input_units
    def summarize_in_units(self, radii, prefix='', whitespace=80,
                           unit_length='arcsec', unit_mass='solMass',
                           redshift_profile=None, redshift_source=None,
                           cosmology=cosmo.Planck15, **kwargs):
        summary = super().summarize_in_units(
            radii=radii, prefix='', unit_length=unit_length, unit_mass=unit_mass,
            redshift_profile=redshift_profile, redshift_source=redshift_source,
            cosmology=cosmology, kwargs=kwargs)

        einstein_radius = self.einstein_radius_in_units(unit_length=unit_length,
                                                        redshift_profile=redshift_profile,
                                                        cosmology=cosmology,
                                                        kwargs=kwargs)

        summary += [
            af.text_util.label_value_and_unit_string(label=prefix + 'einstein_radius',
                                                     value=einstein_radius,
                                                     unit=unit_length,
                                                     whitespace=whitespace,
                                                     )]

        einstein_mass = self.einstein_mass_in_units(unit_mass=unit_mass,
                                                    redshift_profile=redshift_profile,
                                                    redshift_source=redshift_source,
                                                    cosmology=cosmology, kwargs=kwargs)

        summary += [
            af.text_util.label_value_and_unit_string(label=prefix + 'einstein_mass',
                                                     value=einstein_mass,
                                                     unit=unit_mass,
                                                     whitespace=whitespace,
                                                     )]

        for radius in radii:
            mass = self.mass_within_circle_in_units(unit_mass=unit_mass, radius=radius,
                                                    redshift_profile=redshift_profile,
                                                    redshift_source=redshift_source,
                                                    cosmology=cosmology,
                                                    kwargs=kwargs)

            summary += [text_util.within_radius_label_value_and_unit_string(
                prefix=prefix + 'mass', radius=radius, unit_length=unit_length,
                value=mass,
                unit_value=unit_mass, whitespace=whitespace)]

        return summary

    @property
    def ellipticity_rescale(self):
        return NotImplementedError()


class EllipticalCoredPowerLaw(EllipticalMassProfile, MassProfile):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 einstein_radius: dim.Length = 1.0,
                 slope: float = 2.0,
                 core_radius: dim.Length = 0.01):
        """
        Represents a cored elliptical power-law density distribution

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            The elliptical mass profile's minor-to-major axis ratio (b/a).
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis.
        einstein_radius : float
            The arc-second Einstein radius.
        slope : float
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        core_radius : float
            The arc-second radius of the inner core.
        """
        super(EllipticalCoredPowerLaw, self).__init__(centre=centre,
                                                      axis_ratio=axis_ratio, phi=phi)
        self.einstein_radius = einstein_radius
        self.slope = slope
        self.core_radius = core_radius

    @property
    def einstein_radius_rescaled(self):
        """Rescale the einstein radius by slope and axis_ratio, to reduce its degeneracy with other mass-profiles
        parameters"""
        return ((3 - self.slope) / (1 + self.axis_ratio)) * self.einstein_radius ** (self.slope - 1)

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def convergence_from_grid(self, grid):
        """ Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the surface density is computed on.
        """

        surface_density_grid = np.zeros(grid.shape[0])

        grid_eta = self.grid_to_elliptical_radii(grid)

        for i in range(grid.shape[0]):
            surface_density_grid[i] = self.convergence_func(grid_eta[i])

        return surface_density_grid

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        potential_grid = quad_grid(self.potential_func, 0.0, 1.0, grid,
                                   args=(
                                       self.axis_ratio, self.slope, self.core_radius))[
            0]

        return self.einstein_radius_rescaled * self.axis_ratio * potential_grid

    @grids.grid_interpolate
    @geometry_profiles.cache
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        def calculate_deflection_component(npow, index):
            einstein_radius_rescaled = self.einstein_radius_rescaled

            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= quad_grid(self.deflection_func, 0.0, 1.0,
                                         grid, args=(npow, self.axis_ratio,
                                                     einstein_radius_rescaled,
                                                     self.slope,
                                                     self.core_radius))[0]

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T))

    def convergence_func(self, radius):
        return self.einstein_radius_rescaled * (
                self.core_radius ** 2 + radius ** 2) ** (-(self.slope - 1) / 2.0)

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        eta = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (eta / u) * ((3.0 - slope) * eta) ** -1.0 * \
               ((core_radius ** 2.0 + eta ** 2.0) ** ((3.0 - slope) / 2.0) -
                core_radius ** (3 - slope)) / ((1 - (1 - axis_ratio ** 2) * u) ** 0.5)

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, einstein_radius_rescaled, slope,
                        core_radius):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return einstein_radius_rescaled * (core_radius ** 2 + eta_u ** 2) ** (
                -(slope - 1) / 2.0) / (
                       (1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio) / 2.0)

    @dim.convert_units_to_input_units
    def summarize_in_units(self, radii, prefix='', whitespace=80,
                           unit_length='arcsec', unit_mass='solMass',
                           redshift_profile=None, redshift_source=None,
                           cosmology=cosmo.Planck15,
                           **kwargs):
        summary = super().summarize_in_units(
            radii=radii, prefix=prefix, unit_length=unit_length, unit_mass=unit_mass,
            redshift_profile=redshift_profile, redshift_source=redshift_source,
            cosmology=cosmology,
            whitespace=whitespace, kwargs=kwargs)

        return summary

    @property
    def unit_mass(self):
        return 'angular'


class SphericalCoredPowerLaw(EllipticalCoredPowerLaw):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 einstein_radius: dim.Length = 1.0,
                 slope: float = 2.0,
                 core_radius: dim.Length = 0.01):
        """
        Represents a cored spherical power-law density distribution

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius.
        slope : float
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        core_radius : float
            The arc-second radius of the inner core.
        """
        super(SphericalCoredPowerLaw, self).__init__(centre=centre, axis_ratio=1.0,
                                                     phi=0.0,
                                                     einstein_radius=einstein_radius,
                                                     slope=slope,
                                                     core_radius=core_radius)

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        eta = self.grid_to_grid_radii(grid=grid)
        deflection = np.multiply(2. * self.einstein_radius_rescaled, np.divide(
            np.add(np.power(np.add(self.core_radius ** 2, np.square(eta)),
                            (3. - self.slope) / 2.),
                   -self.core_radius ** (3 - self.slope)),
            np.multiply((3. - self.slope), eta)))
        return self.grid_to_grid_cartesian(grid=grid, radius=deflection)


class EllipticalPowerLaw(EllipticalCoredPowerLaw):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 einstein_radius: dim.Length = 1.0,
                 slope: float = 2.0):
        """
        Represents an elliptical power-law density distribution.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            The elliptical mass profile's minor-to-major axis ratio (b/a).
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis.
        einstein_radius : float
            The arc-second Einstein radius.
        slope : float
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super(EllipticalPowerLaw, self).__init__(centre=centre, axis_ratio=axis_ratio,
                                                 phi=phi,
                                                 einstein_radius=einstein_radius,
                                                 slope=slope,
                                                 core_radius=dim.Length(0.0))

    def convergence_func(self, radius):
        if radius > 0.0:
            return self.einstein_radius_rescaled * radius ** (-(self.slope - 1))
        else:
            return np.inf

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (eta_u / u) * ((3.0 - slope) * eta_u) ** -1.0 * eta_u ** (3.0 - slope) / \
               ((1 - (1 - axis_ratio ** 2) * u) ** 0.5)

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, einstein_radius_rescaled, slope,
                        core_radius):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return einstein_radius_rescaled * eta_u ** (-(slope - 1)) / (
                (1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))


class SphericalPowerLaw(EllipticalPowerLaw):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 einstein_radius: dim.Length = 1.0,
                 slope: float = 2.0):
        """
        Represents a spherical power-law density distribution.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius.
        slope : float
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super(SphericalPowerLaw, self).__init__(centre=centre, axis_ratio=1.0, phi=0.0,
                                                einstein_radius=einstein_radius,
                                                slope=slope)

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        eta = self.grid_to_grid_radii(grid)
        deflection_r = 2.0 * self.einstein_radius_rescaled * np.divide(
            np.power(eta, (3.0 - self.slope)),
            np.multiply((3.0 - self.slope), eta))
        return self.grid_to_grid_cartesian(grid, deflection_r)


class EllipticalCoredIsothermal(EllipticalCoredPowerLaw):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 einstein_radius: dim.Length = 1.0,
                 core_radius: dim.Length = 0.01):
        """
        Represents a cored elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            The elliptical mass profile's minor-to-major axis ratio (b/a).
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis.
        einstein_radius : float
            The arc-second Einstein radius.
        core_radius : float
            The arc-second radius of the inner core.
        """
        super(EllipticalCoredIsothermal, self).__init__(centre=centre,
                                                        axis_ratio=axis_ratio, phi=phi,
                                                        einstein_radius=einstein_radius,
                                                        slope=2.0,
                                                        core_radius=core_radius)


class SphericalCoredIsothermal(SphericalCoredPowerLaw):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 einstein_radius: dim.Length = 1.0,
                 core_radius: dim.Length = 0.01):
        """
        Represents a cored spherical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius.
        core_radius : float
            The arc-second radius of the inner core.
        """
        super(SphericalCoredIsothermal, self).__init__(centre=centre,
                                                       einstein_radius=einstein_radius,
                                                       slope=2.0,
                                                       core_radius=core_radius)


class EllipticalIsothermal(EllipticalPowerLaw):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 einstein_radius: dim.Length = 1.0):
        """
        Represents an elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            The elliptical mass profile's minor-to-major axis ratio (b/a).
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis.
        einstein_radius : float
            The arc-second Einstein radius.
        """
        super(EllipticalIsothermal, self).__init__(centre=centre, axis_ratio=axis_ratio,
                                                   phi=phi,
                                                   einstein_radius=einstein_radius,
                                                   slope=2.0)

    # @classmethod
    # def from_mass_in_solar_masses(cls, redshift_lens=0.5, redshift_source=1.0, centre: units.Position = (0.0, 0.0), axis_ratio_=0.9,
    #                               phi: float = 0.0, mass=10e10):
    #
    #     return self.constant_kpc * self.angular_diameter_distance_of_plane_to_earth(j) / \
    #            (self.angular_diameter_distance_between_planes(i, j) *
    #             self.angular_diameter_distance_of_plane_to_earth(i))

    # critical_surface_density =

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore, \
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        factor = 2.0 * self.einstein_radius_rescaled * self.axis_ratio / np.sqrt(
            1 - self.axis_ratio ** 2)

        psi = np.sqrt(np.add(np.multiply(self.axis_ratio ** 2, np.square(grid[:, 1])),
                             np.square(grid[:, 0])))

        deflection_y = np.arctanh(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio ** 2), grid[:, 0]), psi))
        deflection_x = np.arctan(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio ** 2), grid[:, 1]), psi))
        return self.rotate_grid_from_profile(
            np.multiply(factor, np.vstack((deflection_y, deflection_x)).T))


class SphericalIsothermal(EllipticalIsothermal):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 einstein_radius: dim.Length = 1.0):
        """
        Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius.
        """
        super(SphericalIsothermal, self).__init__(centre=centre, axis_ratio=1.0,
                                                  phi=0.0,
                                                  einstein_radius=einstein_radius)

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        eta = self.grid_to_elliptical_radii(grid)
        return 2.0 * self.einstein_radius_rescaled * eta

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        return self.grid_to_grid_cartesian(grid=grid,
                                           radius=np.full(grid.shape[0],
                                                          2.0 * self.einstein_radius_rescaled))


# noinspection PyAbstractClass
class AbstractEllipticalGeneralizedNFW(EllipticalMassProfile, MassProfile):
    epsrel = 1.49e-5

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 kappa_s: float = 0.05,
                 inner_slope: float = 1.0,
                 scale_radius: dim.Length = 1.0):
        """
        The elliptical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis.
        kappa_s : float
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        inner_slope : float
            The inner slope of the dark matter halo
        scale_radius : float
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super(AbstractEllipticalGeneralizedNFW, self).__init__(centre=centre,
                                                               axis_ratio=axis_ratio,
                                                               phi=phi)
        super(MassProfile, self).__init__()
        self.kappa_s = kappa_s
        self.scale_radius = scale_radius
        self.inner_slope = inner_slope

    def tabulate_integral(self, grid, tabulate_bins):
        """Tabulate an integral over the surface density of deflection potential of a mass profile. This is used in \
        the GeneralizedNFW profile classes to speed up the integration procedure.

        Parameters
        -----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the potential / deflection_stacks are computed on.
        tabulate_bins : int
            The number of bins to tabulate the inner integral of this profile.
        """
        eta_min = 1.0e-4
        eta_max = 1.05 * np.max(self.grid_to_elliptical_radii(grid))

        minimum_log_eta = np.log10(eta_min)
        maximum_log_eta = np.log10(eta_max)
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)

        return eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def convergence_from_grid(self, grid):
        """ Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the surface density is computed on.
        """

        surface_density_grid = np.zeros(shape=grid.shape[0])

        grid_eta = self.grid_to_elliptical_radii(grid)

        for i in range(grid.shape[0]):
            surface_density_grid[i] = self.convergence_func(grid_eta[i])

        return surface_density_grid

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio) / 2.0)

    @staticmethod
    def coord_func_f(grid_radius):
        f = np.where(np.real(grid_radius) > 1.0,
                     (1.0 / np.sqrt(np.square(grid_radius) - 1.0)) * np.arccos(
                         np.divide(1.0, grid_radius)),
                     (1.0 / np.sqrt(1.0 - np.square(grid_radius))) * np.arccosh(
                         np.divide(1.0, grid_radius)))
        f[np.isnan(f)] = 1.0
        return f

    def coord_func_g(self, grid_radius):
        f_r = self.coord_func_f(grid_radius=grid_radius)

        g = np.where(np.real(grid_radius) > 1.0,
                     (1.0 - f_r) / (np.square(grid_radius) - 1.0),
                     (f_r - 1.0) / (1.0 - np.square(grid_radius)))
        g[np.isnan(g)] = 1.0 / 3.0
        return g

    def coord_func_h(self, grid_radius):
        return np.log(grid_radius / 2.0) + self.coord_func_f(grid_radius=grid_radius)

    @dim.convert_units_to_input_units
    def rho_at_scale_radius_for_units(self, redshift_profile, redshift_source,
                                      unit_length='arcsec', unit_mass='solMass',
                                      cosmology=cosmo.Planck15, **kwargs):
        """The Cosmic average density is defined at the redshift of the profile."""

        kpc_per_arcsec = kwargs[
            'kpc_per_arcsec'] if 'kpc_per_arcsec' in kwargs else None
        critical_surface_density = kwargs[
            'critical_surface_density'] if 'critical_surface_density' in kwargs else None

        rho_at_scale_radius = self.kappa_s * critical_surface_density / self.scale_radius

        rho_at_scale_radius = dim.MassOverLength3(value=rho_at_scale_radius,
                                                  unit_length=unit_length,
                                                  unit_mass=unit_mass)

        return rho_at_scale_radius.convert(unit_length=unit_length, unit_mass=unit_mass,
                                           kpc_per_arcsec=kpc_per_arcsec,
                                           critical_surface_density=critical_surface_density)

    @dim.convert_units_to_input_units
    def delta_concentration_for_units(self, redshift_profile, redshift_source,
                                      unit_length='arcsec', unit_mass='solMass',
                                      redshift_of_cosmic_average_density='profile',
                                      cosmology=cosmo.Planck15, **kwargs):
        cosmic_average_density = kwargs[
            'cosmic_average_density'] if 'cosmic_average_density' in kwargs else None

        rho_scale_radius = self.rho_at_scale_radius_for_units(unit_length=unit_length,
                                                              unit_mass=unit_mass,
                                                              redshift_profile=redshift_profile,
                                                              redshift_source=redshift_source,
                                                              cosmology=cosmology,
                                                              kwargs=kwargs)

        return rho_scale_radius / cosmic_average_density

    @dim.convert_units_to_input_units
    def concentration_for_units(self, redshift_profile, redshift_source,
                                unit_length='arcsec', unit_mass='solMass',
                                redshift_of_cosmic_average_density='profile',
                                cosmology=cosmo.Planck15, **kwargs):
        delta_concentration = self.delta_concentration_for_units(
            redshift_profile=redshift_profile, redshift_source=redshift_source,
            unit_length=unit_length,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            unit_mass=unit_mass,
            cosmology=cosmology, kwargs=kwargs)

        return \
            fsolve(func=self.concentration_func, x0=10.0, args=(delta_concentration,))[
                0]

    @staticmethod
    def concentration_func(concentration, delta_concentration):
        return 200.0 / 3.0 * (concentration * concentration * concentration /
                              (np.log(1 + concentration) - concentration / (
                                      1 + concentration))) - delta_concentration

    @dim.convert_units_to_input_units
    def radius_at_200_for_units(self, redshift_profile, redshift_source,
                                unit_length='arcsec', unit_mass='solMass',
                                redshift_of_cosmic_average_density='profile',
                                cosmology=cosmo.Planck15, **kwargs):
        kpc_per_arcsec = kwargs[
            'kpc_per_arcsec'] if 'kpc_per_arcsec' in kwargs else None

        concentration = self.concentration_for_units(redshift_profile=redshift_profile,
                                                     redshift_source=redshift_source,
                                                     unit_length=unit_length,
                                                     unit_mass=unit_mass,
                                                     redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
                                                     cosmology=cosmology, kwargs=kwargs)

        radius_at_200 = dim.Length(value=concentration * self.scale_radius,
                                   unit_length=unit_length)

        return radius_at_200.convert(unit_length=unit_length,
                                     kpc_per_arcsec=kpc_per_arcsec)

    @dim.convert_units_to_input_units
    def mass_at_200_for_units(self, redshift_profile, redshift_source,
                              unit_length='arcsec', unit_mass='solMass',
                              redshift_of_cosmic_average_density='profile',
                              cosmology=cosmo.Planck15, **kwargs):
        cosmic_average_density = kwargs[
            'cosmic_average_density'] if 'cosmic_average_density' in kwargs else None
        critical_surface_density = kwargs[
            'critical_surface_density'] if 'critical_surface_density' in kwargs else None

        radius_at_200 = self.radius_at_200_for_units(redshift_profile=redshift_profile,
                                                     redshift_source=redshift_source,
                                                     unit_length=unit_length,
                                                     unit_mass=unit_mass,
                                                     cosmology=cosmology,
                                                     redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
                                                     kwargs=kwargs)

        mass_at_200 = dim.Mass(
            200.0 * ((4.0 / 3.0) * np.pi) * cosmic_average_density * (
                    radius_at_200 ** 3.0),
            unit_mass=unit_mass)

        return mass_at_200.convert(unit_mass=unit_mass,
                                   critical_surface_density=critical_surface_density)

    @dim.convert_units_to_input_units
    def summarize_in_units(self, radii, prefix='', whitespace=80,
                           unit_length='arcsec', unit_mass='solMass',
                           redshift_profile=None, redshift_source=None,
                           redshift_of_cosmic_average_density='profile',
                           cosmology=cosmo.Planck15,
                           **kwargs):
        summary = super().summarize_in_units(
            radii=radii, prefix=prefix, unit_length=unit_length, unit_mass=unit_mass,
            redshift_profile=redshift_profile, redshift_source=redshift_source,
            cosmology=cosmology,
            whitespace=whitespace, kwargs=kwargs)

        rho_at_scale_radius = self.rho_at_scale_radius_for_units(
            radii=radii, unit_length=unit_length, unit_mass=unit_mass,
            redshift_profile=redshift_profile, redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology, kwargs=kwargs)

        summary += [af.text_util.label_value_and_unit_string(
            label=prefix + 'rho_at_scale_radius', value=rho_at_scale_radius,
            unit=unit_mass + '/' + unit_length + '3',
            whitespace=whitespace)]

        delta_concentration = self.delta_concentration_for_units(
            radii=radii, unit_length=unit_length, unit_mass=unit_mass,
            redshift_profile=redshift_profile, redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology, kwargs=kwargs)

        summary += [af.text_util.label_and_value_string(
            label=prefix + 'delta_concentration', value=delta_concentration,
            whitespace=whitespace)]

        concentration = self.concentration_for_units(
            radii=radii, unit_length=unit_length, unit_mass=unit_mass,
            redshift_profile=redshift_profile, redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology, kwargs=kwargs)

        summary += [af.text_util.label_and_value_string(
            label=prefix + 'concentration', value=concentration, whitespace=whitespace,
        )]

        radius_at_200 = self.radius_at_200_for_units(
            radii=radii, unit_length=unit_length, unit_mass=unit_mass,
            redshift_profile=redshift_profile, redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology, kwargs=kwargs)

        summary += [af.text_util.label_value_and_unit_string(
            label=prefix + 'radius_at_200x_cosmic_density', value=radius_at_200,
            unit=unit_length, whitespace=whitespace,
        )]

        mass_at_200 = self.mass_at_200_for_units(
            radii=radii, unit_length=unit_length, unit_mass=unit_mass,
            redshift_profile=redshift_profile, redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology, kwargs=kwargs)

        summary += [af.text_util.label_value_and_unit_string(
            label=prefix + 'mass_at_200x_cosmic_density', value=mass_at_200,
            unit=unit_mass, whitespace=whitespace,
        )]

        return summary

    @property
    def unit_mass(self):
        return 'angular'


class EllipticalGeneralizedNFW(AbstractEllipticalGeneralizedNFW):

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def potential_from_grid(self, grid, tabulate_bins=1000):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        tabulate_bins : int
            The number of bins to tabulate the inner integral of this profile.
        """

        @jit_integrand
        def deflection_integrand(x, kappa_radius, scale_radius, inner_slope):
            return (x + kappa_radius / scale_radius) ** (inner_slope - 3) * (
                    (1 - np.sqrt(1 - x ** 2)) / x)

        eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size = self.tabulate_integral(
            grid, tabulate_bins)

        potential_grid = np.zeros(grid.shape[0])

        deflection_integral = np.zeros((tabulate_bins,))

        for i in range(tabulate_bins):
            eta = 10. ** (minimum_log_eta + (i - 1) * bin_size)

            integral = \
                quad(deflection_integrand, a=0.0, b=1.0,
                     args=(eta, self.scale_radius, self.inner_slope),
                     epsrel=EllipticalGeneralizedNFW.epsrel)[0]

            deflection_integral[i] = ((eta / self.scale_radius) ** (
                    2 - self.inner_slope)) * (
                                             (1.0 / (3 - self.inner_slope)) *
                                             special.hyp2f1(3 - self.inner_slope,
                                                            3 - self.inner_slope,
                                                            4 - self.inner_slope,
                                                            - (
                                                                    eta / self.scale_radius)) + integral)

        for i in range(grid.shape[0]):
            potential_grid[i] = (2.0 * self.kappa_s * self.axis_ratio) * \
                                quad(self.potential_func, a=0.0, b=1.0,
                                     args=(grid[i, 0], grid[i, 1],
                                           self.axis_ratio, minimum_log_eta,
                                           maximum_log_eta, tabulate_bins,
                                           deflection_integral),
                                     epsrel=EllipticalGeneralizedNFW.epsrel)[0]

        return potential_grid

    @grids.grid_interpolate
    @geometry_profiles.cache
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid, tabulate_bins=1000):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        tabulate_bins : int
            The number of bins to tabulate the inner integral of this profile.
        """

        @jit_integrand
        def surface_density_integrand(x, kappa_radius, scale_radius, inner_slope):
            return (3 - inner_slope) * (x + kappa_radius / scale_radius) ** (
                    inner_slope - 4) * (1 - np.sqrt(1 - x * x))

        def calculate_deflection_component(npow, index):
            deflection_grid = 2.0 * self.kappa_s * self.axis_ratio * grid[:, index]
            deflection_grid *= quad_grid(self.deflection_func, 0.0, 1.0,
                                         grid, args=(
                    npow, self.axis_ratio, minimum_log_eta, maximum_log_eta,
                    tabulate_bins, surface_density_integral),
                                         epsrel=EllipticalGeneralizedNFW.epsrel)[0]

            return deflection_grid

        eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size = self.tabulate_integral(
            grid, tabulate_bins)

        surface_density_integral = np.zeros((tabulate_bins,))

        for i in range(tabulate_bins):
            eta = 10. ** (minimum_log_eta + (i - 1) * bin_size)

            integral = \
                quad(surface_density_integrand, a=0.0, b=1.0,
                     args=(eta, self.scale_radius,
                           self.inner_slope),
                     epsrel=EllipticalGeneralizedNFW.epsrel)[0]

            surface_density_integral[i] = ((eta / self.scale_radius) ** (
                    1 - self.inner_slope)) * \
                                          (((1 + eta / self.scale_radius) ** (
                                                  self.inner_slope - 3)) + integral)

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T))

    def convergence_func(self, radius):

        def integral_y(y, eta):
            return (y + eta) ** (self.inner_slope - 4) * (1 - np.sqrt(1 - y ** 2))

        radius = (1.0 / self.scale_radius) * radius
        integral_y = quad(integral_y, a=0.0, b=1.0, args=radius,
                          epsrel=EllipticalGeneralizedNFW.epsrel)[0]

        return 2.0 * self.kappa_s * (radius ** (1 - self.inner_slope)) * (
                (1 + radius) ** (self.inner_slope - 3) + (
                (3 - self.inner_slope) * integral_y))

    @staticmethod
    # TODO : Decorator needs to know that potential_integral is 1D array
    #    @jit_integrand
    def potential_func(u, y, x, axis_ratio, minimum_log_eta, maximum_log_eta,
                       tabulate_bins, potential_integral):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)
        i = 1 + int((np.log10(eta_u) - minimum_log_eta) / bin_size)
        r1 = 10. ** (minimum_log_eta + (i - 1) * bin_size)
        r2 = r1 * 10. ** bin_size
        phi = potential_integral[i] + (
                potential_integral[i + 1] - potential_integral[i]) * (
                      eta_u - r1) / (r2 - r1)
        return eta_u * (phi / u) / (1.0 - (1.0 - axis_ratio ** 2) * u) ** 0.5

    @staticmethod
    # TODO : Decorator needs to know that surface_density_integral is 1D array
    #    @jit_integrand
    def deflection_func(u, y, x, npow, axis_ratio, minimum_log_eta, maximum_log_eta,
                        tabulate_bins,
                        surface_density_integral):

        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)
        i = 1 + int((np.log10(eta_u) - minimum_log_eta) / bin_size)
        r1 = 10. ** (minimum_log_eta + (i - 1) * bin_size)
        r2 = r1 * 10. ** bin_size
        kap = surface_density_integral[i] + (
                surface_density_integral[i + 1] - surface_density_integral[i]) * (
                      eta_u - r1) / (r2 - r1)
        return kap / (1.0 - (1.0 - axis_ratio ** 2) * u) ** (npow + 0.5)


class SphericalGeneralizedNFW(EllipticalGeneralizedNFW):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 kappa_s: float = 0.05,
                 inner_slope: float = 1.0,
                 scale_radius: dim.Length = 1.0):
        """
        The spherical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        kappa_s : float
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        inner_slope : float
            The inner slope of the dark matter halo.
        scale_radius : float
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super(SphericalGeneralizedNFW, self).__init__(centre=centre, axis_ratio=1.0,
                                                      phi=0.0, kappa_s=kappa_s,
                                                      inner_slope=inner_slope,
                                                      scale_radius=scale_radius)

    @grids.grid_interpolate
    @geometry_profiles.cache
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        eta = np.multiply(1. / self.scale_radius, self.grid_to_grid_radii(grid))

        deflection_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            deflection_grid[i] = np.multiply(4. * self.kappa_s * self.scale_radius,
                                             self.deflection_func_sph(eta[i]))

        return self.grid_to_grid_cartesian(grid, deflection_grid)

    @staticmethod
    def deflection_integrand(y, eta, inner_slope):
        return (y + eta) ** (inner_slope - 3) * ((1 - np.sqrt(1 - y ** 2)) / y)

    def deflection_func_sph(self, eta):
        integral_y_2 = \
            quad(self.deflection_integrand, a=0.0, b=1.0, args=(eta, self.inner_slope),
                 epsrel=1.49e-6)[0]
        return eta ** (2 - self.inner_slope) * ((1.0 / (3 - self.inner_slope)) *
                                                special.hyp2f1(3 - self.inner_slope,
                                                               3 - self.inner_slope,
                                                               4 - self.inner_slope,
                                                               -eta) + integral_y_2)


class SphericalTruncatedNFW(AbstractEllipticalGeneralizedNFW):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 kappa_s: float = 0.05,
                 scale_radius: dim.Length = 1.0,
                 truncation_radius: dim.Length = 2.0):
        super(SphericalTruncatedNFW, self).__init__(centre=centre, axis_ratio=1.0,
                                                    phi=0.0, kappa_s=kappa_s,
                                                    inner_slope=1.0,
                                                    scale_radius=scale_radius)

        self.truncation_radius = truncation_radius
        self.tau = self.truncation_radius / self.scale_radius

    def coord_func_k(self, grid_radius):
        return np.log(np.divide(grid_radius,
                                np.sqrt(np.square(grid_radius) + np.square(self.tau)) +
                                self.tau))

    def coord_func_l(self, grid_radius):
        f_r = self.coord_func_f(grid_radius=grid_radius)
        g_r = self.coord_func_g(grid_radius=grid_radius)
        k_r = self.coord_func_k(grid_radius=grid_radius)

        return np.divide(self.tau ** 2.0, (self.tau ** 2.0 + 1.0) ** 2.0) * (
                ((self.tau ** 2.0 + 1.0) * g_r) +
                (2 * f_r) -
                (np.pi / (np.sqrt(self.tau ** 2.0 + grid_radius ** 2.0))) +
                (((self.tau ** 2.0 - 1.0) / (self.tau *
                                             (np.sqrt(
                                                 self.tau ** 2.0 + grid_radius ** 2.0)))) * k_r))

    def coord_func_m(self, grid_radius):
        f_r = self.coord_func_f(grid_radius=grid_radius)
        k_r = self.coord_func_k(grid_radius=grid_radius)

        return (self.tau ** 2.0 / (self.tau ** 2.0 + 1.0) ** 2.0) * (
                ((self.tau ** 2.0 + 2.0 * grid_radius ** 2.0 - 1.0) * f_r) +
                (np.pi * self.tau) +
                ((self.tau ** 2.0 - 1.0) * np.log(self.tau)) +
                (np.sqrt(grid_radius ** 2.0 + self.tau ** 2.0) * (
                        ((self.tau ** 2.0 - 1.0) / self.tau) * k_r - np.pi)))

    def convergence_func(self, grid_radius):
        grid_radius = ((1.0 / self.scale_radius) * grid_radius) + 0j
        return np.real(2.0 * self.kappa_s * self.coord_func_l(grid_radius=grid_radius))

    def deflection_func_sph(self, grid_radius):
        grid_radius = grid_radius + 0j
        return np.real(self.coord_func_m(grid_radius=grid_radius))

    def potential_from_grid(self, grid):
        return np.zeros((grid.shape[0],))

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        eta = np.multiply(1. / self.scale_radius, self.grid_to_grid_radii(grid))

        deflection_grid = np.multiply((4. * self.kappa_s * self.scale_radius / eta),
                                      self.deflection_func_sph(eta))

        return self.grid_to_grid_cartesian(grid, deflection_grid)

    @dim.convert_units_to_input_units
    def mass_at_truncation_radius(self, redshift_profile, redshift_source,
                                  unit_length='arcsec', unit_mass='solMass',
                                  redshift_of_cosmic_average_density='profile',
                                  cosmology=cosmo.Planck15, **kwargs):
        mass_at_200 = self.mass_at_200_for_units(redshift_profile=redshift_profile,
                                                 redshift_source=redshift_source,
                                                 unit_length=unit_length,
                                                 unit_mass=unit_mass,
                                                 redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
                                                 cosmology=cosmology, kwargs=kwargs)

        return mass_at_200 * (self.tau ** 2.0 / (self.tau ** 2.0 + 1.0) ** 2.0) * \
               (((self.tau ** 2.0 - 1) * np.log(self.tau)) + (self.tau * np.pi) - (
                       self.tau ** 2.0 + 1))

    @dim.convert_units_to_input_units
    def summarize_in_units(self, radii, prefix='', whitespace=80,
                           unit_length='arcsec', unit_mass='solMass',
                           redshift_profile=None, redshift_source=None,
                           redshift_of_cosmic_average_density='profile',
                           cosmology=cosmo.Planck15, **kwargs):
        summary = super().summarize_in_units(
            radii=radii, prefix=prefix, whitespace=whitespace,
            unit_length=unit_length, unit_mass=unit_mass,
            redshift_profile=redshift_profile, redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology, kwargs=kwargs)

        mass_at_truncation_radius = self.mass_at_truncation_radius(
            radii=radii, unit_length=unit_length, unit_mass=unit_mass,
            redshift_profile=redshift_profile, redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology, kwargs=kwargs)

        summary += [af.text_util.label_value_and_unit_string(
            label=prefix + 'mass_at_truncation_radius', value=mass_at_truncation_radius,
            unit=unit_mass,
            whitespace=whitespace)]

        return summary


class SphericalTruncatedNFWChallenge(SphericalTruncatedNFW):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 kappa_s: float = 0.05,
                 scale_radius: dim.Length = 1.0):
        def solve_c(c, de_c):
            '''
            Equation need for solving concentration c for a given delta_c
            '''
            return 200.0 / 3.0 * (c * c * c / (np.log(1 + c) - c / (1 + c))) - de_c

        kpc_per_arcsec = 6.68549148608755
        scale_radius_kpc = scale_radius * kpc_per_arcsec
        cosmic_average_density = 262.30319684750657
        critical_surface_density = 1940654909.413325
        rho_s = kappa_s * critical_surface_density / scale_radius_kpc
        de_c = rho_s / cosmic_average_density  # delta_c
        concentration = fsolve(solve_c, 10.0, args=(de_c,))[0]
        r200 = concentration * scale_radius_kpc / kpc_per_arcsec  # R_200

        super(SphericalTruncatedNFWChallenge, self).__init__(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            truncation_radius=2.0 * r200)

    @dim.convert_units_to_input_units
    def summarize_in_units(self, radii,
                           unit_length='arcsec', unit_mass='solMass',
                           redshift_profile=None, redshift_source=None,
                           redshift_of_cosmic_average_density='profile',
                           cosmology=cosmo.Planck15, **kwargs):
        summary = super().summarize_in_units(
            radii=radii, unit_length=unit_length, unit_mass=unit_mass,
            redshift_profile=redshift_profile, redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology, kwargs=kwargs)

        return summary


class EllipticalNFW(AbstractEllipticalGeneralizedNFW):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 kappa_s: float = 0.05,
                 scale_radius: dim.Length = 1.0):
        """
        The elliptical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis.
        kappa_s : float
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        scale_radius : float
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super(EllipticalNFW, self).__init__(centre=centre, axis_ratio=axis_ratio,
                                            phi=phi, kappa_s=kappa_s,
                                            inner_slope=1.0, scale_radius=scale_radius)

    @staticmethod
    def coord_func(r):
        if r > 1:
            return (1.0 / np.sqrt(r ** 2 - 1)) * np.arctan(np.sqrt(r ** 2 - 1))
        elif r < 1:
            return (1.0 / np.sqrt(1 - r ** 2)) * np.arctanh(np.sqrt(1 - r ** 2))
        elif r == 1:
            return 1

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        potential_grid = quad_grid(self.potential_func, 0.0, 1.0, grid,
                                   args=(
                                       self.axis_ratio, self.kappa_s,
                                       self.scale_radius),
                                   epsrel=1.49e-5)[0]

        return potential_grid

    @grids.grid_interpolate
    @geometry_profiles.cache
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        def calculate_deflection_component(npow, index):
            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= quad_grid(self.deflection_func, 0.0, 1.0, grid,
                                         args=(npow, self.axis_ratio, self.kappa_s,
                                               self.scale_radius))[0]

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T))

    def convergence_func(self, grid_radius):
        grid_radius = (1.0 / self.scale_radius) * grid_radius + 0j
        return np.real(2.0 * self.kappa_s * self.coord_func_g(grid_radius=grid_radius))

    @staticmethod
    def potential_func(u, y, x, axis_ratio, kappa_s, scale_radius):
        eta_u = (1.0 / scale_radius) * np.sqrt(
            (u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))

        if eta_u > 1:
            eta_u_2 = (1.0 / np.sqrt(eta_u ** 2 - 1)) * np.arctan(
                np.sqrt(eta_u ** 2 - 1))
        elif eta_u < 1:
            eta_u_2 = (1.0 / np.sqrt(1 - eta_u ** 2)) * np.arctanh(
                np.sqrt(1 - eta_u ** 2))
        else:
            eta_u_2 = 1

        return 4.0 * kappa_s * scale_radius * (axis_ratio / 2.0) * (eta_u / u) * (
                (np.log(eta_u / 2.0) + eta_u_2) / eta_u) / (
                       (1 - (1 - axis_ratio ** 2) * u) ** 0.5)

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, kappa_s, scale_radius):
        eta_u = (1.0 / scale_radius) * np.sqrt(
            (u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))

        if eta_u > 1:
            eta_u_2 = (1.0 / np.sqrt(eta_u ** 2 - 1)) * np.arctan(
                np.sqrt(eta_u ** 2 - 1))
        elif eta_u < 1:
            eta_u_2 = (1.0 / np.sqrt(1 - eta_u ** 2)) * np.arctanh(
                np.sqrt(1 - eta_u ** 2))
        else:
            eta_u_2 = 1

        return 2.0 * kappa_s * (1 - eta_u_2) / (eta_u ** 2 - 1) / (
                (1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))


class SphericalNFW(EllipticalNFW):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 kappa_s: float = 0.05,
                 scale_radius: dim.Length = 1.0):
        """
        The spherical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        kappa_s : float
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        scale_radius : float
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super(SphericalNFW, self).__init__(centre=centre, axis_ratio=1.0, phi=0.0,
                                           kappa_s=kappa_s,
                                           scale_radius=scale_radius)

    # TODO : The 'func' routines require a different input to the elliptical cases, meaning they cannot be overridden.
    # TODO : Should be able to refactor code to deal with this nicely, but will wait until we're clear on numba.

    # TODO : Make this use numpy arithmetic

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        eta = (1.0 / self.scale_radius) * self.grid_to_grid_radii(grid) + 0j
        return np.real(
            2.0 * self.scale_radius * self.kappa_s * self.potential_func_sph(eta))

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        eta = np.multiply(1. / self.scale_radius, self.grid_to_grid_radii(grid=grid))
        deflection_r = np.multiply(4. * self.kappa_s * self.scale_radius,
                                   self.deflection_func_sph(eta))

        return self.grid_to_grid_cartesian(grid, deflection_r)

    @staticmethod
    def potential_func_sph(eta):
        return ((np.log(eta / 2.0)) ** 2) - (np.arctanh(np.sqrt(1 - eta ** 2))) ** 2

    @staticmethod
    def deflection_func_sph(eta):
        conditional_eta = np.copy(eta)
        conditional_eta[eta > 1] = np.multiply(
            np.divide(1.0, np.sqrt(np.add(np.square(eta[eta > 1]), - 1))),
            np.arctan(np.sqrt(np.add(np.square(eta[eta > 1]), - 1))))
        conditional_eta[eta < 1] = np.multiply(
            np.divide(1.0, np.sqrt(np.add(1, - np.square(eta[eta < 1])))),
            np.arctanh(np.sqrt(np.add(1, - np.square(eta[eta < 1])))))

        return np.divide(np.add(np.log(np.divide(eta, 2.)), conditional_eta), eta)


# noinspection PyAbstractClass
class AbstractEllipticalSersic(EllipticalMassProfile):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 sersic_index: float = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0):
        """
        The Sersic mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens \
        model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The radius containing half the light of this profile.
        sersic_index : float
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(AbstractEllipticalSersic, self).__init__(centre=centre,
                                                       axis_ratio=axis_ratio, phi=phi)
        super(EllipticalMassProfile, self).__init__(centre=centre,
                                                    axis_ratio=axis_ratio, phi=phi)
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def convergence_from_grid(self, grid):
        """ Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the surface density is computed on.
        """
        return self.convergence_func(self.grid_to_eccentric_radii(grid))

    def convergence_func(self, radius):
        return self.mass_to_light_ratio * self.intensity_at_radius(radius)

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio) / 2.0)

    def intensity_at_radius(self, radius):
        """ Compute the intensity of the profile at a given radius.

        Parameters
        ----------
        radius : float
            The distance from the centre of the profile.
        """
        return self.intensity * np.exp(
            -self.sersic_constant * (((radius / self.effective_radius) ** (
                    1. / self.sersic_index)) - 1))

    @property
    def sersic_constant(self):
        """ A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
        total integrated light.
        """
        return (2 * self.sersic_index) - (1. / 3.) + (
                4. / (405. * self.sersic_index)) + (
                       46. / (25515. * self.sersic_index ** 2)) + (
                       131. / (1148175. * self.sersic_index ** 3)) - (
                       2194697. / (30690717750. * self.sersic_index ** 4))

    @property
    def elliptical_effective_radius(self):
        """The effective_radius of a Sersic light profile is defined as the circular effective radius. This is the \
        radius within which a circular aperture contains half the profiles's total integrated light. For elliptical \
        systems, this won't robustly capture the light profile's elliptical shape.

        The elliptical effective radius instead describes the major-axis radius of the ellipse containing \
        half the light, and may be more appropriate for highly flattened systems like disk galaxies."""
        return self.effective_radius / np.sqrt(self.axis_ratio)

    @property
    def unit_mass(self):
        return self.mass_to_light_ratio.unit_mass


class EllipticalSersic(AbstractEllipticalSersic):

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, intensity, sersic_index,
                        effective_radius, mass_to_light_ratio,
                        sersic_constant):
        eta_u = np.sqrt(axis_ratio) * np.sqrt(
            (u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))

        return mass_to_light_ratio * intensity * np.exp(
            -sersic_constant * (
                    ((eta_u / effective_radius) ** (1. / sersic_index)) - 1)) / (
                       (1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))

    @grids.grid_interpolate
    @geometry_profiles.cache
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        def calculate_deflection_component(npow, index):
            sersic_constant = self.sersic_constant

            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= quad_grid(self.deflection_func, 0.0, 1.0, grid,
                                         args=(npow, self.axis_ratio, self.intensity,
                                               self.sersic_index, self.effective_radius,
                                               self.mass_to_light_ratio,
                                               sersic_constant))[0]

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T))


class SphericalSersic(EllipticalSersic):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 sersic_index: float = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0):
        """
        The Sersic mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens
        model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : float
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profile.
        """
        super(SphericalSersic, self).__init__(centre=centre, axis_ratio=1.0, phi=0.0,
                                              intensity=intensity,
                                              effective_radius=effective_radius,
                                              sersic_index=sersic_index,
                                              mass_to_light_ratio=mass_to_light_ratio)


class EllipticalExponential(EllipticalSersic):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0):
        """
        The EllipticalExponential mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(EllipticalExponential, self).__init__(centre=centre,
                                                    axis_ratio=axis_ratio, phi=phi,
                                                    intensity=intensity,
                                                    effective_radius=effective_radius,
                                                    sersic_index=1.0,
                                                    mass_to_light_ratio=mass_to_light_ratio)


class SphericalExponential(EllipticalExponential):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0):
        """
        The Exponential mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens
        model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles.
        """
        super(SphericalExponential, self).__init__(centre=centre, axis_ratio=1.0,
                                                   phi=0.0, intensity=intensity,
                                                   effective_radius=effective_radius,
                                                   mass_to_light_ratio=mass_to_light_ratio)


class EllipticalDevVaucouleurs(EllipticalSersic):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0):
        """
        The EllipticalDevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The radius containing half the light of this profile.
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profile.
        """
        super(EllipticalDevVaucouleurs, self).__init__(centre=centre,
                                                       axis_ratio=axis_ratio, phi=phi,
                                                       intensity=intensity,
                                                       effective_radius=effective_radius,
                                                       sersic_index=4.0,
                                                       mass_to_light_ratio=mass_to_light_ratio)


class SphericalDevVaucouleurs(EllipticalDevVaucouleurs):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0):
        """
        The DevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit and subtract the
        lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles.
        """
        super(SphericalDevVaucouleurs, self).__init__(centre=centre, axis_ratio=1.0,
                                                      phi=0.0, intensity=intensity,
                                                      effective_radius=effective_radius,
                                                      mass_to_light_ratio=mass_to_light_ratio)


class EllipticalSersicRadialGradient(AbstractEllipticalSersic):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 sersic_index: float = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
                 mass_to_light_gradient: float = 0.0):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : float
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profile.
        mass_to_light_gradient : float
            The mass-to-light radial gradient.
        """
        super(EllipticalSersicRadialGradient, self).__init__(centre=centre,
                                                             axis_ratio=axis_ratio,
                                                             phi=phi,
                                                             intensity=intensity,
                                                             effective_radius=effective_radius,
                                                             sersic_index=sersic_index,
                                                             mass_to_light_ratio=mass_to_light_ratio)
        self.mass_to_light_gradient = mass_to_light_gradient

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def convergence_from_grid(self, grid):
        """ Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the surface density is computed on.
        """
        return self.convergence_func(self.grid_to_eccentric_radii(grid))

    @grids.grid_interpolate
    @geometry_profiles.cache
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        def calculate_deflection_component(npow, index):
            sersic_constant = self.sersic_constant

            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= quad_grid(self.deflection_func, 0.0, 1.0, grid,
                                         args=(npow, self.axis_ratio, self.intensity,
                                               self.sersic_index, self.effective_radius,
                                               self.mass_to_light_ratio,
                                               self.mass_to_light_gradient,
                                               sersic_constant))[0]
            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T))

    def convergence_func(self, radius):
        return (self.mass_to_light_ratio * (
                ((self.axis_ratio *
                  radius) /
                 self.effective_radius) ** -self.mass_to_light_gradient) * self.intensity_at_radius(
            radius))

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, intensity, sersic_index,
                        effective_radius, mass_to_light_ratio,
                        mass_to_light_gradient, sersic_constant):
        eta_u = np.sqrt(axis_ratio) * np.sqrt(
            (u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))

        return mass_to_light_ratio * (
                ((
                         axis_ratio * eta_u) / effective_radius) ** -mass_to_light_gradient) * intensity * np.exp(
            -sersic_constant * (
                    ((eta_u / effective_radius) ** (1. / sersic_index)) - 1)) / (
                       (1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))


class SphericalSersicRadialGradient(EllipticalSersicRadialGradient):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 sersic_index: float = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
                 mass_to_light_gradient: float = 0.0):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : float
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profile.
        mass_to_light_gradient : float
            The mass-to-light radial gradient.
        """
        super(SphericalSersicRadialGradient, self).__init__(
            centre=centre,
            axis_ratio=1.0, phi=0.0,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient
        )


class MassSheet(geometry_profiles.SphericalProfile, MassProfile):

    @af.map_types
    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 kappa: float = 0.0):
        """
        Represents a mass-sheet

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        kappa : float
            The magnitude of the convergence of the mass-sheet.
        """
        super(MassSheet, self).__init__(centre=centre)
        self.kappa = kappa

    def convergence_from_grid(self, grid):
        return np.full(shape=grid.shape[0], fill_value=self.kappa)

    def potential_from_grid(self, grid):
        return np.zeros((grid.shape[0],))

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return self.grid_to_grid_cartesian(grid=grid, radius=self.kappa * grid_radii)


# noinspection PyAbstractClass
class ExternalShear(geometry_profiles.EllipticalProfile, MassProfile):

    @af.map_types
    def __init__(self,
                 magnitude: float = 0.2,
                 phi: float = 0.0):
        """
        An external shear term, to model the line-of-sight contribution of other galaxies / satellites.

        The shear angle phi is defined in the direction of stretching of the image. Therefore, if an object located \
        outside the lens is responsible for the shear, it will be offset 90 degrees from the value of phi.

        Parameters
        ----------
        magnitude : float
            The overall magnitude of the shear (gamma).
        phi : float
            The rotation axis of the shear.
        """

        super(ExternalShear, self).__init__(centre=(0.0, 0.0), phi=phi, axis_ratio=1.0)
        self.magnitude = magnitude

    def einstein_radius_in_units(self, unit_mass='solMass',
                                 redshift_profile=None, cosmology=cosmo.Planck15,
                                 **kwargs):
        return 0.0

    def einstein_mass_in_units(self, unit_mass='solMass',
                               redshift_profile=None, redshift_source=None,
                               cosmology=cosmo.Planck15, **kwargs):
        return 0.0

    def convergence_from_grid(self, grid):
        return np.zeros((grid.shape[0],))

    def potential_from_grid(self, grid):
        return np.zeros((grid.shape[0],))

    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        deflection_y = -np.multiply(self.magnitude, grid[:, 0])
        deflection_x = np.multiply(self.magnitude, grid[:, 1])
        return self.rotate_grid_from_profile(np.vstack((deflection_y, deflection_x)).T)

import inspect

import numba
import numpy as np
from numba import cfunc
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable
from scipy import special
from scipy.integrate import quad

from autolens.profiles import geometry_profiles
from autolens.profiles import light_profiles


def jit_integrand(integrand_function):
    jitted_function = numba.jit(integrand_function, nopython=True, cache=True)
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
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7])
    elif no_args == 9:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7], xx[8])
    elif no_args == 10:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7], xx[8], xx[9])
    elif no_args == 11:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7], xx[8], xx[9], xx[10])

    cf = cfunc(float64(intc, CPointer(float64)))

    return LowLevelCallable(cf(wrapped).ctypes)


class MassProfile(object):

    def surface_density_func(self, eta):
        raise NotImplementedError("surface_density_at_radius should be overridden")

    def surface_density_from_grid(self, grid):
        pass
        # raise NotImplementedError("surface_density_from_grid should be overridden")

    def potential_from_grid(self, grid):
        pass
        # raise NotImplementedError("potential_from_grid should be overridden")

    def deflections_from_grid(self, grid):
        raise NotImplementedError("deflections_from_grid should be overridden")

    def dimensionless_mass_within_ellipse(self, major_axis):
        raise NotImplementedError()


# noinspection PyAbstractClass
class EllipticalMassProfile(geometry_profiles.EllipticalProfile, MassProfile):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The centre of the profile
        axis_ratio : float
            Ellipse's minor-to-major axis ratio (b/a)
        phi : float
            Rotation angle of profile's ellipse counter-clockwise from positive x-axis
        """
        super(EllipticalMassProfile, self).__init__(centre, axis_ratio, phi)
        self.axis_ratio = axis_ratio
        self.phi = phi

    def dimensionless_mass_within_circle(self, radius):
        """ Compute the mass profiles's total dimensionless mass within a circle of specified radius. This is \ 
        performed via integration of the surface density profiles and is centred on the mass profile.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.

        Returns
        -------
        dimensionless_mass : float
            The total dimensionless mass within the specified circle.
        """
        return quad(self.dimensionless_mass_integral, a=0.0, b=radius, args=(1.0,))[0]

    def dimensionless_mass_within_ellipse(self, major_axis):
        """ Compute the mass profiles's total dimensionless mass within an ellipse of specified radius. This is \
        performed via integration of the surface density profiles and is centred and rotationally aligned with the \
        mass profile.

        Parameters
        ----------
        major_axis : float
            The major-axis radius of the ellipse.

        Returns
        -------
        dimensionless_mass : float
            The total dimensionless mass within the specified circle.
        """
        return quad(self.dimensionless_mass_integral, a=0.0, b=major_axis, args=(self.axis_ratio,))[0]

    def dimensionless_mass_integral(self, x, axis_ratio):
        """Routine to integrate an elliptical light profiles - set axis ratio to 1 to compute the luminosity within a \
        circle"""
        r = x * axis_ratio
        return 2 * np.pi * r * self.surface_density_func(x)

    def mass_within_circle(self, radius, conversion_factor):
        """ Compute the mass profiles's total mass within a circle of specified radius. This is performed via
        integration of the surface density profiles and is centred on the mass profile.

        The result is multiplied by a conversion factor to compute the physical mass of the system. For strong lensing,
        this factor is the critical surface mass density.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.
        conversion_factor : float
            The factor the dimensionless mass is multiplied by to convert it to a physical mass.
        """
        return conversion_factor*self.dimensionless_mass_within_circle(radius=radius)

    def mass_within_ellipse(self, major_axis, conversion_factor):
        """ Compute the mass profiles's total mass within an ellipse of specified radius. This is performed via
        integration of the surface density profiles and is centred and rotationally aligned with the mass profile.

        The result is multiplied by a conversion factor to compute the physical mass of the system. For strong lensing,
        this factor is the critical surface mass density.

        Parameters
        ----------
        major_axis : float
            The major-axis radius of the ellipse.
        conversion_factor : float
            The factor the dimensionless mass is multiplied by to convert it to a physical mass.
        """
        return conversion_factor*self.dimensionless_mass_within_ellipse(major_axis=major_axis)

class EllipticalCoredPowerLaw(EllipticalMassProfile, MassProfile):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0, slope=2.0, core_radius=0.01):
        """
        Represents a cored elliptical power-law density distribution

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Elliptical mass profile's minor-to-major axis ratio (b/a)
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profiles
        slope : float
            power-law density slope of mass profiles
        core_radius : float
            The radius of the inner core
        """
        super(EllipticalCoredPowerLaw, self).__init__(centre, axis_ratio, phi)
        self.einstein_radius = einstein_radius
        self.slope = slope
        self.core_radius = core_radius

    @property
    def einstein_radius_rescaled(self):
        """Rescale the einstein radius by slope and axis_ratio, to reduce its degeneracy with other mass-profiles
        parameters"""
        return ((3 - self.slope) / (1 + self.axis_ratio)) * self.einstein_radius ** (self.slope - 1)

    @geometry_profiles.transform_grid
    def surface_density_from_grid(self, grid):
        """ Calculate the projected surface density in dimensionless units at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the surface density is computed on.
        """

        surface_density_grid = np.zeros(grid.shape[0])

        grid_eta = self.grid_to_elliptical_radii(grid)

        for i in range(grid.shape[0]):
            surface_density_grid[i] = self.surface_density_func(grid_eta[i])

        return surface_density_grid

    @geometry_profiles.transform_grid
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """
        potential_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            potential_grid[i] = quad(self.potential_func, a=0.0, b=1.0,
                                     args=(grid[i, 0], grid[i, 1], self.axis_ratio, self.slope, self.core_radius))[0]

        return self.einstein_radius_rescaled * self.axis_ratio * potential_grid

    @geometry_profiles.transform_grid
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """

        def calculate_deflection_component(npow, index):
            deflection_grid = np.zeros(grid.shape[0])

            einstein_radius_rescaled = self.einstein_radius_rescaled

            for i in range(grid.shape[0]):
                deflection_grid[i] = self.axis_ratio * grid[i, index] * quad(self.deflection_func, a=0.0, b=1.0,
                                                                             args=(grid[i, 0], grid[i, 1], npow,
                                                                                   self.axis_ratio,
                                                                                   einstein_radius_rescaled, self.slope,
                                                                                   self.core_radius))[0]

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T))

    def surface_density_func(self, radius):
        return self.einstein_radius_rescaled * (self.core_radius ** 2 + radius ** 2) ** (-(self.slope - 1) / 2.0)

    @staticmethod
    @jit_integrand
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        eta = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (eta / u) * ((3.0 - slope) * eta) ** -1.0 * \
               ((core_radius ** 2 + eta ** 2) ** ((3.0 - slope) / 2.0) -
                core_radius ** (3 - slope)) / ((1 - (1 - axis_ratio ** 2) * u) ** 0.5)

    @staticmethod
    @jit_integrand
    def deflection_func(u, y, x, npow, axis_ratio, einstein_radius_rescaled, slope, core_radius):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return einstein_radius_rescaled * (core_radius ** 2 + eta_u ** 2) ** (-(slope - 1) / 2.0) / (
                (1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))


class SphericalCoredPowerLaw(EllipticalCoredPowerLaw):

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0, core_radius=0.0):
        """
        Represents a cored spherical power-law density distribution

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        einstein_radius : float
            Einstein radius of power-law mass profiles
        slope : float
            power-law density slope of mass profiles
        core_radius : float
            The radius of the inner core
        """
        super(SphericalCoredPowerLaw, self).__init__(centre, 1.0, 0.0, einstein_radius, slope, core_radius)

    @geometry_profiles.transform_grid
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """
        eta = self.grid_to_radius(grid)
        deflection = np.multiply(2. * self.einstein_radius_rescaled, np.divide(
            np.add(np.power(np.add(self.core_radius ** 2, np.square(eta)), (3. - self.slope) / 2.),
                   -self.core_radius ** (3 - self.slope)), np.multiply((3. - self.slope), eta)))
        return self.grid_radius_to_cartesian(grid, deflection)


class EllipticalPowerLaw(EllipticalCoredPowerLaw):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0, slope=2.0):
        """
        Represents an elliptical power-law density distribution.

        Parameters
        ----------
        centre: (float, float)
            The (x,y) coordinates of the centre of the profile.
        axis_ratio : float
            Elliptical mass profile's minor-to-major axis ratio (b/a)
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profile.
        slope : float
            power-law density slope of mass profile.
        """

        super(EllipticalPowerLaw, self).__init__(centre, axis_ratio, phi, einstein_radius, slope, 0.0)

    def surface_density_func(self, radius):
        if radius > 0.0:
            return self.einstein_radius_rescaled * radius ** (-(self.slope - 1))
        else:
            return np.inf

    @staticmethod
    @jit_integrand
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (eta_u / u) * ((3.0 - slope) * eta_u) ** -1.0 * eta_u ** (3.0 - slope) / \
               ((1 - (1 - axis_ratio ** 2) * u) ** 0.5)

    @staticmethod
    @jit_integrand
    def deflection_func(u, y, x, npow, axis_ratio, einstein_radius_rescaled, slope, core_radius):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return einstein_radius_rescaled * eta_u ** (-(slope - 1)) / ((1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))


class SphericalPowerLaw(EllipticalPowerLaw):

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0):
        """
        Represents a spherical power-law density distribution.

        Parameters
        ----------
        centre: (float, float)
            The (x,y) coordinates of the centre of the profile.
        einstein_radius : float
            Einstein radius of power-law mass profiles
        slope : float
            power-law density slope of mass profiles
        """

        super(SphericalPowerLaw, self).__init__(centre, 1.0, 0.0, einstein_radius, slope)

    @geometry_profiles.transform_grid
    def deflections_from_grid(self, grid):
        eta = self.grid_to_radius(grid)
        deflection_r = 2.0 * self.einstein_radius_rescaled * np.divide(np.power(eta, (3.0 - self.slope)),
                                                                       np.multiply((3.0 - self.slope), eta))
        return self.grid_radius_to_cartesian(grid, deflection_r)


class EllipticalCoredIsothermal(EllipticalCoredPowerLaw):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0, core_radius=0.05):
        """
        Represents a cored elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope=2.0

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Elliptical mass profile's minor-to-major axis ratio (b/a)
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profiles
        core_radius : float
            The radius of the inner core
        """

        super(EllipticalCoredIsothermal, self).__init__(centre, axis_ratio, phi, einstein_radius, 2.0,
                                                        core_radius)


class SphericalCoredIsothermal(SphericalCoredPowerLaw):

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0, core_radius=0.05):
        """
        Represents a cored spherical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope=2.0

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        einstein_radius : float
            Einstein radius of power-law mass profiles
        core_radius : float
            The radius of the inner core
        """

        super(SphericalCoredIsothermal, self).__init__(centre, einstein_radius, 2.0, core_radius)


class EllipticalIsothermal(EllipticalPowerLaw):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=0.9, phi=0.0, einstein_radius=1.0):
        """
        Represents an elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope=2.0

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Elliptical mass profile's minor-to-major axis ratio (b/a)
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profiles
        """

        super(EllipticalIsothermal, self).__init__(centre, axis_ratio, phi, einstein_radius, 2.0)

    @geometry_profiles.transform_grid
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """

        np.seterr(all='ignore')

        try:
            factor = 2.0 * self.einstein_radius_rescaled * self.axis_ratio / np.sqrt(1 - self.axis_ratio ** 2)

            psi = np.sqrt(np.add(np.multiply(self.axis_ratio ** 2, np.square(grid[:, 1])), np.square(grid[:, 0])))

            deflection_y = np.arctanh(np.divide(np.multiply(np.sqrt(1 - self.axis_ratio ** 2), grid[:, 0]), psi))
            deflection_x = np.arctan(np.divide(np.multiply(np.sqrt(1 - self.axis_ratio ** 2), grid[:, 1]), psi))

            return self.rotate_grid_from_profile(np.multiply(factor, np.vstack((deflection_y, deflection_x)).T))
        except ZeroDivisionError:
            return self.grid_radius_to_cartesian(grid, np.full(grid.shape[0], 2.0 * self.einstein_radius_rescaled))


class SphericalIsothermal(EllipticalIsothermal):

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0):
        """
        Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
        density distribution for the value slope=2.0

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        einstein_radius : float
            Einstein radius of power-law mass profiles
        """

        super(SphericalIsothermal, self).__init__(centre, 1.0, 0.0, einstein_radius)

    @geometry_profiles.transform_grid
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """
        eta = self.grid_to_elliptical_radii(grid)
        return 2.0 * self.einstein_radius_rescaled * eta

    @geometry_profiles.transform_grid
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """
        return self.grid_radius_to_cartesian(grid, np.full(grid.shape[0], 2.0 * self.einstein_radius_rescaled))


# noinspection PyAbstractClass
class AbstractEllipticalGeneralizedNFW(geometry_profiles.EllipticalProfile, MassProfile):
    epsrel = 1.49e-5

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, kappa_s=0.05, inner_slope=1.0, scale_radius=5.0):
        """
        The elliptical NFW profiles, used to fit_normal the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        kappa_s : float
            The overall normalization of the dark matter halo
        inner_slope : float
            The inner slope of the dark matter halo
        scale_radius : float
            The radius containing half the light of this model_mapper
        """

        super(AbstractEllipticalGeneralizedNFW, self).__init__(centre, axis_ratio, phi)
        super(MassProfile, self).__init__()
        self.kappa_s = kappa_s
        self.scale_radius = scale_radius
        self.inner_slope = inner_slope

    def tabulate_integral(self, grid, tabulate_bins):
        """Tabulate an integral over the surface density of deflection potential of a mass profile. This is used in \
        the GeneralizedNFW profile classes to speed up the integration procedure.

        Parameters
        -----------
        grid : masks.ImageGrid
            The grid of coordinates the potential / deflections are computed on.
        tabulate_bins : int
            The number of bins used to tabulate the integral over.
        """
        eta_min = 1.0e-4
        eta_max = 1.05 * np.max(self.grid_to_elliptical_radii(grid))

        minimum_log_eta = np.log10(eta_min)
        maximum_log_eta = np.log10(eta_max)
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)

        return eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size

    @geometry_profiles.transform_grid
    def surface_density_from_grid(self, grid):
        """ Calculate the projected surface density in dimensionless units at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the surface density is computed on.
        """

        surface_density_grid = np.zeros(grid.shape[0])

        grid_eta = self.grid_to_elliptical_radii(grid)

        for i in range(grid.shape[0]):
            surface_density_grid[i] = self.surface_density_func(grid_eta[i])

        return surface_density_grid


class EllipticalGeneralizedNFW(AbstractEllipticalGeneralizedNFW):

    @geometry_profiles.transform_grid
    def potential_from_grid(self, grid, tabulate_bins=1000):
        """
        Calculate the potential at a given set of gridded coordinates.

        Parameters
        ----------
        tabulate_bins
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """

        @jit_integrand
        def deflection_integrand(x, kappa_radius, scale_radius, inner_slope):
            return (x + kappa_radius / scale_radius) ** (inner_slope - 3) * ((1 - np.sqrt(1 - x ** 2)) / x)

        eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size = self.tabulate_integral(grid, tabulate_bins)

        potential_grid = np.zeros(grid.shape[0])

        deflection_integral = np.zeros((tabulate_bins,))

        for i in range(tabulate_bins):
            eta = 10. ** (minimum_log_eta + (i - 1) * bin_size)

            integral = \
                quad(deflection_integrand, a=0.0, b=1.0, args=(eta, self.scale_radius, self.inner_slope),
                     epsrel=EllipticalGeneralizedNFW.epsrel)[0]

            deflection_integral[i] = ((eta / self.scale_radius) ** (2 - self.inner_slope)) * (
                    (1.0 / (3 - self.inner_slope)) *
                    special.hyp2f1(3 - self.inner_slope, 3 - self.inner_slope, 4 - self.inner_slope,
                                   - (eta / self.scale_radius)) + integral)

        for i in range(grid.shape[0]):
            potential_grid[i] = (2.0 * self.kappa_s * self.axis_ratio) * \
                                quad(self.potential_func, a=0.0, b=1.0, args=(grid[i, 0], grid[i, 1],
                                                                              self.axis_ratio, minimum_log_eta,
                                                                              maximum_log_eta, tabulate_bins,
                                                                              deflection_integral),
                                     epsrel=EllipticalGeneralizedNFW.epsrel)[0]

        return potential_grid

    @geometry_profiles.transform_grid
    def deflections_from_grid(self, grid, tabulate_bins=1000):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        tabulate_bins
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """

        @jit_integrand
        def surface_density_integrand(x, kappa_radius, scale_radius, inner_slope):
            return (3 - inner_slope) * (x + kappa_radius / scale_radius) ** (inner_slope - 4) * (1 - np.sqrt(1 - x * x))

        def calculate_deflection_component(npow, index):

            deflection_grid = np.zeros(grid.shape[0])

            for j in range(grid.shape[0]):
                coeff = 2.0 * self.kappa_s * self.axis_ratio * grid[j, index]
                deflection_grid[j] = coeff * quad(self.deflection_func, a=0.0, b=1.0, args=(
                    grid[j, 0],
                    grid[j, 1],
                    npow,
                    self.axis_ratio,
                    minimum_log_eta,
                    maximum_log_eta,
                    tabulate_bins,
                    surface_density_integral), epsrel=EllipticalGeneralizedNFW.epsrel)[0]

            return deflection_grid

        eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size = self.tabulate_integral(grid, tabulate_bins)

        surface_density_integral = np.zeros((tabulate_bins,))

        for i in range(tabulate_bins):
            eta = 10. ** (minimum_log_eta + (i - 1) * bin_size)

            integral = quad(surface_density_integrand, a=0.0, b=1.0, args=(eta, self.scale_radius,
                                                                           self.inner_slope),
                            epsrel=EllipticalGeneralizedNFW.epsrel)[0]

            surface_density_integral[i] = ((eta / self.scale_radius) ** (1 - self.inner_slope)) * \
                                          (((1 + eta / self.scale_radius) ** (self.inner_slope - 3)) + integral)

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T))

    def surface_density_func(self, radius):

        def integral_y(y, eta):
            return (y + eta) ** (self.inner_slope - 4) * (1 - np.sqrt(1 - y ** 2))

        radius = (1.0 / self.scale_radius) * radius
        integral_y = quad(integral_y, a=0.0, b=1.0, args=radius, epsrel=EllipticalGeneralizedNFW.epsrel)[0]

        return 2.0 * self.kappa_s * (radius ** (1 - self.inner_slope)) * (
                (1 + radius) ** (self.inner_slope - 3) + ((3 - self.inner_slope) * integral_y))

    @staticmethod
    # TODO : Decorator needs to know that potential_integral is 1D array
    #    @jit_integrand
    def potential_func(u, y, x, axis_ratio, minimum_log_eta, maximum_log_eta, tabulate_bins, potential_integral):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)
        i = 1 + int((np.log10(eta_u) - minimum_log_eta) / bin_size)
        r1 = 10. ** (minimum_log_eta + (i - 1) * bin_size)
        r2 = r1 * 10. ** bin_size
        phi = potential_integral[i] + (potential_integral[i + 1] - potential_integral[i]) * (eta_u - r1) / (r2 - r1)
        return eta_u * (phi / u) / (1.0 - (1.0 - axis_ratio ** 2) * u) ** 0.5

    @staticmethod
    # TODO : Decorator needs to know that surface_density_integral is 1D array
    #    @jit_integrand
    def deflection_func(u, y, x, npow, axis_ratio, minimum_log_eta, maximum_log_eta, tabulate_bins,
                        surface_density_integral):

        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)
        i = 1 + int((np.log10(eta_u) - minimum_log_eta) / bin_size)
        r1 = 10. ** (minimum_log_eta + (i - 1) * bin_size)
        r2 = r1 * 10. ** bin_size
        kap = surface_density_integral[i] + (surface_density_integral[i + 1] - surface_density_integral[i]) * (
                eta_u - r1) / (r2 - r1)
        return kap / (1.0 - (1.0 - axis_ratio ** 2) * u) ** (npow + 0.5)


class SphericalGeneralizedNFW(EllipticalGeneralizedNFW):

    def __init__(self, centre=(0.0, 0.0), kappa_s=0.05, inner_slope=1.0, scale_radius=5.0):
        """
        The spherical NFW profiles, used to fit_normal the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        kappa_s : float
            The overall normalization of the dark matter halo
        inner_slope : float
            The inner slope of the dark matter halo
        scale_radius : float
            The radius containing half the light of this model_mapper
        """

        super(SphericalGeneralizedNFW, self).__init__(centre, 1.0, 0.0, kappa_s, inner_slope, scale_radius)

    @geometry_profiles.transform_grid
    def deflections_from_grid(self, grid, **kwargs):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """

        eta = np.multiply(1. / self.scale_radius, self.grid_to_radius(grid))

        deflection_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            deflection_grid[i] = np.multiply(4. * self.kappa_s * self.scale_radius, self.deflection_func_sph(eta[i]))

        return self.grid_radius_to_cartesian(grid, deflection_grid)

    @staticmethod
    def deflection_integrand(y, eta, inner_slope):
        return (y + eta) ** (inner_slope - 3) * ((1 - np.sqrt(1 - y ** 2)) / y)

    def deflection_func_sph(self, eta):
        integral_y_2 = quad(self.deflection_integrand, a=0.0, b=1.0, args=(eta, self.inner_slope), epsrel=1.49e-6)[0]
        return eta ** (2 - self.inner_slope) * ((1.0 / (3 - self.inner_slope)) *
                                                special.hyp2f1(3 - self.inner_slope, 3 - self.inner_slope,
                                                               4 - self.inner_slope, -eta) + integral_y_2)


class EllipticalNFW(AbstractEllipticalGeneralizedNFW):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, kappa_s=0.05, scale_radius=5.0):
        """
        The elliptical NFW profiles, used to fit_normal the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        kappa_s : float
            The overall normalization of the dark matter halo
        scale_radius : float
            The radius containing half the light of this model_mapper
        """

        super(EllipticalNFW, self).__init__(centre, axis_ratio, phi, kappa_s, 1.0, scale_radius)

    @staticmethod
    def coord_func(r):
        if r > 1:
            return (1.0 / np.sqrt(r ** 2 - 1)) * np.arctan(np.sqrt(r ** 2 - 1))
        elif r < 1:
            return (1.0 / np.sqrt(1 - r ** 2)) * np.arctanh(np.sqrt(1 - r ** 2))
        elif r == 1:
            return 1

    @geometry_profiles.transform_grid
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """

        potential_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            potential_grid[i] = quad(self.potential_func, a=0.0, b=1.0,
                                     args=(grid[i, 0], grid[i, 1], self.axis_ratio, self.kappa_s, self.scale_radius),
                                     epsrel=1.49e-5)[0]

        return potential_grid

    @geometry_profiles.transform_grid
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """

        def calculate_deflection_component(npow, index):
            deflection_grid = np.zeros(grid.shape[0])

            for i in range(grid.shape[0]):
                deflection_grid[i] = self.axis_ratio * grid[i, index] * quad(self.deflection_func, a=0.0, b=1.0,
                                                                             args=(grid[i, 0], grid[i, 1], npow,
                                                                                   self.axis_ratio, self.kappa_s,
                                                                                   self.scale_radius))[0]

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T))

    def surface_density_func(self, radius):
        radius = (1.0 / self.scale_radius) * radius
        return 2.0 * self.kappa_s * (1 - self.coord_func(radius)) / (radius ** 2 - 1)

    @staticmethod
    @jit_integrand
    def potential_func(u, y, x, axis_ratio, kappa_s, scale_radius):

        eta_u = (1.0 / scale_radius) * np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))

        if eta_u > 1:
            eta_u_2 = (1.0 / np.sqrt(eta_u ** 2 - 1)) * np.arctan(np.sqrt(eta_u ** 2 - 1))
        elif eta_u < 1:
            eta_u_2 = (1.0 / np.sqrt(1 - eta_u ** 2)) * np.arctanh(np.sqrt(1 - eta_u ** 2))
        else:
            eta_u_2 = 1

        return 4.0 * kappa_s * scale_radius * (axis_ratio / 2.0) * (eta_u / u) * (
                (np.log(eta_u / 2.0) + eta_u_2) / eta_u) / (
                       (1 - (1 - axis_ratio ** 2) * u) ** 0.5)

    @staticmethod
    @jit_integrand
    def deflection_func(u, y, x, npow, axis_ratio, kappa_s, scale_radius):

        eta_u = (1.0 / scale_radius) * np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))

        if eta_u > 1:
            eta_u_2 = (1.0 / np.sqrt(eta_u ** 2 - 1)) * np.arctan(np.sqrt(eta_u ** 2 - 1))
        elif eta_u < 1:
            eta_u_2 = (1.0 / np.sqrt(1 - eta_u ** 2)) * np.arctanh(np.sqrt(1 - eta_u ** 2))
        else:
            eta_u_2 = 1

        return 2.0 * kappa_s * (1 - eta_u_2) / (eta_u ** 2 - 1) / ((1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))


class SphericalNFW(EllipticalNFW):

    def __init__(self, centre=(0.0, 0.0), kappa_s=0.05, scale_radius=5.0):
        """
        The spherical NFW profiles, used to fit_normal the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        kappa_s : float
            The overall normalization of the dark matter halo
        scale_radius : float
            The radius containing half the light of this model_mapper
        """

        super(SphericalNFW, self).__init__(centre, 1.0, 0.0, kappa_s, scale_radius)

    # TODO : The 'func' routines require a different input to the elliptical cases, meaning they cannot be overridden.
    # TODO : Should be able to refactor code to deal with this nicely, but will wait until we're clear on numba.

    # TODO : Make this use numpy arithmetic

    @geometry_profiles.transform_grid
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """
        eta = (1.0 / self.scale_radius) * self.grid_to_radius(grid)
        return 2.0 * self.scale_radius * self.kappa_s * self.potential_func_sph(eta)

    @geometry_profiles.transform_grid
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """
        eta = np.multiply(1. / self.scale_radius, self.grid_to_radius(grid))
        deflection_r = np.multiply(4. * self.kappa_s * self.scale_radius, self.deflection_func_sph(eta))

        return self.grid_radius_to_cartesian(grid, deflection_r)

    @staticmethod
    def potential_func_sph(eta):
        return ((np.log(eta / 2.0)) ** 2) - (np.arctanh(np.sqrt(1 - eta ** 2))) ** 2

    @staticmethod
    def deflection_func_sph(eta):
        conditional_eta = np.copy(eta)
        conditional_eta[eta > 1] = np.multiply(np.divide(1.0, np.sqrt(np.add(np.square(eta[eta > 1]), - 1))),
                                               np.arctan(np.sqrt(np.add(np.square(eta[eta > 1]), - 1))))
        conditional_eta[eta < 1] = np.multiply(np.divide(1.0, np.sqrt(np.add(1, - np.square(eta[eta < 1])))),
                                               np.arctanh(np.sqrt(np.add(1, - np.square(eta[eta < 1])))))

        return np.divide(np.add(np.log(np.divide(eta, 2.)), conditional_eta), eta)


# noinspection PyAbstractClass
class AbstractEllipticalSersic(light_profiles.AbstractEllipticalSersic, EllipticalMassProfile):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0, mass_to_light_ratio=1.0):
        """
        The Sersic mass profile, the mass profiles of the light profiles that are used to fit_normal and subtract the lens \
        model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        sersic_index : Int
            The concentration of the light profiles
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(AbstractEllipticalSersic, self).__init__(centre, axis_ratio, phi, intensity, effective_radius,
                                                       sersic_index)
        super(EllipticalMassProfile, self).__init__(centre, axis_ratio, phi)
        self.mass_to_light_ratio = mass_to_light_ratio

    @geometry_profiles.transform_grid
    def surface_density_from_grid(self, grid):
        """ Calculate the projected surface density in dimensionless units at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the surface density is computed on.
        """
        return self.surface_density_func(self.grid_to_eccentric_radii(grid))

    def surface_density_func(self, radius):
        return self.mass_to_light_ratio * self.intensity_at_radius(radius)


class EllipticalSersic(AbstractEllipticalSersic):
    @staticmethod
    @jit_integrand
    def deflection_func(u, y, x, npow, axis_ratio, intensity, sersic_index, effective_radius, mass_to_light_ratio,
                        sersic_constant):
        eta_u = np.sqrt(axis_ratio) * np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))

        return mass_to_light_ratio * intensity * np.exp(
            -sersic_constant * (((eta_u / effective_radius) ** (1. / sersic_index)) - 1)) / (
                       (1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))

    @geometry_profiles.transform_grid
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """

        def calculate_deflection_component(npow, index):
            deflection_grid = np.zeros(grid.shape[0])

            sersic_constant = self.sersic_constant

            for i in range(grid.shape[0]):
                deflection_grid[i] = self.axis_ratio * grid[i, index] * quad(self.deflection_func, a=0.0, b=1.0,
                                                                             args=(grid[i, 0], grid[i, 1], npow,
                                                                                   self.axis_ratio, self.intensity,
                                                                                   self.sersic_index,
                                                                                   self.effective_radius,
                                                                                   self.mass_to_light_ratio,
                                                                                   sersic_constant))[0]

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T))


class SphericalSersic(EllipticalSersic):

    def __init__(self, centre=(0.0, 0.0), intensity=0.1, effective_radius=0.6, sersic_index=4.0,
                 mass_to_light_ratio=1.0):
        """
        The Sersic mass profile, the mass profiles of the light profiles that are used to fit_normal and subtract the lens
        model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        sersic_index : Int
            The concentration of the light profiles
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(SphericalSersic, self).__init__(centre, 1.0, 0.0, intensity, effective_radius, sersic_index,
                                              mass_to_light_ratio)


class EllipticalExponential(EllipticalSersic):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 mass_to_light_ratio=1.0):
        """
        The EllipticalExponential mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(EllipticalExponential, self).__init__(centre, axis_ratio, phi, intensity, effective_radius, 1.0,
                                                    mass_to_light_ratio)

    @classmethod
    def from_exponential_light_profile(cls, exponential_light_profile, mass_to_light_ratio):
        return EllipticalExponential.from_profile(exponential_light_profile,
                                                  mass_to_light_ratio=mass_to_light_ratio)


class SphericalExponential(EllipticalExponential):

    def __init__(self, centre=(0.0, 0.0), intensity=0.1, effective_radius=0.6, mass_to_light_ratio=1.0):
        """
        The Exponential mass profile, the mass profiles of the light profiles that are used to fit_normal and subtract the lens
        model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(SphericalExponential, self).__init__(centre, 1.0, 0.0, intensity, effective_radius, mass_to_light_ratio)


class EllipticalDevVaucouleurs(EllipticalSersic):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 mass_to_light_ratio=1.0):
        """
        The EllipticalDevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(EllipticalDevVaucouleurs, self).__init__(centre, axis_ratio, phi, intensity, effective_radius, 4.0,
                                                       mass_to_light_ratio)

    @classmethod
    def from_dev_vaucouleurs_light_profile(cls, dev_vaucouleurs_light_profile, mass_to_light_ratio):
        return EllipticalDevVaucouleurs.from_profile(dev_vaucouleurs_light_profile,
                                                     mass_to_light_ratio=mass_to_light_ratio)


class SphericalDevVaucouleurs(EllipticalDevVaucouleurs):

    def __init__(self, centre=(0.0, 0.0), intensity=0.1, effective_radius=0.6, mass_to_light_ratio=1.0):
        """
        The DevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit_normal and subtract the
        lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(SphericalDevVaucouleurs, self).__init__(centre, 1.0, 0.0, intensity, effective_radius,
                                                      mass_to_light_ratio)


class EllipticalSersicRadialGradient(AbstractEllipticalSersic):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0, mass_to_light_ratio=1.0, mass_to_light_gradient=0.0):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre: (float, float)
            The centre of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        sersic_index : Int
            The concentration of the light profiles
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        mass_to_light_gradient : float
            The mass-to-light radial gradient.
        """
        super(EllipticalSersicRadialGradient, self).__init__(centre, axis_ratio, phi, intensity, effective_radius,
                                                             sersic_index, mass_to_light_ratio)
        self.mass_to_light_gradient = mass_to_light_gradient

    @geometry_profiles.transform_grid
    def surface_density_from_grid(self, grid):
        """ Calculate the projected surface density in dimensionless units at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the surface density is computed on.
        """
        return self.surface_density_func(self.grid_to_eccentric_radii(grid))

    @geometry_profiles.transform_grid
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """

        def calculate_deflection_component(npow, index):
            deflection_grid = np.zeros(grid.shape[0])

            sersic_constant = self.sersic_constant

            for i in range(grid.shape[0]):
                deflection_grid[i] = self.axis_ratio * grid[i, index] * quad(self.deflection_func, a=0.0, b=1.0,
                                                                             args=(grid[i, 0], grid[i, 1], npow,
                                                                                   self.axis_ratio, self.intensity,
                                                                                   self.sersic_index,
                                                                                   self.effective_radius,
                                                                                   self.mass_to_light_ratio,
                                                                                   self.mass_to_light_gradient,
                                                                                   sersic_constant))[0]

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T))

    def surface_density_func(self, radius):
        return (self.mass_to_light_ratio * (
                ((self.axis_ratio *
                  radius) /
                 self.effective_radius) ** -self.mass_to_light_gradient) * self.intensity_at_radius(radius))

    @staticmethod
    @jit_integrand
    def deflection_func(u, y, x, npow, axis_ratio, intensity, sersic_index, effective_radius, mass_to_light_ratio,
                        mass_to_light_gradient, sersic_constant):
        eta_u = np.sqrt(axis_ratio) * np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))

        return mass_to_light_ratio * (
                ((axis_ratio * eta_u) / effective_radius) ** -mass_to_light_gradient) * intensity * np.exp(
            -sersic_constant * (((eta_u / effective_radius) ** (1. / sersic_index)) - 1)) / (
                       (1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))


class SphericalSersicRadialGradient(EllipticalSersicRadialGradient):

    def __init__(self, centre=(0.0, 0.0), intensity=0.1, effective_radius=0.6, sersic_index=4.0,
                 mass_to_light_ratio=1.0, mass_to_light_gradient=0.0):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre: (float, float)
            The centre of the profiles
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        sersic_index : Int
            The concentration of the light profiles
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        mass_to_light_gradient : float
            The mass-to-light radial gradient.
        """
        super(SphericalSersicRadialGradient, self).__init__(centre, 1.0, 0.0, intensity, effective_radius,
                                                            sersic_index, mass_to_light_ratio, mass_to_light_gradient)


# noinspection PyAbstractClass
class ExternalShear(geometry_profiles.EllipticalProfile, MassProfile):

    def __init__(self, magnitude=0.2, phi=0.0):
        """
        An external shear term, to model the line-of-sight contribution of other galaxies / satellites.

        Parameters
        ----------
        magnitude : float
            The overall magnitude of the shear (gamma).
        phi : float
            The rotation axis of the shear.
        """

        super(ExternalShear, self).__init__(centre=(0.0, 0.0), phi=phi, axis_ratio=1.0)
        self.magnitude = magnitude

    def surface_density_from_grid(self, grid):
        return np.zeros((grid.shape[0],))

    def potential_from_grid(self, grid):
        return np.zeros((grid.shape[0],))

    @geometry_profiles.transform_grid
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """
        deflection_y = -np.multiply(self.magnitude, grid[:, 0])
        deflection_x = np.multiply(self.magnitude, grid[:, 1])
        return self.rotate_grid_from_profile(np.vstack((deflection_y, deflection_x)).T)

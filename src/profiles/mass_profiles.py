from src.profiles import geometry_profiles
from src.profiles import light_profiles
import math
from scipy.integrate import quad
from scipy import special
from itertools import count
import numpy as np


# noinspection PyMethodMayBeStatic
class MassProfile(object):

    def surface_density_at_radius(self, eta):
        raise NotImplementedError("Surface density at radius should be overridden")

    def surface_density_at_coordinates(self, coordinates):
        raise NotImplementedError("Surface density at image_grid should be overridden")

    def potential_at_coordinates(self, coordinates):
        raise NotImplementedError("Potential at image_grid should be overridden")

    def deflections_at_coordinates(self, coordinates):
        raise NotImplementedError("Deflection angles at image_grid should be overridden")


class EllipticalMassProfile(geometry_profiles.EllipticalProfile, MassProfile):
    """Generic class for an elliptical light profiles"""

    _ids = count()

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0):
        """

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        """
        super(EllipticalMassProfile, self).__init__(centre, axis_ratio, phi)
        self.axis_ratio = axis_ratio
        self.phi = phi
        self.component_number = next(self._ids)

    @property
    def subscript(self):
        return ''

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi']

    def dimensionless_mass_within_circle(self, radius):
        """
        Compute the mass profiles's total dimensionless mass within a circle of specified radius. This is performed via
        integration of the surface density profiles and is centred on the mass model_mapper.

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
        """
        Compute the mass profiles's total dimensionless mass within an ellipse of specified radius. This is performed
        via integration of the surface density profiles and is centred and rotationally aligned with the mass
        model_mapper.

        Parameters
        ----------
        major_axis

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
        return 2 * math.pi * r * self.surface_density_at_radius(x)


class EllipticalPowerLaw(EllipticalMassProfile, MassProfile):
    """Represents an elliptical power-law density distribution"""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0, slope=2.0):
        """

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of mass profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profiles ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profiles
        slope : float
            power-law density slope of mass profiles
        """

        super(EllipticalPowerLaw, self).__init__(centre, axis_ratio, phi)
        super(MassProfile, self).__init__()

        self.einstein_radius = einstein_radius
        self.slope = slope

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', r'\theta', r'\alpha']

    @property
    def einstein_radius_rescaled(self):
        """Rescale the einstein radius by slope and axis_ratio, to reduce its degeneracy with other mass-profiles
        parameters"""
        return ((3 - self.slope) / (1 + self.axis_ratio)) * self.einstein_radius ** (self.slope - 1)

    def surface_density_at_radius(self, radius):
        return self.einstein_radius_rescaled * radius ** (-(self.slope - 1))

    @geometry_profiles.transform_coordinates
    def surface_density_at_coordinates(self, coordinates):
        """
        Calculate the projected surface density in dimensionless units at a given set of image_grid plane image_grid.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those image_grid
        """

        eta = self.coordinates_to_elliptical_radius(coordinates)
        return self.surface_density_at_radius(eta)

    def potential_func(self, u, coordinates):
        eta = self.eta_u(u, coordinates)
        return (eta / u) * ((3.0 - self.slope) * eta) ** -1.0 * eta ** (3.0 - self.slope) / \
               ((1 - (1 - self.axis_ratio ** 2) * u) ** 0.5)

    @geometry_profiles.transform_coordinates
    def potential_at_coordinates(self, coordinates):
        """
        Calculate the gravitational potential at a given set of image_grid plane image_grid

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The gravitational potential [phi(eta)] (r-direction) at those image_grid
        """

        potential = quad(self.potential_func, a=0.0, b=1.0, args=(coordinates,))[0]
        return self.einstein_radius_rescaled * self.axis_ratio * potential

    def deflection_func(self, u, coordinates, npow):
        eta = self.eta_u(u, coordinates)
        return self.surface_density_at_radius(eta) / ((1 - (1 - self.axis_ratio ** 2) * u) ** (npow + 0.5))

    @geometry_profiles.transform_coordinates
    def deflections_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image_grid plane image_grid

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those image_grid
        """

        def calculate_deflection_component(npow, index):
            deflection = quad(self.deflection_func, a=0.0, b=1.0, args=(coordinates, npow))[0]
            return self.axis_ratio * coordinates[index] * deflection

        deflection_x = calculate_deflection_component(0.0, 0)
        deflection_y = calculate_deflection_component(1.0, 1)

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))


class SphericalPowerLaw(EllipticalPowerLaw):
    """Represents a spherical power-law density distribution"""

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0):
        """
        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        einstein_radius : float
            Einstein radius of power-law mass profiles
        slope : float
            power-law density slope of mass profiles
        """

        super(SphericalPowerLaw, self).__init__(centre, 1.0, 0.0, einstein_radius, slope)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\alpha']

    @geometry_profiles.transform_coordinates
    def deflections_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image_grid plane image_grid

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those image_grid
        """
        eta = self.coordinates_to_elliptical_radius(coordinates)
        deflection_r = 2.0 * self.einstein_radius_rescaled * ((3.0 - self.slope) * eta) ** -1.0 * eta ** (
                3.0 - self.slope)
        return self.coordinates_radius_to_x_and_y(coordinates, deflection_r)

    @geometry_profiles.transform_grid
    def deflections_from_coordinate_grid(self, grid):
        eta = self.grid_to_elliptical_radius(grid)
        deflection_r = np.divide(np.power(eta, (3.0 - self.slope)),
                                 2.0 * self.einstein_radius_rescaled * np.multiply((3.0 - self.slope), eta))
        return self.grid_radius_to_cartesian(grid, deflection_r)


class EllipticalIsothermal(EllipticalPowerLaw):
    """Represents an elliptical isothermal density distribution, which is equivalent to the elliptical power-law
    density distribution for the value slope=2.0"""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=0.9, phi=0.0, einstein_radius=1.0):
        """
        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of mass profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profiles ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profiles
        """

        super(EllipticalIsothermal, self).__init__(centre, axis_ratio, phi, einstein_radius, 2.0)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', r'\theta']

    @geometry_profiles.transform_coordinates
    def deflections_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image_grid plane image_grid

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those image_grid
        """

        # TODO: psi sometimes throws a division by zero error. May need to check value of psi, try/except or even
        # TODO: throw an assertion error if the inputs causing the error are invalid?

        psi = math.sqrt((self.axis_ratio ** 2) * (coordinates[0] ** 2) + coordinates[1] ** 2)

        deflection_x = 2.0 * self.einstein_radius_rescaled * self.axis_ratio / math.sqrt(
            1 - self.axis_ratio ** 2) * math.atan((math.sqrt(1 - self.axis_ratio ** 2) * coordinates[0]) / psi)
        deflection_y = 2.0 * self.einstein_radius_rescaled * self.axis_ratio / math.sqrt(
            1 - self.axis_ratio ** 2) * math.atanh((math.sqrt(1 - self.axis_ratio ** 2) * coordinates[1]) / psi)

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))

    @geometry_profiles.transform_grid
    def deflections_from_coordinate_grid(self, grid):
        """
        Calculate the deflection angle at a given set of image_grid plane image_grid

        Parameters
        ----------
        grid

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those image_grid
        """

        # TODO: psi sometimes throws a division by zero error. May need to check value of psi, try/except or even
        # TODO: throw an assertion error if the inputs causing the error are invalid?
        factor = 2.0 * self.einstein_radius_rescaled * self.axis_ratio / math.sqrt(1 - self.axis_ratio ** 2)

        psi = np.sqrt(np.add(np.multiply(self.axis_ratio ** 2, np.square(grid[:, 0])), np.square(grid[:, 1])))

        deflection_x = factor * np.arctan(np.divide(np.multiply(math.sqrt(1 - self.axis_ratio ** 2), grid[:, 0]), psi))
        deflection_y = factor * np.arctanh(np.divide(np.multiply(math.sqrt(1 - self.axis_ratio ** 2), grid[:, 1]), psi))

        return self.rotate_grid_from_profile(np.vstack((deflection_x, deflection_y)).T)


class SphericalIsothermal(EllipticalIsothermal):
    """Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
    density distribution for the value slope=2.0"""

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0):
        """

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        einstein_radius : float
            Einstein radius of power-law mass profiles
        """

        super(SphericalIsothermal, self).__init__(centre, 1.0, 0.0, einstein_radius)

    @property
    def parameter_labels(self):
        return ['x', 'y', r'\theta']

    @geometry_profiles.transform_coordinates
    def potential_at_coordinates(self, coordinates):
        """
        Calculate the gravitational potential at a given set of image_grid plane image_grid

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The gravitational potential [phi(eta)] (r-direction) at those image_grid
        """
        eta = self.coordinates_to_elliptical_radius(coordinates)
        return 2.0 * self.einstein_radius_rescaled * eta

    @geometry_profiles.transform_coordinates
    def deflections_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image_grid plane image_grid

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those image_grid
        """
        return self.coordinates_radius_to_x_and_y(coordinates, 2.0 * self.einstein_radius_rescaled)


class EllipticalCoredPowerLaw(EllipticalPowerLaw):
    """Represents a cored elliptical power-law density distribution"""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0, slope=2.0, core_radius=0.05):
        """

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of mass profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profiles ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profiles
        slope : float
            power-law density slope of mass profiles
        core_radius : float
            The radius of the inner core
        """
        super(EllipticalCoredPowerLaw, self).__init__(centre, axis_ratio, phi, einstein_radius, slope)

        self.core_radius = core_radius

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', r'\theta', r'\alpha', 'S']

    def surface_density_at_radius(self, radius):
        return self.einstein_radius_rescaled * (self.core_radius ** 2 + radius ** 2) ** (-(self.slope - 1) / 2.0)

    def potential_func(self, u, coordinates):
        eta = self.eta_u(u, coordinates)
        return (eta / u) * ((3.0 - self.slope) * eta) ** -1.0 * \
               ((self.core_radius ** 2 + eta ** 2) ** ((3.0 - self.slope) / 2.0) -
                self.core_radius ** (3 - self.slope)) / ((1 - (1 - self.axis_ratio ** 2) * u) ** 0.5)


class SphericalCoredPowerLaw(EllipticalCoredPowerLaw):
    """Represents a cored spherical power-law density distribution"""

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0, core_radius=0.05):
        """

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

    @property
    def parameter_labels(self):
        return ['x', 'y', r'\theta', r'\alpha', 'S']

    @geometry_profiles.transform_coordinates
    def deflections_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image_grid plane image_grid

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those image_grid
        """
        eta = self.coordinates_to_elliptical_radius(coordinates)
        deflection_r = 2.0 * self.einstein_radius_rescaled * ((3.0 - self.slope) * eta) ** -1.0 * (
                (self.core_radius ** 2 + eta ** 2) ** ((3.0 - self.slope) / 2.0) - self.core_radius ** (3 - self.slope))

        return self.coordinates_radius_to_x_and_y(coordinates, deflection_r)


class EllipticalCoredIsothermal(EllipticalCoredPowerLaw):
    """Represents a cored elliptical isothermal density distribution, which is equivalent to the elliptical power-law
    density distribution for the value slope=2.0"""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0, core_radius=0.05):
        """

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of mass profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profiles ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profiles
        core_radius : float
            The radius of the inner core
        """

        super(EllipticalCoredIsothermal, self).__init__(centre, axis_ratio, phi, einstein_radius, 2.0,
                                                        core_radius)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', r'\theta', 'S']


class SphericalCoredIsothermal(SphericalCoredPowerLaw):
    """Represents a cored spherical isothermal density distribution, which is equivalent to the elliptical power-law
    density distribution for the value slope=2.0"""

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0, core_radius=0.05):
        """

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

    @property
    def parameter_labels(self):
        return ['x', 'y', r'\theta', 'S']


class EllipticalNFW(EllipticalMassProfile, MassProfile):
    """The elliptical NFW profiles, used to fit the dark matter halo of the lens."""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, kappa_s=0.05, scale_radius=5.0):
        """ Setup a NFW dark matter profiles.

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

        super(EllipticalNFW, self).__init__(centre, axis_ratio, phi)
        super(MassProfile, self).__init__()
        self.kappa_s = kappa_s
        self.scale_radius = scale_radius

    @property
    def subscript(self):
        return 'd'

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', r'\kappa', 'Rs']

    @staticmethod
    def coord_func(r):
        if r > 1:
            return (1.0 / math.sqrt(r ** 2 - 1)) * math.atan(math.sqrt(r ** 2 - 1))
        elif r < 1:
            return (1.0 / math.sqrt(1 - r ** 2)) * math.atanh(math.sqrt(1 - r ** 2))
        elif r == 1:
            return 1

    def surface_density_at_radius(self, radius):
        radius = (1.0 / self.scale_radius) * radius
        return 2.0 * self.kappa_s * (1 - self.coord_func(radius)) / (radius ** 2 - 1)

    @geometry_profiles.transform_coordinates
    def surface_density_at_coordinates(self, coordinates):
        """
        Calculate the projected surface density in dimensionless units at a given set of image_grid plane image_grid.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those image_grid
        """

        eta = self.coordinates_to_elliptical_radius(coordinates)

        return self.surface_density_at_radius(eta)

    def potential_func(self, u, coordinates):
        eta = (1.0 / self.scale_radius) * self.eta_u(u, coordinates)
        return (self.axis_ratio / 2.0) * (eta / u) * ((math.log(eta / 2.0) + self.coord_func(eta)) / eta) / (
                (1 - (1 - self.axis_ratio ** 2) * u) ** 0.5)

    @geometry_profiles.transform_coordinates
    def potential_at_coordinates(self, coordinates):
        """
        Calculate the projected gravitational potential in dimensionless units at a given set of image_grid plane
        image_grid.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those image_grid
        """
        potential = quad(self.potential_func, a=0.0, b=1.0, args=(coordinates,))[0]
        return 4.0 * self.kappa_s * self.scale_radius * potential

    def deflection_func(self, u, coordinates, npow):
        eta_u = self.eta_u(u, coordinates)
        return self.surface_density_at_radius(eta_u) / ((1 - (1 - self.axis_ratio ** 2) * u) ** (npow + 0.5))

    @geometry_profiles.transform_coordinates
    def deflections_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image_grid plane image_grid

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those image_grid
        """

        def calculate_deflection_component(npow, index):
            deflection = quad(self.deflection_func, a=0.0, b=1.0, args=(coordinates, npow))[0]
            return deflection * self.axis_ratio * coordinates[index]

        deflection_x = calculate_deflection_component(0.0, 0)
        deflection_y = calculate_deflection_component(1.0, 1)

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))


class SphericalNFW(EllipticalNFW):
    """The spherical NFW profiles, used to fit the dark matter halo of the lens."""

    def __init__(self, centre=(0.0, 0.0), kappa_s=0.05, scale_radius=5.0):
        """ Setup a NFW dark matter profiles.

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

    @property
    def parameter_labels(self):
        return ['x', 'y', r'\kappa', 'Rs']

    @staticmethod
    def potential_func_sph(eta):
        return ((math.log(eta / 2.0)) ** 2) - (math.atanh(math.sqrt(1 - eta ** 2))) ** 2

    # TODO : The 'func' routines require a different input to the elliptical cases, meaning they cannot be overridden.
    # TODO : Should be able to refactor code to deal with this nicely, but will wait until we're clear on numba.

    @geometry_profiles.transform_coordinates
    def potential_at_coordinates(self, coordinates):
        """
        Calculate the projected gravitational potential in dimensionless units at a given set of image_grid plane
        image_grid.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those image_grid
        """
        eta = (1.0 / self.scale_radius) * self.coordinates_to_elliptical_radius(coordinates)
        return 2.0 * self.scale_radius * self.kappa_s * self.potential_func_sph(eta)

    def deflection_func_sph(self, eta):
        return (math.log(eta / 2.0) + self.coord_func(eta)) / eta

    @geometry_profiles.transform_coordinates
    def deflections_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image_grid plane image_grid

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those image_grid
        """
        eta = (1.0 / self.scale_radius) * self.coordinates_to_elliptical_radius(coordinates)
        deflection_r = 4.0 * self.kappa_s * self.scale_radius * self.deflection_func_sph(eta)

        return self.coordinates_radius_to_x_and_y(coordinates, deflection_r)


class EllipticalGeneralizedNFW(EllipticalNFW):
    """The elliptical NFW profiles, used to fit the dark matter halo of the lens."""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, kappa_s=0.05, inner_slope=1.0, scale_radius=5.0):
        """ Setup a NFW dark matter profiles.

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

        super(EllipticalGeneralizedNFW, self).__init__(centre, axis_ratio, phi, kappa_s, scale_radius)
        self.inner_slope = inner_slope

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', r'\kappa', r'\gamma' 'Rs']

    def integral_y(self, y, eta):
        return (y + eta) ** (self.inner_slope - 4) * (1 - math.sqrt(1 - y ** 2))

    def integral_y_2(self, y, eta):
        return (y + eta) ** (self.inner_slope - 3) * ((1 - math.sqrt(1 - y ** 2)) / y)

    def surface_density_at_radius(self, radius):
        radius = (1.0 / self.scale_radius) * radius
        integral_y = quad(self.integral_y, a=0.0, b=1.0, args=radius)[0]

        return 2.0 * self.kappa_s * (radius ** (1 - self.inner_slope)) * (
                (1 + radius) ** (self.inner_slope - 3) + ((3 - self.inner_slope) * integral_y))

    def potential_func_ell(self, u, coordinates):
        eta = (1.0 / self.scale_radius) * self.eta_u(u, coordinates)
        return (eta / u) * (self.deflection_func_sph(eta)) / ((1 - (1 - self.axis_ratio ** 2) * u) ** 0.5)

    @geometry_profiles.transform_coordinates
    def potential_at_coordinates(self, coordinates):
        """
        Calculate the projected gravitational potential in dimensionless units at a given set of image_grid plane
        image_grid.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those image_grid
        """
        potential = quad(self.potential_func_ell, a=0.0, b=1.0, args=(coordinates,))[0]
        return 4.0 * self.kappa_s * self.scale_radius * self.axis_ratio / 2.0 * potential

    def deflection_func_sph(self, eta):
        integral_y_2 = quad(self.integral_y_2, a=0.0, b=1.0, args=eta)[0]
        return eta ** (2 - self.inner_slope) * (
                (1.0 / (3 - self.inner_slope)) *
                special.hyp2f1(3 - self.inner_slope, 3 - self.inner_slope, 4 - self.inner_slope, -eta) + integral_y_2)

    def deflection_func_ell(self, u, coordinates, npow):
        eta_u = self.eta_u(u, coordinates)

        return self.surface_density_at_radius(eta_u) / ((1 - (1 - self.axis_ratio ** 2) * u) ** (npow + 0.5))

    @geometry_profiles.transform_coordinates
    def deflections_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image_grid plane image_grid

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those image_grid
        """

        def calculate_deflection_component(npow, index):
            deflection = quad(self.deflection_func_ell, a=0.0, b=1.0, args=(coordinates, npow))[0]
            return self.axis_ratio * deflection * coordinates[index]

        deflection_x = calculate_deflection_component(0.0, 0)
        deflection_y = calculate_deflection_component(1.0, 1)

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))


class SphericalGeneralizedNFW(EllipticalGeneralizedNFW):
    """The spherical NFW profiles, used to fit the dark matter halo of the lens."""

    def __init__(self, centre=(0.0, 0.0), kappa_s=0.05, inner_slope=1.0, scale_radius=5.0):
        """ Setup a NFW dark matter profiles.

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

    @property
    def parameter_labels(self):
        return ['x', 'y', r'\kappa', r'\gamma' 'Rs']

    @geometry_profiles.transform_coordinates
    def deflections_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image_grid plane image_grid

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those image_grid
        """
        eta = (1.0 / self.scale_radius) * self.coordinates_to_elliptical_radius(coordinates)
        deflection_r = 4.0 * self.kappa_s * self.scale_radius * self.deflection_func_sph(eta)

        return self.coordinates_radius_to_x_and_y(coordinates, deflection_r)


class EllipticalSersicMass(light_profiles.EllipticalSersic, EllipticalMassProfile):
    """The Sersic mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens \
     galaxy's light."""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0, mass_to_light_ratio=1.0):
        """
        Setup a Sersic mass and light profiles.

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
        super(EllipticalSersicMass, self).__init__(centre, axis_ratio, phi, intensity, effective_radius, sersic_index)
        super(EllipticalMassProfile, self).__init__(centre, axis_ratio, phi)
        self.mass_to_light_ratio = mass_to_light_ratio

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R', 'n', r'\Psi']

    def surface_density_at_radius(self, radius):
        return self.mass_to_light_ratio * self.intensity_at_radius(radius)

    @geometry_profiles.transform_coordinates
    def surface_density_at_coordinates(self, coordinates):
        """Calculate the projected surface density in dimensionless units at a given set of image_grid plane image_grid.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those image_grid
        """
        return self.surface_density_at_radius(self.coordinates_to_eccentric_radius(coordinates))

    @property
    def deflection_normalization(self):
        return self.axis_ratio

    def deflection_func(self, u, coordinates, npow):
        eta_u = math.sqrt(self.axis_ratio) * self.eta_u(u, coordinates)
        return self.surface_density_at_radius(eta_u) / ((1 - (1 - self.axis_ratio ** 2) * u) ** (npow + 0.5))

    @geometry_profiles.transform_coordinates
    def deflections_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image_grid plane image_grid

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those image_grid
        """

        def calculate_deflection_component(npow, index):
            deflection = quad(self.deflection_func, a=0.0, b=1.0, args=(coordinates, npow))[0]
            return self.deflection_normalization * deflection * coordinates[index]

        deflection_x = calculate_deflection_component(0.0, 0)
        deflection_y = calculate_deflection_component(1.0, 1)

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))


class EllipticalExponentialMass(EllipticalSersicMass):
    """The EllipticalExponentialMass mass profile, the mass profiles of the light profiles that are used to fit and
    subtract the lens galaxy's light."""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 mass_to_light_ratio=1.0):
        """
        Setup an EllipticalExponentialMass mass and light profile.

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
        super(EllipticalExponentialMass, self).__init__(centre, axis_ratio, phi, intensity, effective_radius, 1.0,
                                                        mass_to_light_ratio)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R', r'\Psi']

    @classmethod
    def from_exponential_light_profile(cls, exponential_light_profile, mass_to_light_ratio):
        return EllipticalExponentialMass.from_profile(exponential_light_profile,
                                                      mass_to_light_ratio=mass_to_light_ratio)


class EllipticalDevVaucouleursMass(EllipticalSersicMass):
    """The EllipticalDevVaucouleursMass mass profile, the mass profiles of the light profiles that are used to fit and
    subtract the lens galaxy's light."""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 mass_to_light_ratio=1.0):
        """
        Setup a EllipticalDevVaucouleursMass mass and light profile.

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
        super(EllipticalDevVaucouleursMass, self).__init__(centre, axis_ratio, phi, intensity, effective_radius, 4.0,
                                                           mass_to_light_ratio)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R', r'\Psi']

    @classmethod
    def from_dev_vaucouleurs_light_profile(cls, dev_vaucouleurs_light_profile, mass_to_light_ratio):
        return EllipticalDevVaucouleursMass.from_profile(dev_vaucouleurs_light_profile,
                                                         mass_to_light_ratio=mass_to_light_ratio)


class EllipticalSersicMassRadialGradient(EllipticalSersicMass):

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
        super(EllipticalSersicMassRadialGradient, self).__init__(centre, axis_ratio, phi, intensity, effective_radius,
                                                                 sersic_index, mass_to_light_ratio)
        self.mass_to_light_gradient = mass_to_light_gradient

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R', 'n', r'\Psi', r'\Tau']

    def surface_density_at_radius(self, radius):
        return self.mass_to_light_ratio * (
                ((self.axis_ratio * radius) / self.effective_radius) ** -self.mass_to_light_gradient) \
               * self.intensity_at_radius(radius)

    @geometry_profiles.transform_coordinates
    def surface_density_at_coordinates(self, coordinates):
        """Calculate the projected surface density in dimensionless units at a given set of image_grid plane image_grid.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those image_grid
        """
        return self.surface_density_at_radius(self.coordinates_to_eccentric_radius(coordinates))


class ExternalShear(geometry_profiles.EllipticalProfile, MassProfile):
    """An external shear term, to model the line-of-sight contribution of other galaxies / satellites."""

    def __init__(self, magnitude=0.2, phi=0.0):
        """ Setup an external shear.

        Parameters
        ----------
        magnitude : float
            The overall magnitude of the shear (gamma).
        phi : float
            The rotation axis of the shear.
        """

        super(ExternalShear, self).__init__(centre=(0.0, 0.0), phi=phi, axis_ratio=1.0)
        self.magnitude = magnitude

    @property
    def subscript(self):
        return 'sh'

    @property
    def parameter_labels(self):
        return [r'\gamma', r'\theta']

    @geometry_profiles.transform_coordinates
    def deflections_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image_grid plane image_grid

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those image_grid
        """
        deflection_x = self.magnitude * coordinates[0]
        deflection_y = -self.magnitude * coordinates[1]
        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))

    @geometry_profiles.transform_grid
    def deflections_from_coordinate_grid(self, grid):
        deflection_x = np.multiply(self.magnitude, grid[:, 0])
        deflection_y = np.multiply(self.magnitude, grid[:, 1])
        return self.rotate_grid_from_profile(np.vstack((deflection_x, deflection_y)).T)

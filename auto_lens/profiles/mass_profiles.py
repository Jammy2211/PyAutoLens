from auto_lens.profiles import geometry_profiles
from auto_lens.profiles import light_profiles
import math
import numpy as np
from scipy.integrate import quad
from scipy import special


class MassProfile(object):

    # noinspection PyMethodMayBeStatic
    def surface_density_at_radius(self, eta):
        raise AssertionError("Surface density at radius should be overridden")

    # noinspection PyMethodMayBeStatic
    def surface_density_at_coordinates(self, coordinates):
        raise AssertionError("Surface density at coordinates should be overridden")

    # noinspection PyMethodMayBeStatic
    def potential_at_coordinates(self, coordinates):
        raise AssertionError("Potential at coordinates should be overridden")

    # noinspection PyMethodMayBeStatic
    def deflection_angles_at_coordinates(self, coordinates):
        raise AssertionError("Deflection angles at coordinates should be overridden")


class EllipticalMassProfile(geometry_profiles.EllipticalProfile, MassProfile):
    """Generic class for an elliptical light profiles"""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The circular radius containing half the light of this model
        sersic_index : Int
            The concentration of the light profiles
        """
        super(EllipticalMassProfile, self).__init__(centre, axis_ratio, phi)
        self.axis_ratio = axis_ratio
        self.phi = phi

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi']

    def dimensionless_mass_within_circle(self, radius):
        """
        Compute the mass profiles's total dimensionless mass within a circle of specified radius. This is performed via \
        integration of the surface density profiles and is centred on the mass model.

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
        Compute the mass profiles's total dimensionless mass within an ellipse of specified radius. This is performed \
        via integration of the surface density profiles and is centred and rotationally aligned with the mass model.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.

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


class EllipticalPowerLawMassProfile(EllipticalMassProfile, MassProfile):
    """Represents an elliptical power-law density distribution"""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0, slope=2.0):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        axis_ratio : float
            Ratio of mass profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profiles ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profiles
        slope : float
            power-law density slope of mass profiles
        """

        super(EllipticalPowerLawMassProfile, self).__init__(centre, axis_ratio, phi)
        super(MassProfile, self).__init__()

        self.einstein_radius = einstein_radius
        self.slope = slope

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', r'\theta', r'\alpha']

    @property
    def einstein_radius_rescaled(self):
        """Rescale the einstein radius by slope and axis_ratio, to reduce its degeneracy with other mass-profiles \
        parameters"""
        return ((3 - self.slope) / (1 + self.axis_ratio)) * self.einstein_radius ** (self.slope - 1)

    def surface_density_at_radius(self, radius):
        return self.einstein_radius_rescaled * radius ** (-(self.slope - 1))

    @geometry_profiles.transform_coordinates
    def surface_density_at_coordinates(self, coordinates):
        """
        Calculate the projected surface density in dimensionless units at a given set of image plane coordinates.

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those coordinates
        """

        eta = self.coordinates_to_elliptical_radius(coordinates)
        return self.surface_density_at_radius(eta)

    def potential_func(self, u, coordinates):
        eta = self.eta_u(u, coordinates)
        return (eta / u) * ((3.0 - self.slope) * eta) ** -1.0 * eta ** (3.0 - self.slope) / \
               ((1 - (1 - self.axis_ratio ** 2) * u) ** (0.5))

    @geometry_profiles.transform_coordinates
    def potential_at_coordinates(self, coordinates):
        """
        Calculate the gravitational potential at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The gravitational potential [phi(eta)] (r-direction) at those coordinates
        """

        potential = quad(self.potential_func, a=0.0, b=1.0, args=(coordinates,))[0]
        return self.einstein_radius_rescaled * self.axis_ratio * potential

    def deflection_func(self, u, coordinates, npow):
        eta = self.eta_u(u, coordinates)
        return self.surface_density_at_radius(eta) / ((1 - (1 - self.axis_ratio ** 2) * u) ** (npow + 0.5))

    @geometry_profiles.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those coordinates
        """

        def calculate_deflection_component(npow, index):
            deflection = quad(self.deflection_func, a=0.0, b=1.0, args=(coordinates, npow))[0]
            return self.axis_ratio * coordinates[index] * deflection

        deflection_x = calculate_deflection_component(0.0, 0)
        deflection_y = calculate_deflection_component(1.0, 1)

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))


class SphericalPowerLawMassProfile(EllipticalPowerLawMassProfile):
    """Represents a spherical power-law density distribution"""

    def __init__(self,centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        axis_ratio : float
            Ratio of mass profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profiles ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profiles
        slope : float
            power-law density slope of mass profiles
        """

        super(SphericalPowerLawMassProfile, self).__init__(centre, 1.0, 0.0, einstein_radius, slope)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\alpha']

    @geometry_profiles.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those coordinates
        """
        eta = self.coordinates_to_elliptical_radius(coordinates)
        deflection_r = 2.0 * self.einstein_radius_rescaled * ((3.0 - self.slope) * eta) ** -1.0 * eta ** (
            3.0 - self.slope)
        return self.coordinates_radius_to_x_and_y(coordinates, deflection_r)


class EllipticalIsothermalMassProfile(EllipticalPowerLawMassProfile):
    """Represents an elliptical isothermal density distribution, which is equivalent to the elliptical power-law
    density distribution for the value slope=2.0"""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        axis_ratio : float
            Ratio of mass profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profiles ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profiles
        """

        super(EllipticalIsothermalMassProfile, self).__init__(centre, axis_ratio, phi, einstein_radius, 2.0)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', r'\theta']

    @geometry_profiles.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those coordinates
        """

        # TODO: psi sometimes throws a division by zero error. May need to check value of psi, try/except or even
        # TODO: throw an assertion error if the inputs causing the error are invalid?

        psi = math.sqrt((self.axis_ratio ** 2) * (coordinates[0] ** 2) + coordinates[1] ** 2)
        deflection_x = 2.0 * self.einstein_radius_rescaled * self.axis_ratio / math.sqrt(1 - self.axis_ratio ** 2) * \
                       math.atan((math.sqrt(1 - self.axis_ratio ** 2) * coordinates[0]) / psi)
        deflection_y = 2.0 * self.einstein_radius_rescaled * self.axis_ratio / math.sqrt(1 - self.axis_ratio ** 2) * \
                       math.atanh((math.sqrt(1 - self.axis_ratio ** 2) * coordinates[1]) / psi)

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))


class SphericalIsothermalMassProfile(EllipticalIsothermalMassProfile):
    """Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
    density distribution for the value slope=2.0"""

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        einstein_radius : float
            Einstein radius of power-law mass profiles
        """

        super(SphericalIsothermalMassProfile, self).__init__(centre, 1.0, 0.0, einstein_radius)

    @property
    def parameter_labels(self):
        return ['x', 'y', r'\theta']

    @geometry_profiles.transform_coordinates
    def potential_at_coordinates(self, coordinates):
        """
        Calculate the gravitational potential at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The gravitational potential [phi(eta)] (r-direction) at those coordinates
        """
        eta = self.coordinates_to_elliptical_radius(coordinates)
        return 2.0 * self.einstein_radius_rescaled * eta

    @geometry_profiles.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those coordinates
        """
        return self.coordinates_radius_to_x_and_y(coordinates, 2.0 * self.einstein_radius_rescaled)


class CoredEllipticalPowerLawMassProfile(EllipticalPowerLawMassProfile):
    """Represents a cored elliptical power-law density distribution"""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0, slope=2.0, core_radius=0.05):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
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
        super(CoredEllipticalPowerLawMassProfile, self).__init__(centre, axis_ratio, phi, einstein_radius, slope)

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


class CoredSphericalPowerLawMassProfile(CoredEllipticalPowerLawMassProfile):
    """Represents a cored spherical power-law density distribution"""

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0, core_radius=0.05):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        einstein_radius : float
            Einstein radius of power-law mass profiles
        slope : float
            power-law density slope of mass profiles
        core_radius : float
            The radius of the inner core
        """
        super(CoredSphericalPowerLawMassProfile, self).__init__(centre, 1.0, 0.0, einstein_radius, slope, core_radius)

    @property
    def parameter_labels(self):
        return ['x', 'y', r'\theta', r'\alpha', 'S']

    @geometry_profiles.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those coordinates
        """
        eta = self.coordinates_to_elliptical_radius(coordinates)
        deflection_r = 2.0 * self.einstein_radius_rescaled * ((3.0 - self.slope) * eta) ** -1.0 * \
                       ((self.core_radius ** 2 + eta ** 2) ** ((3.0 - self.slope) / 2.0) - self.core_radius ** (
                           3 - self.slope))

        return self.coordinates_radius_to_x_and_y(coordinates, deflection_r)


class CoredEllipticalIsothermalMassProfile(CoredEllipticalPowerLawMassProfile):
    """Represents a cored elliptical isothermal density distribution, which is equivalent to the elliptical power-law
    density distribution for the value slope=2.0"""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0, core_radius=0.05):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        axis_ratio : float
            Ratio of mass profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profiles ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profiles
        core_radius : float
            The radius of the inner core
        """

        super(CoredEllipticalIsothermalMassProfile, self).__init__(centre, axis_ratio, phi, einstein_radius, 2.0,
                                                                   core_radius)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', r'\theta', 'S']


class CoredSphericalIsothermalMassProfile(CoredSphericalPowerLawMassProfile):
    """Represents a cored spherical isothermal density distribution, which is equivalent to the elliptical power-law
    density distribution for the value slope=2.0"""

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0, core_radius=0.05):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        einstein_radius : float
            Einstein radius of power-law mass profiles
        core_radius : float
            The radius of the inner core
        """

        super(CoredSphericalIsothermalMassProfile, self).__init__(centre, einstein_radius, 2.0, core_radius)

    @property
    def parameter_labels(self):
        return ['x', 'y', r'\theta', 'S']


class EllipticalNFWMassProfile(EllipticalMassProfile, MassProfile):
    """The elliptical NFW profiles, used to fit the dark matter halo of the lens."""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, kappa_s=0.05, scale_radius=5.0):
        """ Setup a NFW dark matter profiles.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        kappa_s : float
            The overall normalization of the dark matter halo
        scale_radius : float
            The radius containing half the light of this model
        """

        super(EllipticalNFWMassProfile, self).__init__(centre, axis_ratio, phi)
        super(MassProfile, self).__init__()
        self.kappa_s = kappa_s
        self.scale_radius = scale_radius

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', r'\kappa', 'Rs']

    @property
    def subscript_label(self):
        return 'd'

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
        Calculate the projected surface density in dimensionless units at a given set of image plane coordinates.

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those coordinates
        """

        eta = self.coordinates_to_elliptical_radius(coordinates)

        return self.surface_density_at_radius(eta)

    def potential_func(self, u, coordinates):
        eta = (1.0 / self.scale_radius) * self.eta_u(u, coordinates)
        return (self.axis_ratio / 2.0) * (eta / u) * ((math.log(eta / 2.0) + self.coord_func(eta)) / eta) / (
            (1 - (1 - self.axis_ratio ** 2) * u) ** (0.5))

    @geometry_profiles.transform_coordinates
    def potential_at_coordinates(self, coordinates):
        """
        Calculate the projected gravitational potential in dimensionless units at a given set of image plane coordinates.

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those coordinates
        """
        potential = quad(self.potential_func, a=0.0, b=1.0, args=(coordinates,))[0]
        return 4.0 * self.kappa_s * self.scale_radius * potential

    def deflection_func(self, u, coordinates, npow):
        eta_u = self.eta_u(u, coordinates)
        return self.surface_density_at_radius(eta_u) / ((1 - (1 - self.axis_ratio ** 2) * u) ** (npow + 0.5))

    @geometry_profiles.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those coordinates
        """

        def calculate_deflection_component(npow, index):
            deflection = quad(self.deflection_func, a=0.0, b=1.0, args=(coordinates, npow))[0]
            return deflection * self.axis_ratio * coordinates[index]

        deflection_x = calculate_deflection_component(0.0, 0)
        deflection_y = calculate_deflection_component(1.0, 1)

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))


class SphericalNFWMassProfile(EllipticalNFWMassProfile):
    """The spherical NFW profiles, used to fit the dark matter halo of the lens."""

    def __init__(self, centre=(0.0, 0.0), kappa_s=0.05, scale_radius=5.0):
        """ Setup a NFW dark matter profiles.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        kappa_s : float
            The overall normalization of the dark matter halo
        scale_radius : float
            The radius containing half the light of this model
        """

        super(SphericalNFWMassProfile, self).__init__(centre, 1.0, 0.0, kappa_s, scale_radius)

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
        Calculate the projected gravitational potential in dimensionless units at a given set of image plane coordinates.

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those coordinates
        """
        eta = (1.0 / self.scale_radius) * self.coordinates_to_elliptical_radius(coordinates)
        return 2.0 * self.scale_radius * self.kappa_s * self.potential_func_sph(eta)

    def deflection_func_sph(self, eta):
        return (math.log(eta / 2.0) + self.coord_func(eta)) / eta

    @geometry_profiles.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those coordinates
        """
        eta = (1.0 / self.scale_radius) * self.coordinates_to_elliptical_radius(coordinates)
        deflection_r = 4.0 * self.kappa_s * self.scale_radius * self.deflection_func_sph(eta)

        return self.coordinates_radius_to_x_and_y(coordinates, deflection_r)


class EllipticalGeneralizedNFWMassProfile(EllipticalNFWMassProfile):
    """The elliptical NFW profiles, used to fit the dark matter halo of the lens."""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, kappa_s=0.05, inner_slope=1.0, scale_radius=5.0):
        """ Setup a NFW dark matter profiles.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        kappa_s : float
            The overall normalization of the dark matter halo
        inner_slope : float
            The inner slope of the dark matter halo
        scale_radius : float
            The radius containing half the light of this model
        """

        super(EllipticalGeneralizedNFWMassProfile, self).__init__(centre, axis_ratio, phi, kappa_s, scale_radius)
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
        integral_y = quad(self.integral_y, a=0.0, b=1.0, args=(radius))[0]

        return 2.0 * self.kappa_s * (radius ** (1 - self.inner_slope)) * \
               ((1 + radius) ** (self.inner_slope - 3) + ((3 - self.inner_slope) * integral_y))

    def potential_func_ell(self, u, coordinates):
        eta = (1.0 / self.scale_radius) * self.eta_u(u, coordinates)
        return (eta / u) * (self.deflection_func_sph(eta)) / ((1 - (1 - self.axis_ratio ** 2) * u) ** (0.5))

    @geometry_profiles.transform_coordinates
    def potential_at_coordinates(self, coordinates):
        """
        Calculate the projected gravitational potential in dimensionless units at a given set of image plane coordinates.

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those coordinates
        """
        potential = quad(self.potential_func_ell, a=0.0, b=1.0, args=(coordinates,))[0]
        return 4.0 * self.kappa_s * self.scale_radius * self.axis_ratio / 2.0 * potential

    def deflection_func_sph(self, eta):
        integral_y_2 = quad(self.integral_y_2, a=0.0, b=1.0, args=(eta))[0]
        return eta ** (2 - self.inner_slope) * (
            (1.0 / (3 - self.inner_slope)) *
            special.hyp2f1(3 - self.inner_slope, 3 - self.inner_slope, 4 - self.inner_slope, -eta) + integral_y_2)

    def deflection_func_ell(self, u, coordinates, npow):
        eta_u = self.eta_u(u, coordinates)

        return self.surface_density_at_radius(eta_u) / ((1 - (1 - self.axis_ratio ** 2) * u) ** (npow + 0.5))

    @geometry_profiles.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those coordinates
        """

        def calculate_deflection_component(npow, index):
            deflection = quad(self.deflection_func_ell, a=0.0, b=1.0, args=(coordinates, npow))[0]
            return self.axis_ratio * deflection * coordinates[index]

        deflection_x = calculate_deflection_component(0.0, 0)
        deflection_y = calculate_deflection_component(1.0, 1)

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))


class SphericalGeneralizedNFWMassProfile(EllipticalGeneralizedNFWMassProfile):
    """The spherical NFW profiles, used to fit the dark matter halo of the lens."""

    def __init__(self, centre=(0.0, 0.0), kappa_s=0.05, inner_slope=1.0, scale_radius=5.0):
        """ Setup a NFW dark matter profiles.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        kappa_s : float
            The overall normalization of the dark matter halo
        inner_slope : float
            The inner slope of the dark matter halo
        scale_radius : float
            The radius containing half the light of this model
        """

        super(SphericalGeneralizedNFWMassProfile, self).__init__(centre, 1.0, 0.0, kappa_s, inner_slope, scale_radius)

    @property
    def parameter_labels(self):
        return ['x', 'y', r'\kappa', r'\gamma' 'Rs']

    @geometry_profiles.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those coordinates
        """
        eta = (1.0 / self.scale_radius) * self.coordinates_to_elliptical_radius(coordinates)
        deflection_r = 4.0 * self.kappa_s * self.scale_radius * self.deflection_func_sph(eta)

        return self.coordinates_radius_to_x_and_y(coordinates, deflection_r)


class SersicMassProfile(light_profiles.SersicLightProfile, EllipticalMassProfile):
    """The Sersic mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens \
     galaxy's light."""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0, mass_to_light_ratio=1.0):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model
        sersic_index : Int
            The concentration of the light profiles
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(SersicMassProfile, self).__init__(centre, axis_ratio, phi, intensity, effective_radius, sersic_index)
        super(EllipticalMassProfile, self).__init__(centre, axis_ratio, phi)
        self.mass_to_light_ratio = mass_to_light_ratio

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R', 'n', r'\Psi']

    @classmethod
    def from_sersic_light_profile(cls, sersic_light_profile, mass_to_light_ratio):
        return SersicMassProfile.from_profile(sersic_light_profile, mass_to_light_ratio=mass_to_light_ratio)

    def surface_density_at_radius(self, radius):
        return self.mass_to_light_ratio * self.intensity_at_radius(radius)

    @geometry_profiles.transform_coordinates
    def surface_density_at_coordinates(self, coordinates):
        """Calculate the projected surface density in dimensionless units at a given set of image plane coordinates.

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those coordinates
        """
        return self.mass_to_light_ratio * self.intensity_at_coordinates(coordinates)

    @property
    def deflection_normalization(self):
        return self.mass_to_light_ratio * self.axis_ratio

    def deflection_func(self, u, coordinates, npow):
        eta_u = math.sqrt(self.axis_ratio) * self.eta_u(u, coordinates)
        return self.intensity_at_radius(eta_u) / ((1 - (1 - self.axis_ratio ** 2) * u) ** (npow + 0.5))

    @geometry_profiles.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
        """
        Calculate the deflection angle at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those coordinates
        """

        def calculate_deflection_component(npow, index):
            deflection = quad(self.deflection_func, a=0.0, b=1.0, args=(coordinates, npow))[0]
            return self.mass_to_light_ratio * self.axis_ratio * deflection * coordinates[index]

        deflection_x = calculate_deflection_component(0.0, 0)
        deflection_y = calculate_deflection_component(1.0, 1)

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))


class ExponentialMassProfile(SersicMassProfile):
    """The Exponential mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens \
     galaxy's light."""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0,phi=0.0, intensity=0.1, effective_radius=0.6,
                 mass_to_light_ratio=1.0):
        """
        Setup an Exponential mass and light profile.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(ExponentialMassProfile, self).__init__(centre, axis_ratio, phi, intensity, effective_radius, 1.0,
                                                     mass_to_light_ratio)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R', r'\Psi']

    @classmethod
    def from_exponential_light_profile(cls, exponential_light_profile, mass_to_light_ratio):
        return ExponentialMassProfile.from_profile(exponential_light_profile, mass_to_light_ratio=mass_to_light_ratio)


class DevVaucouleursMassProfile(SersicMassProfile):
    """The DevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens \
     galaxy's light."""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0,phi=0.0, intensity=0.1, effective_radius=0.6,
                 mass_to_light_ratio=1.0):
        """
        Setup a DevVaucouleurs mass and light profile.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(DevVaucouleursMassProfile, self).__init__(centre, axis_ratio, phi, intensity, effective_radius, 4.0,
                                                     mass_to_light_ratio)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R', r'\Psi']

    @classmethod
    def from_dev_vaucouleurs_light_profile(cls, dev_vaucouleurs_light_profile, mass_to_light_ratio):
        return DevVaucouleursMassProfile.from_profile(dev_vaucouleurs_light_profile, mass_to_light_ratio=mass_to_light_ratio)
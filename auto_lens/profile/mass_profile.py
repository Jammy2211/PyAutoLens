import profile
import math
from scipy.integrate import quad


class MassProfile(object):
    # noinspection PyMethodMayBeStatic
    def compute_deflection_angle(self, coordinates):
        raise AssertionError("Compute deflection angles should be overridden")


class CombinedMassProfile(list, MassProfile):
    """A mass profile comprising one or more mass profiles"""

    def __init__(self, *mass_profiles):
        super(CombinedMassProfile, self).__init__(mass_profiles)

    def compute_deflection_angle(self, coordinates):
        """
        Calculate the deflection angle at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angle at those coordinates
        """
        sum_tuple = (0, 0)
        for t in map(lambda p: p.compute_deflection_angle(coordinates), self):
            sum_tuple = (sum_tuple[0] + t[0], sum_tuple[1] + t[1])
        return sum_tuple


class EllipticalPowerLawMassProfile(profile.EllipticalProfile, MassProfile):
    """Represents an elliptical power-law density distribution"""

    def __init__(self, axis_ratio, phi, einstein_radius, slope, centre=(0, 0)):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profile
        axis_ratio : float
            Ratio of mass profile ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profile ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profile
        slope : float
            power-law density slope of mass profile
        """

        super(EllipticalPowerLawMassProfile, self).__init__(axis_ratio, phi, centre)
        super(MassProfile, self).__init__()

        self.einstein_radius = einstein_radius
        self.slope = slope

    @property
    def einstein_radius_rescaled(self):
        return ((3 - self.slope) / (1 + self.axis_ratio)) * self.einstein_radius ** (self.slope - 1)

    def eta_u(self, u, coordinates):
        return math.sqrt((u * ((coordinates[0] ** 2) + (coordinates[1] ** 2 / (1 - (1 - self.axis_ratio ** 2) * u)))))

    def surface_density_func(self, eta):
        return self.einstein_radius_rescaled * eta ** (-(self.slope - 1))

    @profile.transform_coordinates
    def compute_surface_density(self, coordinates):
        """
        Calculate the projected surface density in dimensionless units at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those coordinates
        """

        eta = math.sqrt((coordinates[0] ** 2) + (coordinates[1] ** 2) / (self.axis_ratio ** 2))

        return self.surface_density_func(eta)

    @property
    def potential_normalization(self):
        return 2.0 * self.einstein_radius_rescaled * self.axis_ratio / 2.0

    def potential_func(self, u, coordinates):
        eta = self.eta_u(u, coordinates)
        return (eta / u) * ((3.0 - self.slope) * eta) ** -1.0 * eta ** (3.0 - self.slope) / (
            (1 - (1 - self.axis_ratio ** 2) * u) ** (0.5))

    @profile.transform_coordinates
    def compute_potential(self, coordinates):
        """
        Calculate the gravitational potential at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The gravitational potential [phi(eta)] (r-direction) at those coordinates
        """

        potential = quad(self.potential_func, a=0.0, b=1.0, args=(coordinates,))[0]
        return self.potential_normalization * potential

    @property
    def deflection_normalization(self):
        return self.axis_ratio

    def deflection_func(self, u, coordinates, npow):
        eta = self.eta_u(u, coordinates)
        return self.surface_density_func(eta) / ((1 - (1 - self.axis_ratio ** 2) * u) ** (npow + 0.5))

    @profile.transform_coordinates
    def compute_deflection_angle(self, coordinates):
        """
        Calculate the deflection angle at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those coordinates
        """

        def calculate_deflection_component(npow, index):
            deflection = quad(self.deflection_func, a=0.0, b=1.0, args=(coordinates, npow))[0]
            return self.deflection_normalization * deflection * coordinates[index]

        deflection_x = calculate_deflection_component(0.0, 0)
        deflection_y = calculate_deflection_component(1.0, 1)

        return deflection_x, deflection_y


class EllipticalIsothermalMassProfile(EllipticalPowerLawMassProfile):
    """Represents an elliptical isothermal density distribution, which is equivalent to the elliptical power-law
    density distribution for the value slope=2.0"""

    def __init__(self, axis_ratio, phi, einstein_radius, centre=(0, 0)):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profile
        axis_ratio : float
            Ratio of mass profile ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profile ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profile
        """

        super(EllipticalIsothermalMassProfile, self).__init__(axis_ratio, phi, einstein_radius, 2.0, centre)

    @profile.transform_coordinates
    def compute_potential(self, coordinates):
        """
        Calculate the gravitational potential at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The gravitational potential [phi(eta)] at these coordinates
        """

        deflections = self.compute_deflection_angle(coordinates)

        return coordinates[0] * deflections[0] + coordinates[1] * deflections[1]

    @property
    def deflection_normalization(self):
        return 2.0 * self.einstein_radius_rescaled * self.axis_ratio / (math.sqrt(1 - self.axis_ratio ** 2))

    @profile.transform_coordinates
    def compute_deflection_angle(self, coordinates):
        """
        Calculate the deflection angle at a given set of image plane coordinates

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angles [alpha(eta)] (x and y components) at those coordinates
        """

        # TODO: psi sometimes throws a division by zero error. May need to check value of psi, try/except or even
        # TODO: throw an assertion error if the inputs causing the error are invalid?

        psi = math.sqrt((self.axis_ratio ** 2) * (coordinates[0] ** 2) + coordinates[1] ** 2)
        deflection_x = self.deflection_normalization * math.atan(
            (math.sqrt(1 - self.axis_ratio ** 2) * coordinates[0]) / psi)
        deflection_y = self.deflection_normalization * math.atanh(
            (math.sqrt(1 - self.axis_ratio ** 2) * coordinates[1]) / psi)

        return deflection_x, deflection_y


class CoredEllipticalPowerLawMassProfile(EllipticalPowerLawMassProfile):
    """Represents a cored elliptical power-law density distribution"""

    def __init__(self, axis_ratio, phi, einstein_radius, slope, core_radius, centre=(0, 0)):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profile
        axis_ratio : float
            Ratio of mass profile ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profile ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profile
        slope : float
            power-law density slope of mass profile
        core_radius : float
            The radius of the inner core
        """
        super(CoredEllipticalPowerLawMassProfile, self).__init__(axis_ratio, phi, einstein_radius, slope, centre)

        self.core_radius = core_radius

    @property
    def einstein_radius_rescaled(self):
        return ((3 - self.slope) / (1 + self.axis_ratio)) * (self.einstein_radius + self.core_radius ** 2) ** (
            self.slope - 1)

    def surface_density_func(self, eta):
        return self.einstein_radius_rescaled * (self.core_radius ** 2 + eta ** 2) ** (-(self.slope - 1) / 2.0)

    def potential_func(self, u, coordinates):
        eta = self.eta_u(u, coordinates)
        return (eta / u) * ((3.0 - self.slope) * eta) ** -1.0 * \
               ((self.core_radius ** 2 + eta ** 2) ** ((3.0 - self.slope) / 2.0) -
                self.core_radius ** (3 - self.slope)) / ((1 - (1 - self.axis_ratio ** 2) * u) ** 0.5)


class CoredEllipticalIsothermalMassProfile(CoredEllipticalPowerLawMassProfile):
    """Represents a cored elliptical isothermal density distribution, which is equivalent to the elliptical power-law
    density distribution for the value slope=2.0"""

    def __init__(self, axis_ratio, phi, einstein_radius, core_radius, centre=(0, 0)):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profile
        axis_ratio : float
            Ratio of mass profile ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profile ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profile
        core_radius : float
            The radius of the inner core
        """

        super(CoredEllipticalIsothermalMassProfile, self).__init__(axis_ratio, phi, einstein_radius, 2.0, core_radius,
                                                                   centre)

import profile, light_profile
import math
from scipy.integrate import quad


class MassProfile(object):
    # noinspection PyMethodMayBeStatic
    def surface_density_at_coordinates(self, coordinates):
        raise AssertionError("Surface density at coordinates should be overridden")

    # noinspection PyMethodMayBeStatic
    def potential_at_coordinates(self, coordinates):
        raise AssertionError("Potential at coordinates should be overridden")

    # noinspection PyMethodMayBeStatic
    def deflection_angles_at_coordinates(self, coordinates):
        raise AssertionError("Deflection angles at coordinaates should be overridden")


class CombinedMassProfile(list, MassProfile):
    """A mass profile comprising one or more mass profiles"""

    def __init__(self, *mass_profiles):
        super(CombinedMassProfile, self).__init__(mass_profiles)

    def surface_density_at_coordinates(self, coordinates):
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
        sum = 0.0
        for t in map(lambda p: p.surface_density_at_coordinates(coordinates), self):
            sum += t
        return sum

    def potential_at_coordinates(self, coordinates):
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
        sum = 0.0
        for t in map(lambda p: p.potential_at_coordinates(coordinates), self):
            sum += t
        return sum

    def deflection_angles_at_coordinates(self, coordinates):
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
        for t in map(lambda p: p.deflection_angles_at_coordinates(coordinates), self):
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

    def surface_density_func(self, eta):
        return self.einstein_radius_rescaled * eta ** (-(self.slope - 1))

    @profile.transform_coordinates
    def surface_density_at_coordinates(self, coordinates):
        """
        Calculate the projected surface density in dimensionless units at a given set of image plane coordinates.

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those coordinates
        """

        eta = self.coordinates_to_elliptical_radius(coordinates)

        return self.surface_density_func(eta)

    @property
    def potential_normalization(self):
        return 2.0 * self.einstein_radius_rescaled * self.axis_ratio / 2.0

    def potential_func(self, u, coordinates):
        eta = self.eta_u(u, coordinates)
        return (eta / u) * ((3.0 - self.slope) * eta) ** -1.0 * eta ** (3.0 - self.slope) / (
            (1 - (1 - self.axis_ratio ** 2) * u) ** (0.5))

    @profile.transform_coordinates
    def potential_at_coordinates(self, coordinates):
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
    def deflection_angles_at_coordinates(self, coordinates):
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

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))


class SphericalPowerLawMassProfile(EllipticalPowerLawMassProfile):
    """Represents a spherical power-law density distribution"""

    def __init__(self, einstein_radius, slope, centre=(0, 0)):
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

        super(SphericalPowerLawMassProfile, self).__init__(1.0, 0.0, einstein_radius, slope, centre)

    @property
    def deflection_normalization(self):
        return 2.0 * self.einstein_radius_rescaled

    @profile.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
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
        eta = self.coordinates_to_elliptical_radius(coordinates)
        deflection_r = self.deflection_normalization * ((3.0 - self.slope) * eta) ** -1.0 * eta ** (3.0 - self.slope)
        return self.coordinates_radius_to_x_and_y(coordinates, deflection_r)


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

    @property
    def deflection_normalization(self):
        return 2.0 * self.einstein_radius_rescaled * self.axis_ratio / (math.sqrt(1 - self.axis_ratio ** 2))

    @profile.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
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

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))


class SphericalIsothermalMassProfile(EllipticalIsothermalMassProfile):
    """Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
    density distribution for the value slope=2.0"""
    def __init__(self, einstein_radius, centre=(0.0, 0.0)):
        """

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profile
        einstein_radius : float
            Einstein radius of power-law mass profile
        """

        super(SphericalIsothermalMassProfile, self).__init__(1.0, 0.0, einstein_radius, centre)

    @profile.transform_coordinates
    def potential_at_coordinates(self, coordinates):
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
        eta = self.coordinates_to_elliptical_radius(coordinates)
        return self.deflection_normalization * eta

    @property
    def deflection_normalization(self):
        return 2.0 * self.einstein_radius_rescaled

    @profile.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
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
        return self.coordinates_radius_to_x_and_y(coordinates, self.deflection_normalization)


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


class CoredSphericalPowerLawMassProfile(CoredEllipticalPowerLawMassProfile):
    """Represents a cored spherical power-law density distribution"""

    def __init__(self, einstein_radius, slope, core_radius, centre=(0, 0)):
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
        super(CoredSphericalPowerLawMassProfile, self).__init__(1.0, 0.0, einstein_radius, slope, core_radius, centre)

    @property
    def einstein_radius_rescaled(self):
        return ((3 - self.slope) / (1 + self.axis_ratio)) * (self.einstein_radius + self.core_radius ** 2) ** (
            self.slope - 1)

    @property
    def deflection_normalization(self):
        return 2.0 * self.einstein_radius_rescaled

    @profile.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
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
        eta = self.coordinates_to_elliptical_radius(coordinates)
        deflection_r = self.deflection_normalization * ((3.0 - self.slope) * eta) ** -1.0 *  \
                       ( (self.core_radius ** 2 + eta) ** (3.0 - self.slope) - self.core_radius**(3-self.slope) )
        return self.coordinates_radius_to_x_and_y(coordinates, deflection_r)

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


class EllipticalNFWMassProfile(profile.EllipticalProfile, MassProfile):
    """The elliptical NFW profile, used to fit the dark matter halo of the lens."""

    def __init__(self, axis_ratio, phi, kappa_s, scale_radius, centre=(0, 0)):
        """ Setup a NFW dark matter profile.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profile
        axis_ratio : float
            Ratio of profile ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profile ellipse counter-clockwise from positive x-axis
        kappa_s : float
            The overall normalization of the dark matter halo
        scale_radius : float
            The radius containing half the light of this model
        """

        super(EllipticalNFWMassProfile, self).__init__(axis_ratio, phi, centre)
        super(MassProfile, self).__init__()
        self.kappa_s = kappa_s
        self.scale_radius = scale_radius

    @staticmethod
    def coord_func(r):
        if r > 1:
            return (1.0 / math.sqrt(r ** 2 - 1)) * math.atan(math.sqrt(r ** 2 - 1))
        elif r < 1:
            return (1.0 / math.sqrt(1 - r ** 2)) * math.atanh(math.sqrt(1 - r ** 2))
        elif r == 1:
            return 1

    @property
    def surface_density_normalization(self):
        return 2.0 * self.kappa_s

    def surface_density_func(self, eta):
        return self.surface_density_normalization * (1 - self.coord_func(eta)) / (eta ** 2 - 1)

    @profile.transform_coordinates
    def surface_density_at_coordinates(self, coordinates):
        """
        Calculate the projected surface density in dimensionless units at a given set of image plane coordinates.

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those coordinates
        """

        eta = (1.0 / self.scale_radius) * self.coordinates_to_elliptical_radius(coordinates)

        return self.surface_density_func(eta)

    @property
    def potential_normalization(self):
        return 4.0 * self.kappa_s * self.scale_radius

    def potential_func(self, u, coordinates):
        eta = (1.0 / self.scale_radius) * self.eta_u(u, coordinates)
        return (eta / u) * ((math.log(eta / 2.0) + self.coord_func(eta)) / eta) / (
            (1 - (1 - self.axis_ratio ** 2) * u) ** (0.5))

    @profile.transform_coordinates
    def potential_at_coordinates(self, coordinates):
        """
        Calculate the projected gravitational potential in dimensionless units at a given set of image plane coordinates.

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those coordinates
        """
        potential = quad(self.potential_func, a=0.0, b=1.0, args=(coordinates,))[0]
        return self.potential_normalization * potential

    @property
    def deflection_normalization(self):
        return self.axis_ratio

    def deflection_func(self, u, coordinates, npow):
        eta_u = (1.0 / self.scale_radius) * self.eta_u(u, coordinates)
        return self.surface_density_func(eta_u) / ((1 - (1 - self.axis_ratio ** 2) * u) ** (npow + 0.5))

    @profile.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
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

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))


class SphericalNFWMassProfile(EllipticalNFWMassProfile):
    """The spherical NFW profile, used to fit the dark matter halo of the lens."""

    def __init__(self, kappa_s, scale_radius, centre=(0, 0)):
        """ Setup a NFW dark matter profile.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profile
        kappa_s : float
            The overall normalization of the dark matter halo
        scale_radius : float
            The radius containing half the light of this model
        """

        super(SphericalNFWMassProfile, self).__init__(1.0, 0.0, kappa_s, scale_radius, centre)

    # TODO : There is a factor of kappa_s difference between the Elliptical and Spherical NFW profiles, even though
    # TODO : I've checked the equations are copied correct. Figure out why...

    @property
    def potential_normalization(self):
        return 2.0 * (self.scale_radius ** 2) # *self.kappa_s

    def potential_func_sph(self, eta):
        return ((math.log(eta/2.0)**2) - math.atanh(math.sqrt(1 - eta**2))**2)

    # TODO : The 'func' routines require a different input to the elliptical cases, meaning they cannot be over-ridden.
    # TODO : Should be able to refactor code to deal with this nicely, but will wait until wwe're clear on numba.

    @profile.transform_coordinates
    def potential_at_coordinates(self, coordinates):
        """
        Calculate the projected gravitational potential in dimensionless units at a given set of image plane coordinates.

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those coordinates
        """
        eta = (1.0 / self.scale_radius) * self.coordinates_to_elliptical_radius(coordinates)
        return self.potential_normalization * self.potential_func_sph(eta)

    @property
    def deflection_normalization(self):
        return 4.0 * self.kappa_s * self.scale_radius

    def deflection_func_sph(self, eta):
        return (math.log(eta/2.0) + self.coord_func(eta)) / eta

    @profile.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
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
        eta = (1.0 / self.scale_radius) * self.coordinates_to_elliptical_radius(coordinates)
        deflection_r = self.deflection_normalization * self.deflection_func_sph(eta)

        return self.coordinates_radius_to_x_and_y(coordinates, deflection_r)


class SersicMassProfile(light_profile.SersicLightProfile, MassProfile):
    """The Sersic light profile, used to fit and subtract the lens galaxy's light and model its mass."""

    def __init__(self, axis_ratio, phi, flux, effective_radius, sersic_index, mass_to_light_ratio, centre=(0, 0)):
        """
        Setup a Sersic mass and light profile.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profile
        axis_ratio : float
            Ratio of profile ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profile ellipse counter-clockwise from positive x-axis
        flux : float
            Overall flux intensity normalisation in the light profile (electrons per second)
        effective_radius : float
            The radius containing half the light of this model
        sersic_index : Int
            The concentration of the light profile
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profile
        """
        super(SersicMassProfile, self).__init__(axis_ratio, phi, flux, effective_radius, sersic_index, centre)
        super(MassProfile, self).__init__()
        self.mass_to_light_ratio = mass_to_light_ratio

    @classmethod
    def from_sersic_light_profile(cls, sersic_light_profile, mass_to_light_ratio):
        return SersicMassProfile.from_profile(sersic_light_profile, mass_to_light_ratio=mass_to_light_ratio)

    @profile.transform_coordinates
    def surface_density_at_coordinates(self, coordinates):
        """Calculate the projected surface density in dimensionless units at a given set of image plane coordinates.

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The surface density [kappa(eta)] (r-direction) at those coordinates
        """
        return self.mass_to_light_ratio * self.flux_at_coordinates(coordinates)

    @property
    def deflection_normalization(self):
        return self.mass_to_light_ratio * self.axis_ratio

    def deflection_func(self, u, coordinates, npow):
        eta_u = math.sqrt(self.axis_ratio) * self.eta_u(u, coordinates)
        return self.flux_at_radius(eta_u) / ((1 - (1 - self.axis_ratio ** 2) * u) ** (npow + 0.5))

    @profile.transform_coordinates
    def deflection_angles_at_coordinates(self, coordinates):
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

        return self.rotate_coordinates_from_profile((deflection_x, deflection_y))

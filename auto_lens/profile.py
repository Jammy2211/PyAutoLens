import math
import numpy as np


class EllipticalProfile(object):
    """Generic elliptical profile class to contain functions shared by light and mass profiles"""

    def __init__(self, x_cen, y_cen, axis_ratio, phi):
        """
        Parameters
        ----------
        x_cen : float
            x-coordinate of mass profile centre
        y_cen : float
            y-coordinate of mass profile centre
        axis_ratio : float
            Ratio of mass profile ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profile ellipse counter-clockwise from positive x-axis
        """

        self.x_cen = x_cen
        self.y_cen = y_cen
        self.axis_ratio = axis_ratio
        self.phi = phi

    @property
    def cos_phi(self):
        return self.angles_from_x_axis()[0]

    @property
    def sin_phi(self):
        return self.angles_from_x_axis()[1]

    def angles_from_x_axis(self):
        """
        Determine the sin and cosine of the angle between the mass-profile ellipse and positive x-axis, \
        defined counter-clockwise from x.

        Returns
        -------
        The sin and cosine of the angle
        """
        phi_radians = math.radians(self.phi)
        return math.cos(phi_radians), math.sin(phi_radians)

    def coordinates_to_centre(self, coordinates):
        """
        Converts image coordinates to mass profile's centre

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The coordinates at the mass profile centre
        """
        return coordinates[0] - self.x_cen, coordinates[1] - self.y_cen

    def coordinates_to_radius(self, coordinates):
        """
        Compute the distance of image coordinates from (0.0). which should be the mass profile centre

        Parameters
        ----------
        coordinates : (float, float)
            The image coordinates shifted to the mass profile centre (x, y)

        Returns
        -------
        The radius at those coordinates
        """
        shifted_coordinates = self.coordinates_to_centre(coordinates)

        return math.sqrt(shifted_coordinates[0] ** 2 + shifted_coordinates[1] ** 2)

    # TODO: This isn't using any variable from the class. Should it be?
    @staticmethod
    def coordinates_angle_from_x(coordinates):
        """
        Compute the angle between the coordinates and positive x-axis, defined counter-clockwise. Elliptical profiles
        are symmetric after 180 degrees, so angles above 180 are converted to their equipvalent value from 0.
        (e.g. 225 degrees counter-clockwise from the x-axis is equivalent to 45 degrees counter-clockwise)

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image.

        Returns
        ----------
        The angle between the coordinates and the x-axis and mass profile centre
        """
        # TODO : Make a property and / or class variable? <- doesn't really make sense if you're inputting coordinates
        theta_from_x = math.degrees(np.arctan2(coordinates[1], coordinates[0]))
        if theta_from_x < 0:
            theta_from_x += 180
        return theta_from_x

    def coordinates_angle_to_mass_profile(self, theta):
        """
        Compute the sin and cosine of the angle between the shifted coordinates and elliptical mass-profile

        Parameters
        ----------
        theta : Float

        Returns
        ----------
        The sin and cosine of the angle between the shifted coordinates and mass-profile ellipse.
        """
        # TODO: Set up using class variables / a property? <- As above, if you're passing stuff in to the class to get
        # TODO: a result it doesn't really make sense for it to be a property
        theta_coordinate_to_mass = math.radians(theta - self.phi)
        return math.cos(theta_coordinate_to_mass), math.sin(theta_coordinate_to_mass)

    def coordinates_back_to_cartesian(self, coordinates_elliptical):
        """
        Rotate elliptical coordinates back to the original Cartesian grid

        Parameters
        ----------
        coordinates_elliptical : (float, float)
            The x and y coordinates of the image translated to the elliptical coordinate system

        Returns
        ----------
        The coordinates (typically deflection angles) on a regular Cartesian grid
        """
        x_elliptical = coordinates_elliptical[0]
        x = (x_elliptical * self.cos_phi - coordinates_elliptical[1] * self.sin_phi)
        y = (+x_elliptical * self.sin_phi + coordinates_elliptical[1] * self.cos_phi)
        return x, y

    def coordinates_rotate_to_ellipse(self, coordinates):
        """
        Translate Cartesian image coordinates to elliptical mass profile's reference frame

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The coordinates after the elliptical translation
        """

        # TODO: All components below are unit tested, need to add tests for this entire function

        # TODO: This function was made more simple by hiding the coordinate shift calculation in the coordinates to
        # TODO: radius

        # Compute their distance to this centre
        radius = self.coordinates_to_radius(coordinates)

        # Compute the angle between the coordinates and x-axis
        theta_from_x = self.coordinates_angle_from_x(coordinates)

        # Compute the angle between the coordinates and mass-profile ellipse
        cos_theta, sin_theta = self.coordinates_angle_to_mass_profile(theta_from_x)

        # Multiply by radius to get their x / y distance from the mass profile centre in this elliptical unit system
        return radius * cos_theta, radius * sin_theta


class SersicLightProfile(EllipticalProfile):
    """Used to fit the light of a galaxy. It can produce both highly concentrated light profiles (for high Sersic Index)
     or extended flatter profiles (for low Sersic Index)."""

    def __init__(self, x_cen, y_cen, axis_ratio, phi, flux, effective_radius, sersic_index):
        """

        Parameters
        ----------
        x_cen : float
            x-coordinate of mass profile centre
        y_cen : float
            y-coordinate of mass profile centre
        axis_ratio : float
            Ratio of mass profile ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profile ellipse counter-clockwise from positive x-axis
        flux : float
            Overall flux intensity normalisation in the light profile (electrons per second)
        effective_radius : float
            The radius containing half the light of this model
        sersic_index : Int
            The concentration of the light profile
        """
        super(SersicLightProfile, self).__init__(x_cen, y_cen, axis_ratio, phi)
        self.flux = flux
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    @property
    def sersic_constant(self):
        """

        Returns
        -------
        sersic_constant: float
            A parameter, derived from sersic_index, that ensures that effective_radius always contains 50% of the light.
        """
        return (2 * self.sersic_index) - (1. / 3.) + (4. / (405. * self.sersic_index)) + (
            46. / (25515. * self.sersic_index ** 2)) + (131. / (1148175. * self.sersic_index ** 3)) - (
                   2194697. / (30690717750. * self.sersic_index ** 4))

    def flux_at_radius(self, radius):
        """

        Parameters
        ----------
        radius : float
            The distance from the centre of the profile
        Returns
        -------
        flux: float
            The flux at that radius
        """
        return self.flux * math.exp(
            -self.sersic_constant * (((radius / self.effective_radius) ** (1. / self.sersic_index)) - 1))


class ExponentialLightProfile(SersicLightProfile):
    """Used to fit flatter regions of light in a galaxy, typically its disks or stellar halo. It is a subset of the
    Sersic profile, corresponding exactly to the solution sersic_index = 1"""

    def __init__(self, x_cen, y_cen, axis_ratio, phi, flux, effective_radius):
        """

        Parameters
        ----------
        x_cen : float
            x-coordinate of mass profile centre
        y_cen : float
            y-coordinate of mass profile centre
        axis_ratio : float
            Ratio of mass profile ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profile ellipse counter-clockwise from positive x-axis
        flux : float
            Overall flux intensity normalisation in the light profile (electrons per second)
        effective_radius : float
            The radius containing half the light of this model
        """
        super(ExponentialLightProfile, self).__init__(x_cen, y_cen, axis_ratio, phi, flux, effective_radius, 1)


class DevVaucouleursLightProfile(SersicLightProfile):
    """Used to fit the concentrated regions of light in a galaxy, typically its bulge. It may also fit the entire light
    profile of an elliptical / early-type galaxy. It is a subset of the Sersic profile, corresponding exactly to the
    solution sersic_index = 4."""

    def __init__(self, x_cen, y_cen, axis_ratio, phi, flux, effective_radius):
        """

        Parameters
        ----------
        x_cen : float
            x-coordinate of mass profile centre
        y_cen : float
            y-coordinate of mass profile centre
        axis_ratio : float
            Ratio of mass profile ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profile ellipse counter-clockwise from positive x-axis
        flux : float
            Overall flux intensity normalisation in the light profile (electrons per second)
        effective_radius : float
            The radius containing half the light of this model
        """
        super(DevVaucouleursLightProfile, self).__init__(x_cen, y_cen, axis_ratio, phi, flux, effective_radius, 4)


class CoreSersicLightProfile(SersicLightProfile):
    """The Core-Sersic profile is used to fit the light of a galaxy. It is an extension of the Sersic profile and
    flattens the light profiles central values (compared to the extrapolation of a pure Sersic profile), by forcing
    these central regions to behave instead as a power-law."""

    def __init__(self, x_cen, y_cen, axis_ratio, phi, flux, effective_radius, sersic_index, radius_break, flux_break,
                 gamma, alpha):
        """

        Parameters
        ----------
        x_cen : float
            x-coordinate of mass profile centre
        y_cen : float
            y-coordinate of mass profile centre
        axis_ratio : float
            Ratio of mass profile ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profile ellipse counter-clockwise from positive x-axis
        flux : float
            Overall flux intensity normalisation in the light profile (electrons per second)
        effective_radius : float
            The radius containing half the light of this model
        sersic_index : Int
            The concentration of the light profile
        radius_break : Float
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        flux_break : Float
            The intensity at the break radius.
        gamma : Float
            The logarithmic power-law slope of the inner core profile
        alpha :
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """
        super(CoreSersicLightProfile, self).__init__(x_cen, y_cen, axis_ratio, phi, flux, effective_radius,
                                                     sersic_index)
        self.radius_break = radius_break
        self.flux_break = flux_break
        self.alpha = alpha
        self.gamma = gamma

    @property
    def flux_prime(self):
        """Overall flux intensity normalisation in the rescaled Core-Sersic light profile (electrons per second)"""
        return self.flux_break * (2.0 ** (-self.gamma / self.alpha)) * math.exp(
            self.sersic_constant * (((2.0 ** (1.0 / self.alpha)) * self.radius_break) / self.effective_radius) ** (
                1.0 / self.sersic_index))

    def flux_at_radius(self, radius):
        """

        Parameters
        ----------
        radius : float
            The distance from the centre of the profile
        Returns
        -------
        flux: float
            The flux at that radius
        """
        return self.flux_prime * (
            (1 + ((self.radius_break / radius) ** self.alpha)) ** (self.gamma / self.alpha)) * math.exp(
            -self.sersic_constant * (
                (((radius ** self.alpha) + (self.radius_break ** self.alpha)) / (
                    self.effective_radius ** self.alpha)) ** (
                    1.0 / (self.alpha * self.sersic_index))))


class EllipticalPowerLawMassProfile(EllipticalProfile):
    """Represents an elliptical power-law density distribution"""

    def __init__(self, x_cen, y_cen, axis_ratio, phi, einstein_radius, slope):
        """

        Parameters
        ----------
        x_cen : float
            x-coordinate of mass profile centre
        y_cen : float
            y-coordinate of mass profile centre
        axis_ratio : float
            Ratio of mass profile ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of mass profile ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profile
        slope : float
            power-law density slope of mass profile
        """

        super(EllipticalPowerLawMassProfile, self).__init__(x_cen, y_cen, axis_ratio, phi)

        self.einstein_radius = einstein_radius
        self.slope = slope

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

        # TODO : Unit tests missing - need to sort out scipy.integrate

        coordinates_elliptical = self.coordinates_rotate_to_ellipse(coordinates)

        # TODO: implement a numerical integrator for this profile using scipy and / or c++

        # defl_elliptical = scipy.integrate(coordinates_elliptical, kappa_power_law)
        # defl_angles = self.coordinates_back_to_cartesian(coordinates_elliptical=defl_elliptical)
        # defl_angles = self.normalization*defl_angles
        # return defl_angles

        pass

    @property
    def einstein_radius_rescaled(self):
        return ((3 - self.slope) / (1 + self.axis_ratio)) * self.einstein_radius

    def compute_deflection_angles_fast(self, coordinate_list):
        """Place holder for what a c++ deflection angle call will look like"""

        # import fast_lib.defl.PowerLawElliptical

        # return fast_lib.defl.PowerLawElliptical(coordinate_list, self.x_cen, self.y_cen, self.axis_ratio, self.phi,
        #                                   self.einstein_radius, self.slope)

        pass

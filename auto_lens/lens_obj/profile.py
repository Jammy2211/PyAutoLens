import math


# TODO: I've made this all one module for now. As complexity increases it may make sense to have three modules. Even so,
# TODO: The similarities between mass and light profiles makes lumping them together very tempting.

# TODO: It looks like mass and light profiles share a lot of methods so let's use a parent class for both
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

    # TODO: Properties are nice. Rather than calculating lots of stuff in the constructor we can break it down into
    # TODO: functions that are executed on the fly.
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
        # TODO: Note that tuples have the type of their components. "tuple" isn't a type but (float, float) is.
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

        # TODO: It depends how you're going to use it but the code can be more elegant if you use the internal state
        # TODO: of the class to define things. If we did move the centre profile around it would still work without the
        # TODO: user of the class having to understand the internal implementation

        # TODO: DRY (don't repeat yourself)
        shifted_coordinates = self.coordinates_to_centre(coordinates)

        return math.sqrt(shifted_coordinates[0] ** 2 + shifted_coordinates[1] ** 2)

    def coordinates_angle_from_x(self, coordinates):
        """
        Computes sin and cosine of the angle between the shifted coordinates andd positive x-axis, \
        defined counter-clockwise.

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image.

        Returns
        ----------
        The angle between the coordinates and the x-axis and mass profile centre
        """
        # TODO: For example, here we can calculate the radius internally.
        radius = self.coordinates_to_radius(coordinates)
        return coordinates[0] / radius, coordinates[1] / radius

    def coordinates_angle_to_mass_profile(self, cos_theta, sin_theta):
        """
        Compute the sin and cosine of the angle between the shifted coordinates and elliptical mass-profile

        Parameters
        ----------
        sin_theta
        cos_theta

        Returns
        ----------
        The sin and cosine of the angle between the shifted coordinates and mass-profile ellipse.
        """
        # TODO: OK, so phi describes the mass profile and theta is some coordinate. Why not pass in theta and determine
        # TODO: cos_theta and sin_theta using math.cos and math.sin?
        dum = cos_theta
        cos_theta = cos_theta * self.cos_phi + sin_theta * self.sin_phi
        sin_theta = sin_theta * self.cos_phi - dum * self.sin_phi
        return cos_theta, sin_theta

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
        dum = coordinates_elliptical[0]
        x = (dum * self.cos_phi - coordinates_elliptical[1] * self.sin_phi)
        y = (+dum * self.sin_phi + coordinates_elliptical[1] * self.cos_phi)
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
        cos_theta, sin_theta = self.coordinates_angle_from_x(coordinates)

        # Compute the angle between the coordinates and mass-profile ellipse
        cos_theta, sin_theta = self.coordinates_angle_to_mass_profile(cos_theta, sin_theta)

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

    # TODO: I haven't written a test for this. Presumably entering the effective radius would work? You might have to
    # TODO: sum flux values for a bunch of radii smaller than the effective radius and larger than the effective radius
    # TODO: and show that they're approximately equal? Or should flux_at_radius(effective_radius) == flux / 2?
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
        super(ExponentialLightProfile, self).__init__(x_cen, y_cen, axis_ratio, phi, flux, effective_radius, 4)


class DevVaucouleurs(SersicLightProfile):
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
        super(DevVaucouleurs, self).__init__(x_cen, y_cen, axis_ratio, phi, flux, effective_radius, 4)


class EllipticalPowerLaw(EllipticalProfile):
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
        slope : float
            The slope of the power law
        """

        super(EllipticalPowerLaw, self).__init__(x_cen, y_cen, axis_ratio, phi)

        self.einstein_radius = einstein_radius
        self.slope = slope

        # normalization used for power-law model, includes rescaling by axis ratio and density slope.
        self.normalization = (3 - slope) / (1 + axis_ratio)

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

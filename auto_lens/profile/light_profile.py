import profile
from matplotlib import pyplot
import numpy as np
import math


class LightProfile(object):
    """Mixin class that implements functions common to all light profiles"""

    # noinspection PyMethodMayBeStatic
    def flux_at_coordinates(self, coordinates):
        """
        Abstract method for obtaining flux at given coordinates
        Parameters
        ----------
        coordinates : (int, int)
            The coordinates in image space
        Returns
        -------
        flux : float
            The value of flux at the given coordinates
        """
        raise AssertionError("Flux at coordinates should be overridden")

    def plot(self, x_min=-5, y_min=-5, x_max=5, y_max=5, pixel_scale=0.1):
        """
        Draws a plot of this light profile. Upper normalisation limit determined by taking mean plus one standard
        deviation

        Parameters
        ----------
        pixel_scale : float
            The arcsecond (") size of each pixel
        x_min : int
            The minimum x bound
        y_min : int
            The minimum y bound
        x_max : int
            The maximum x bound
        y_max : int
            The maximum y bound

        """
        array = profile.array_function(self.flux_at_coordinates)(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                                                                 pixel_scale=pixel_scale)
        pyplot.imshow(array)
        pyplot.clim(vmax=np.mean(array) + np.std(array))
        pyplot.show()


class CombinedLightProfile(list, LightProfile):
    """A light profile comprising one or more light profiles"""

    def __init__(self, *light_profiles):
        super(CombinedLightProfile, self).__init__(light_profiles)

    def flux_at_coordinates(self, coordinates):
        """
        Method for obtaining flux at given coordinates
        Parameters
        ----------
        coordinates : (int, int)
            The coordinates in image space
        Returns
        -------
        flux : float
            The value of flux at the given coordinates
        """
        return sum(map(lambda p: p.flux_at_coordinates(coordinates), self))


class SersicLightProfile(profile.EllipticalProfile, LightProfile):
    """The Sersic light profile, used to fit and subtract the lens galaxy's light."""

    def __init__(self, axis_ratio, phi, flux, effective_radius, sersic_index, centre=(0, 0)):
        """

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
        """
        super(SersicLightProfile, self).__init__(axis_ratio, phi, centre)
        self.flux = flux
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    @property
    def elliptical_effective_radius(self):
        # Extra physical parameter not used by the model, but has value scientifically TODO: better doc
        return self.effective_radius / self.axis_ratio

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

    def flux_at_coordinates(self, coordinates):
        """
        Method for obtaining flux at given coordinates
        Parameters
        ----------
        coordinates : (int, int)
            The coordinates in image space
        Returns
        -------
        flux : float
            The value of flux at the given coordinates
        """
        radius = self.coordinates_to_eccentric_radius(coordinates)
        return self.flux_at_radius(radius)


class ExponentialLightProfile(SersicLightProfile):
    """Used to fit flatter regions of light in a galaxy, typically its disks or stellar halo. It is a subset of the
    Sersic profile, corresponding exactly to the solution sersic_index = 1"""

    def __init__(self, axis_ratio, phi, flux, effective_radius, centre=(0, 0)):
        """

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
        """
        super(ExponentialLightProfile, self).__init__(axis_ratio, phi, flux, effective_radius, 1, centre)


class DevVaucouleursLightProfile(SersicLightProfile):
    """Used to fit the concentrated regions of light in a galaxy, typically its bulge. It may also fit the entire light
    profile of an elliptical / early-type galaxy. It is a subset of the Sersic profile, corresponding exactly to the
    solution sersic_index = 4."""

    def __init__(self, axis_ratio, phi, flux, effective_radius, centre=(0, 0)):
        """

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
        """
        super(DevVaucouleursLightProfile, self).__init__(axis_ratio, phi, flux, effective_radius, 4, centre)


class CoreSersicLightProfile(SersicLightProfile):
    """The Core-Sersic profile is used to fit the light of a galaxy. It is an extension of the Sersic profile and
    flattens the light profiles central values (compared to the extrapolation of a pure Sersic profile), by forcing
    these central regions to behave instead as a power-law."""

    def __init__(self, axis_ratio, phi, flux, effective_radius, sersic_index, radius_break, flux_break,
                 gamma, alpha, centre=(0, 0)):
        """

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
        radius_break : Float
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        flux_break : Float
            The intensity at the break radius.
        gamma : Float
            The logarithmic power-law slope of the inner core profile
        alpha :
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """
        super(CoreSersicLightProfile, self).__init__(axis_ratio, phi, flux, effective_radius,
                                                     sersic_index, centre)
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

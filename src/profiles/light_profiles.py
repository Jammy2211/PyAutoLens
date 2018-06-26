from src.profiles import geometry_profiles
import math
from scipy.integrate import quad
import numpy as np
from itertools import count


class LightProfileSettings(object):
    # TODO: yuck. This functionality needs moving.

    def __init__(self, iterative_image_plane=True, iterative_precision=1e-4, sub_grid_plane=True,
                 image_plane_override=False):
        """The settings of a light profile, controlling the precision to which it is calculated and its behaviour in
         the ray-tracing calculation.

         Parameters
         -----------
         iterative_image_plane : bool
            *True* - If the light profile is in the image-plane, it is evaluated using an iterative sub-grid_coords.
            *False* - Evaluate the light profile on the grid_coords defined by the variable *sub_grid_plane*.
         iterative_precision : float
            The precision the light profile intensities are calculated too on the iterative grid_coords (image-plane only).
         sub_grid_plane : bool
            *True* - Evaluate the light profile using a sub-grid_coords.
            *False* - Evaluate the light profile using an image-grid_coords.
        image_plane_override : bool
            *True* - Evaluate the light profile in the image-plane, irrespective of it's true plane (e.g. lensed \
            quasar evaluation in the image-plane).
            *False* - Evaluate the light profile in its true plane.
        """
        self.iterative_image_plane = iterative_image_plane
        self.iterative_precision = iterative_precision
        self.sub_grid_plane = sub_grid_plane
        self.image_plane_override = image_plane_override


class LightProfile(object):
    """Mixin class that implements functions common to all light profiles"""

    # noinspection PyMethodMayBeStatic
    def intensity_at_radius(self, radius):
        """
        Abstract method for obtaining intensity at given radius
        Parameters
        ----------
        radius : float
            The distance from the centre of the profiles
        Returns
        -------
        intensity : float
            The value of intensity at the given radius
        """
        raise AssertionError("Flux at radius should be overridden")

    # noinspection PyMethodMayBeStatic
    def intensity_at_coordinates(self, coordinates):
        """
        Abstract method for obtaining intensity at given image_grid
        Parameters
        ----------
        coordinates : ndarray
            The image_grid in image_grid space (arc seconds)
        Returns
        -------
        intensity : float
            The value of intensity at the given image_grid
        """
        raise AssertionError("Flux at image_grid should be overridden")


class EllipticalLightProfile(geometry_profiles.EllipticalProfile, LightProfile):
    """Generic class for an elliptical light profiles"""

    _ids = count()

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, settings=LightProfileSettings()):
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
        super(EllipticalLightProfile, self).__init__(centre, axis_ratio, phi)
        self.axis_ratio = axis_ratio
        self.phi = phi
        self.settings = settings
        self.component_number = next(self._ids)

    @property
    def subscript(self):
        return 'l'

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi']

    def luminosity_within_circle(self, radius):
        """
        Compute the light profiles's total luminosity within a circle of specified radius. This is performed via \
        integration and is centred on the light model_mapper.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the luminosity within.

        Returns
        -------
        luminosity : float
            The total luminosity within the specified circle.
        """
        return quad(self.luminosity_integral, a=0.0, b=radius, args=(1.0,))[0]

    def luminosity_within_ellipse(self, major_axis):
        """
        Compute the light profiles's total luminosity within an ellipse of specified major axis. This is performed via\
        integration and is centred, oriented and aligned with on the light model_mapper.
        Parameters
        ----------
        major_axis: float
            The major-axis of the ellipse to compute the luminosity within.
        Returns
        -------
        intensity : float
            The total luminosity within the specified ellipse.
        """
        return quad(self.luminosity_integral, a=0.0, b=major_axis, args=(self.axis_ratio,))[0]

    def luminosity_integral(self, x, axis_ratio):
        """Routine to integrate an elliptical light profiles - set axis ratio to 1 to compute the luminosity within a \
        circle"""
        r = x * axis_ratio
        return 2 * math.pi * r * self.intensity_at_radius(x)


class EllipticalSersic(EllipticalLightProfile):
    """The Sersic light profiles, used to fit and subtract the lens galaxy's light."""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0, settings=LightProfileSettings()):
        """

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The circular radius containing half the light of this model_mapper
        sersic_index : Int
            The concentration of the light profiles
        """
        super(EllipticalSersic, self).__init__(centre, axis_ratio, phi, settings)
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R', 'n']

    @property
    def elliptical_effective_radius(self):
        """The effective_radius term used in a Sersic light profiles is the circular effective radius. It describes the
         radius within which a circular aperture contains half the light profiles's light. For elliptical (i.e low axis \
         ratio) systems, this circle won't robustly capture the light profiles's elliptical shape.

         The elliptical effective radius therefore instead describes the major-axis radius of the ellipse containing
         half the light, and may be more appropriate for pixelization of highly flattened systems like disk galaxies."""
        return self.effective_radius / math.sqrt(self.axis_ratio)

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

    def intensity_at_radius(self, radius):
        """

        Parameters
        ----------
        radius : float
            The distance from the centre of the profiles
        Returns
        -------
        intensity : float
            The intensity at that distance
        """
        return self.intensity * np.exp(
            -self.sersic_constant * (((radius / self.effective_radius) ** (1. / self.sersic_index)) - 1))

    @geometry_profiles.transform_coordinates
    def intensity_at_coordinates(self, coordinates):
        """
        Method for obtaining intensity at given image_grid.

        This includes a coordinate transform to the light profiles's shifted, rotated and elliptical reference frame.

        Parameters
        ----------
        coordinates : ndarray
            The image_grid in image_grid space
        Returns
        -------
        intensity : float
            The value of intensity at the given image_grid
        """
        return self.intensity_at_radius(self.coordinates_to_eccentric_radius(coordinates))


class EllipticalExponential(EllipticalSersic):
    """Used to fit flatter regions of light in a galaxy, typically a disk.

    It is a subset of the Sersic profiles, corresponding exactly to the solution sersic_index = 1"""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 settings=LightProfileSettings()):
        """

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The circular radius containing half the light of this model_mapper
        """
        super(EllipticalExponential, self).__init__(centre, axis_ratio, phi, intensity, effective_radius, 1.0, settings)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R']


class EllipticalDevVaucouleurs(EllipticalSersic):
    """Used to fit the concentrated regions of light in a galaxy, typically its bulge. It may also fit the entire light
    profiles of an elliptical / early-type galaxy.

    It is a subset of the Sersic profiles, corresponding exactly to the solution sersic_index = 4."""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 settings=LightProfileSettings()):
        """

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The circular radius containing half the light of this model_mapper
        """
        super(EllipticalDevVaucouleurs, self).__init__(centre, axis_ratio, phi, intensity, effective_radius, 4.0,
                                                       settings)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R']


class EllipticalCoreSersic(EllipticalSersic):
    """The Core-Sersic profiles is used to fit the light of a galaxy. It is an extension of the Sersic profiles and \
    flattens the light profiles central values (compared to the extrapolation of a pure Sersic profiles), by forcing \
    these central regions to behave instead as a power-law."""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0, radius_break=0.01, intensity_break=0.05, gamma=0.25, alpha=3.0,
                 settings=LightProfileSettings()):
        """

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The circular radius containing half the light of this model_mapper
        sersic_index : Int
            The concentration of the light profiles
        radius_break : Float
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        intensity_break : Float
            The intensity at the break radius.
        gamma : Float
            The logarithmic power-law slope of the inner core profiles
        alpha :
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """
        super(EllipticalCoreSersic, self).__init__(centre, axis_ratio, phi, intensity, effective_radius, sersic_index,
                                                   settings)
        self.radius_break = radius_break
        self.intensity_break = intensity_break
        self.alpha = alpha
        self.gamma = gamma

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R', 'n', 'Rb', 'Ib', '\gamma', r'\alpha']

    @property
    def intensity_prime(self):
        """Overall intensity normalisation in the rescaled Core-Sersic light profiles (electrons per second)"""
        return self.intensity_break * (2.0 ** (-self.gamma / self.alpha)) * math.exp(
            self.sersic_constant * (((2.0 ** (1.0 / self.alpha)) * self.radius_break) / self.effective_radius) ** (
                    1.0 / self.sersic_index))

    def intensity_at_radius(self, radius):
        """

        Parameters
        ----------
        radius : float
            The distance from the centre of the profiles
        Returns
        -------
        intensity : float
            The intensity at that radius
        """
        return self.intensity_prime * (
                (1 + ((self.radius_break / radius) ** self.alpha)) ** (self.gamma / self.alpha)) * np.exp(
            -self.sersic_constant * ((((radius ** self.alpha) + (self.radius_break ** self.alpha)) / (
                    self.effective_radius ** self.alpha)) ** (
                                             1.0 / (self.alpha * self.sersic_index))))

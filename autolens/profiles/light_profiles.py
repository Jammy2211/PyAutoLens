from autolens.profiles import geometry_profiles
from scipy.integrate import quad
import numpy as np


class LightProfile(object):
    """Mixin class that implements functions common to all light profiles"""

    def intensities_from_grid_radii(self, grid_radii):
        """
        Abstract method for obtaining intensity at on a grid of radii.

        Parameters
        ----------
        grid_radii : float
            The radial distance from the centre of the profiles for each coordinate on the grid.
        """
        raise NotImplementedError("intensity_at_radius should be overridden")

    # noinspection PyMethodMayBeStatic
    def intensities_from_grid(self, grid):
        """
        Abstract method for obtaining intensity at a grid of Cartesian (x,y) coordinates.

        Parameters
        ----------
        grid : ndarray
            The (x, y) coordinates in the original reference frame of the observed image.
        Returns
        -------
        intensity : float
            The value of intensity at the given radius
        """
        raise NotImplementedError("intensity_from_grid should be overridden")


# noinspection PyAbstractClass
class EllipticalLP(geometry_profiles.EllipticalProfile, LightProfile):
    """Generic class for an elliptical light profiles"""

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0):
        """  Abstract class for an elliptical light-profile.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        """
        super(EllipticalLP, self).__init__(centre, axis_ratio, phi)

    def luminosity_within_circle(self, radius):
        """
        Compute the light profiles's total luminosity within a circle of specified radius. This is performed via \
        integration_old and is centred on the light profile's centre.

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
        integration_old and is centred, oriented and aligned with on the ellipitcal light profile.

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
        """Routine to integrate the luminosity of an elliptical light profile.

        The axis ratio is set to 1.0 for computing the luminosity within a circle"""
        r = x * axis_ratio
        return 2 * np.pi * r * self.intensities_from_grid_radii(x)


class EllipticalGaussian(EllipticalLP):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, sigma=0.01):
        """ The elliptical Gaussian profile, used for modeling a PSF.

        Parameters
        ----------
        centre: (float, float)
            The centre of the light profile.
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        sigma : float
            The full-width half-maximum of the Gaussian.
        """
        super(EllipticalGaussian, self).__init__(centre, axis_ratio, phi)

        self.intensity = intensity
        self.sigma = sigma

    def intensities_from_grid_radii(self, grid_radii):
        """
        Calculate the intensity of the Gaussian light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii : float
            The radial distance from the centre of the profiles for each coordinate on the grid.
        """
        return np.multiply(np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
                           np.exp(-0.5 * np.square(np.divide(grid_radii, self.sigma))))

    @geometry_profiles.transform_grid
    def intensities_from_grid(self, grid):
        """
        Calculate the intensity of the light profile on a grid of Cartesian (x,y) coordinates.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid : ndarray
            The (x, y) coordinates in the original reference frame of the observed image.
        Returns
        -------
        intensity : float
            The value of intensity at the given radius
        """
        return self.intensities_from_grid_radii(self.grid_to_elliptical_radii(grid))


class SphericalGaussian(EllipticalGaussian):

    def __init__(self, centre=(0.0, 0.0), intensity=0.1, sigma=0.01):
        """ The spherical Gaussian profile, used for modeling a PSF.

        Parameters
        ----------
        centre: (float, float)
            The centre of the light profile.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        sigma : float
            The full-width half-maximum of the Gaussian.
        """
        super(SphericalGaussian, self).__init__(centre, 1.0, 0.0, intensity, sigma)


class EllipticalSersic(geometry_profiles.EllipticalSersic, EllipticalLP):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0):
        """ The elliptical Sersic profile, used for fitting a galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The centre of the light profile.
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : Int
            Controls the concentration of the of the light profile.
        """
        super(EllipticalSersic, self).__init__(centre, axis_ratio, phi, intensity, effective_radius,
                                               sersic_index)

    def intensities_from_grid_radii(self, grid_radii):
        """
        Calculate the intensity of the Sersic light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii : float
            The radial distance from the centre of the profiles for each coordinate on the grid.
        """
        return np.multiply(self.intensity, np.exp(
            np.multiply(-self.sersic_constant,
                        np.add(np.power(np.divide(grid_radii, self.effective_radius), 1. / self.sersic_index), -1))))

    @geometry_profiles.transform_grid
    def intensities_from_grid(self, grid):
        """
        Calculate the intensity of the light profile on a grid of Cartesian (x,y) coordinates.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid : ndarray
            The (x, y) coordinates in the original reference frame of the observed image.
        Returns
        -------
        intensity : float
            The value of intensity at the given radius
        """
        return self.intensities_from_grid_radii(self.grid_to_eccentric_radii(grid))


class SphericalSersic(EllipticalSersic):

    def __init__(self, centre=(0.0, 0.0), intensity=0.1, effective_radius=0.6, sersic_index=4.0):
        """ The spherical Sersic profile, used for fitting a galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The centre of the light profile.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : Int
            Controls the concentration of the of the light profile.
        """
        super(SphericalSersic, self).__init__(centre, 1.0, 0.0, intensity, effective_radius, sersic_index)


class EllipticalExponential(EllipticalSersic):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6):
        """ The elliptical exponential profile, used for fitting a galaxy's light.

        This is a subset of the elliptical Sersic profile, specific to the case that sersic_index = 1.0.

        Parameters
        ----------
        centre: (float, float)
            The centre of the light profile.
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        """
        super(EllipticalExponential, self).__init__(centre, axis_ratio, phi, intensity, effective_radius, 1.0)


class SphericalExponential(EllipticalExponential):

    def __init__(self, centre=(0.0, 0.0), intensity=0.1, effective_radius=0.6):
        """ The spherical exponential profile, used for fitting a galaxy's light.

        This is a subset of the elliptical Sersic profile, specific to the case that sersic_index = 1.0.

        Parameters
        ----------
        centre: (float, float)
            The centre of the light profile.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        """
        super(SphericalExponential, self).__init__(centre, 1.0, 0.0, intensity, effective_radius)


class EllipticalDevVaucouleurs(EllipticalSersic):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6):
        """ The elliptical Dev Vaucouleurs profile, used for fitting a galaxy's light.

        This is a subset of the elliptical Sersic profile, specific to the case that sersic_index = 4.0.

        Parameters
        ----------
        centre: (float, float)
            The centre of the light profile.
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        """
        super(EllipticalDevVaucouleurs, self).__init__(centre, axis_ratio, phi, intensity, effective_radius, 4.0)


class SphericalDevVaucouleurs(EllipticalDevVaucouleurs):

    def __init__(self, centre=(0.0, 0.0), intensity=0.1, effective_radius=0.6):
        """ The spherical Dev Vaucouleurs profile, used for fitting a galaxy's light.

        This is a subset of the elliptical Sersic profile, specific to the case that sersic_index = 1.0.

        Parameters
        ----------
        centre: (float, float)
            The centre of the light profile.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        """
        super(SphericalDevVaucouleurs, self).__init__(centre, 1.0, 0.0, intensity, effective_radius)


class EllipticalCoreSersic(EllipticalSersic):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0, radius_break=0.01, intensity_break=0.05, gamma=0.25, alpha=3.0):
        """ The elliptical cored-Sersic profile, used for fitting a galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The centre of the light profile.
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : Int
            Controls the concetration of the of the light profile.
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
                                                   )
        self.radius_break = radius_break
        self.intensity_break = intensity_break
        self.alpha = alpha
        self.gamma = gamma

    @property
    def intensity_prime(self):
        """Overall intensity normalisation in the rescaled Core-Sersic light profiles (electrons per second)"""
        return self.intensity_break * (2.0 ** (-self.gamma / self.alpha)) * np.exp(
            self.sersic_constant * (((2.0 ** (1.0 / self.alpha)) * self.radius_break) / self.effective_radius) ** (
                    1.0 / self.sersic_index))

    def intensities_from_grid_radii(self, grid_radii):
        """
        Calculate the intensity of the cored-Sersic light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii : float
            The radial distance from the centre of the profiles for each coordinate on the grid.
        """
        return np.multiply(np.multiply(self.intensity_prime, np.power(
            np.add(1, np.power(np.divide(self.radius_break, grid_radii), self.alpha)), (self.gamma / self.alpha))),
                           np.exp(np.multiply(-self.sersic_constant,
                                              (np.power(np.divide(np.add(np.power(grid_radii, self.alpha), (
                                                      self.radius_break ** self.alpha)),
                                                                  (self.effective_radius ** self.alpha)), (
                                                                1.0 / (self.alpha * self.sersic_index)))))))


class SphericalCoreSersic(EllipticalCoreSersic):

    def __init__(self, centre=(0.0, 0.0), intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0, radius_break=0.01, intensity_break=0.05, gamma=0.25, alpha=3.0):
        """ The elliptical cored-Sersic profile, used for fitting a galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The centre of the light profile.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : Int
            Controls the concetration of the of the light profile.
        radius_break : Float
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        intensity_break : Float
            The intensity at the break radius.
        gamma : Float
            The logarithmic power-law slope of the inner core profiles
        alpha :
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """
        super(SphericalCoreSersic, self).__init__(centre, 1.0, 0.0, intensity, effective_radius, sersic_index,
                                                  radius_break, intensity_break, gamma, alpha)
        self.radius_break = radius_break
        self.intensity_break = intensity_break
        self.alpha = alpha
        self.gamma = gamma

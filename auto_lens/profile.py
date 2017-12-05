import math
import numpy as np
from matplotlib import pyplot
from functools import wraps


class EllipticalProfile(object):
    """Generic elliptical profile class to contain functions shared by light and mass profiles"""

    def __init__(self, axis_ratio, phi, centre=(0, 0)):
        """
        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profile
        axis_ratio : float
            Ratio of profile ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profile ellipse counter-clockwise from positive x-axis
        """

        self.centre = centre
        self.axis_ratio = axis_ratio
        self.phi = phi

    @property
    def x_cen(self):
        return self.centre[0]

    @property
    def y_cen(self):
        return self.centre[1]

    @property
    def cos_phi(self):
        return self.angles_from_x_axis()[0]

    @property
    def sin_phi(self):
        return self.angles_from_x_axis()[1]

    def angles_from_x_axis(self):
        """
        Determine the sin and cosine of the angle between the profile ellipse and positive x-axis, \
        defined counter-clockwise from x.

        Returns
        -------
        The sin and cosine of the angle
        """
        phi_radians = math.radians(self.phi)
        return math.cos(phi_radians), math.sin(phi_radians)

    def coordinates_to_centre(self, coordinates):
        """
        Converts image coordinates to profile's centre

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
        Convert the coordinates to a radius

        Parameters
        ----------
        coordinates : (float, float)
            The image coordinates (x, y)

        Returns
        -------
        The radius at those coordinates
        """
        shifted_coordinates = self.coordinates_to_centre(coordinates)
        return math.sqrt(shifted_coordinates[0] ** 2 + shifted_coordinates[1] ** 2)

    def coordinates_to_eccentric_radius(self, coordinates):
        """
        Convert the coordinates to a radius in elliptical space.

        Parameters
        ----------
        coordinates : (float, float)
            The image coordinates (x, y)
        Returns
        -------
        The radius at those coordinates
        """

        shifted_coordinates = self.coordinates_rotate_to_elliptical(coordinates)
        return math.sqrt(self.axis_ratio) * math.sqrt(
            shifted_coordinates[0] ** 2 + (shifted_coordinates[1] / self.axis_ratio) ** 2)

    # TODO: This isn't using any variable from the class. Should it be?
    @staticmethod
    def coordinates_angle_from_x(coordinates):
        """
        Compute the angle between the coordinates and positive x-axis, defined counter-clockwise. Elliptical profiles
        are symmetric after 180 degrees, so angles above 180 are converted to their equivalent value from 0.
        (e.g. 225 degrees counter-clockwise from the x-axis is equivalent to 45 degrees counter-clockwise)

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image.

        Returns
        ----------
        The angle between the coordinates and the x-axis and profile centre
        """
        theta_from_x = math.degrees(np.arctan2(coordinates[1], coordinates[0]))

        return theta_from_x

    def coordinates_angle_to_profile(self, theta):
        """
        Compute the sin and cosine of the angle between the shifted coordinates and elliptical profile

        Parameters
        ----------
        theta : Float

        Returns
        ----------
        The sin and cosine of the angle between the shifted coordinates and profile ellipse.
        """
        theta_coordinate_to_profile = math.radians(theta - self.phi)
        return math.cos(theta_coordinate_to_profile), math.sin(theta_coordinate_to_profile)

    def coordinates_back_to_cartesian(self, coordinates_elliptical):
        """
        Rotate elliptical coordinates back to the original Cartesian grid (for a circular profile this
        returns the input coordinates)

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

    def coordinates_rotate_to_elliptical(self, coordinates):
        """
        Translate Cartesian image coordinates to the lens profile's reference frame (for a circular profile this
        returns the input coordinates)

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The coordinates after the elliptical translation
        """

        # Compute distance of coordinates to the lens profile centre
        radius = self.coordinates_to_radius(coordinates)

        # Shift coordinates to lens profile centre (this is performed internally in the function above)
        shifted_coordinates = self.coordinates_to_centre(coordinates)

        # Compute the angle between the coordinates and x-axis
        theta_from_x = self.coordinates_angle_from_x(shifted_coordinates)

        # Compute the angle between the coordinates and profile ellipse
        cos_theta, sin_theta = self.coordinates_angle_to_profile(theta_from_x)

        # Multiply by radius to get their x / y distance from the profile centre in this elliptical unit system
        return radius * cos_theta, radius * sin_theta


class SphericalProfile(EllipticalProfile):
    """Generic circular profile class to contain functions shared by light and mass profiles"""

    def __init__(self, centre=(0, 0)):
        """
        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profile
        """
        super(SphericalProfile, self).__init__(1.0, 0.0, centre)


def array_for_function(func, x_min, y_min, x_max, y_max, pixel_scale, mask=None):
    """

    Parameters
    ----------
    mask : Mask
        An object that has an is_masked method which returns True if (x, y) coordinates should be masked (i.e. not
        return a value)
    func : function(coordinates)
        A function that takes coordinates and returns a value
    x_min : float
        The minimum x bound
    y_min : float
        The minimum y bound
    x_max : float
        The maximum x bound
    y_max : float
        The maximum y bound
    pixel_scale : float
        The arcsecond (") size of each pixel

    Returns
    -------
    array
        A 2D numpy array of values returned by the function at each coordinate
    """
    x_size = side_length(x_min, x_max, pixel_scale)
    y_size = side_length(y_min, y_max, pixel_scale)

    array = []

    for i in range(x_size):
        row = []
        for j in range(y_size):
            x = pixel_to_coordinate(x_min, pixel_scale, i)
            y = pixel_to_coordinate(y_min, pixel_scale, j)
            if mask is not None and mask.is_masked((x, y)):
                row.append(None)
            else:
                row.append(func((x, y)))
        array.append(row)
    # This conversion was to resolve a bug with putting tuples in the array. It might increase execution time.
    return np.array(array)


def side_length(dim_min, dim_max, pixel_scale):
    return int((dim_max - dim_min) / pixel_scale)


def avg(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        """

        Parameters
        ----------
        results : Sized
            A collection of numerical values or tuples
        Returns
        -------
            The logical average of that collection
        """
        try:
            return sum(results) / len(results)
        except TypeError:
            sum_tuple = (0, 0)
            for t in results:
                sum_tuple = (sum_tuple[0] + t[0], sum_tuple[1] + t[1])
            return sum_tuple[0] / len(results), sum_tuple[1] / len(results)

    return wrapper


def subgrid(func):
    """
    Decorator to permit generic subgridding
    Parameters
    ----------
    func : function(coordinates) -> value OR (value, value)
        Function that takes coordinates and calculates some value
    Returns
    -------
    func: function(coordinates, pixel_scale, grid_size)
        Function that takes coordinates and pixel scale/grid_size required for subgridding
    """

    @wraps(func)
    def wrapper(self, coordinates, pixel_scale=0.1, grid_size=1):
        """

        Parameters
        ----------
        self
        coordinates : (float, float)
            A coordinate pair
        pixel_scale : float
            The scale of a pixel
        grid_size : int
            The side length of the subgrid (i.e. there will be grid_size^2 pixels)
        Returns
        -------
        result : [value] or [(value, value)]
            A list of results
        """

        # TODO : if coordinate = 0.15", a 2x2 subgrid should be at 0.1" and 0.2" for pixel_scale = 0.3"
        # TODO : below - half = 0.3/2 = 0.15", step = 0.3/2 = 0.15" (for 2x2)
        # TODO : x = 0.15" - 0.15" + (0.15"/2) + 0*(0.15/2) = 0.075" (x = 0)
        # TODO : x = 0.15" (x = 1)
        # TODO : Updated function below using step = pixel_scale / (grid_size+1) and deleting the thierd term in the
        # TODO : loop equations

        # TODO : now, step = 0.3 / 3, 0.1, so x = 0.1 " and 0.2 ", as expected.

        # TODO : Does the 3x3 case work?
        # TODO : half = 0.15", step = 0.3 / 4 = 0.075, so x = 0.075" 0.15", 0.0225", as expeected :)

        half = pixel_scale / 2
        step = pixel_scale / (grid_size + 1)
        results = []
        for x in range(grid_size):
            for y in range(grid_size):
                x = coordinates[0] - half + (x + 1) * step
                y = coordinates[1] - half + (y + 1) * step
                results.append(func(self, (x, y)))
        return results

    return wrapper


def iterative_subgrid(subgrid_func):
    """
    Decorator to iteratively increase the grid size until the difference between results reaches a defined threshold
    Parameters
    ----------
    subgrid_func : function(coordinates, pixel_scale, grid_size) -> value
        A function decorated with subgrid and average
    Returns
    -------
        A function that will iteratively increase grid size until a desired accuracy is reached
    """
    @wraps(subgrid_func)
    def wrapper(self, coordinates, pixel_scale=0.1, threshold=0.0001):
        """

        Parameters
        ----------
        self : Profile
            The instance that owns the function being wrapped
        coordinates : (float, float)
            x, y coordinates in image space
        pixel_scale : float
            The size of a pixel
        threshold : float
            The minimum difference between the result at two different grid sizes
        Returns
        -------
            The last result calculated once the difference between two results becomes lower than the threshold
        """
        last_result = None
        grid_size = 1
        while True:
            next_result = subgrid_func(self, coordinates, pixel_scale=pixel_scale, grid_size=grid_size)
            if last_result is not None and abs(next_result - last_result) < threshold:
                return next_result
            last_result = next_result
            grid_size += 1

    return wrapper


def pixel_to_coordinate(dim_min, pixel_scale, pixel_coordinate):
    return dim_min + pixel_coordinate * pixel_scale


class LightProfile(object):
    """Mixin class that implements functions common to all light profiles"""

    def as_array(self, x_min=-5, y_min=-5, x_max=5, y_max=5, pixel_scale=0.1, mask=None):
        """

        Parameters
        ----------
        pixel_scale : float
            The arcsecond (") size of each pixel
        x_min : float
            The minimum x bound
        y_min : float
            The minimum y bound
        x_max : float
            The maximum x bound
        y_max : float
            The maximum y bound
        mask : Mask
            An object that has an is_masked method which returns True if (x, y) coordinates should be masked (i.e. not
            return a value)
        Returns
        -------
        array
            A numpy array illustrating this light profile between the given bounds
        """
        return array_for_function(self.flux_at_coordinates, x_min, y_min, x_max, y_max, pixel_scale, mask)

    # noinspection PyMethodMayBeStatic
    @avg
    @subgrid
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

    # TODO: find a good test for subgridding of a light profile
    @iterative_subgrid
    def flux_at_coordinates_iteratively_subgridded(self, coordinates):
        return self.flux_at_coordinates(coordinates)

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
        array = self.as_array(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, pixel_scale=pixel_scale)
        pyplot.imshow(array)
        pyplot.clim(vmax=np.mean(array) + np.std(array))
        pyplot.show()


class CombinedLightProfile(list, LightProfile):
    """A light profile comprising one or more light profiles"""

    def __init__(self, *light_profiles):
        super(CombinedLightProfile, self).__init__(light_profiles)

    @avg
    @subgrid
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


class SersicLightProfile(EllipticalProfile, LightProfile):
    """Used to fit the light of a galaxy. It can produce both highly concentrated light profiles (for high Sersic Index)
     or extended flatter profiles (for low Sersic Index)."""

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

    def as_sersic_profile(self):
        """

        Returns
        -------
        profile : SersicLightProfile
            A new sersic profile with parameters taken from this profile
        """
        return SersicLightProfile(axis_ratio=self.axis_ratio, phi=self.phi, flux=self.flux,
                                  effective_radius=self.effective_radius, sersic_index=self.sersic_index,
                                  centre=self.centre)

    def as_core_sersic_profile(self, radius_break, flux_break, gamma, alpha):
        """

        Returns
        -------
        profile : CoreSersicLightProfile
            A new core sersic profile with parameters taken from this profile
        """
        return CoreSersicLightProfile(axis_ratio=self.axis_ratio, phi=self.phi, flux=self.flux,
                                      effective_radius=self.effective_radius, sersic_index=self.sersic_index,
                                      radius_break=radius_break, flux_break=flux_break, gamma=gamma, alpha=alpha,
                                      centre=self.centre)

    def as_exponential_profile(self):
        """

        Returns
        -------
        profile : ExponentialLightProfile
            A new exponential profile with parameters taken from this profile
        """
        return ExponentialLightProfile(axis_ratio=self.axis_ratio, phi=self.phi, flux=self.flux,
                                       effective_radius=self.effective_radius, centre=self.centre)

    def as_dev_vaucouleurs_profile(self):
        """

        Returns
        -------
        profile : DevVaucouleursLightProfile
            A new dev vaucouleurs profile with parameters taken from this profile
        """
        return DevVaucouleursLightProfile(axis_ratio=self.axis_ratio, phi=self.phi, flux=self.flux,
                                          effective_radius=self.effective_radius, centre=self.centre)

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

    @avg
    @subgrid
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


class MassProfile(object):
    def deflection_angle_array(self, x_min=-5, y_min=-5, x_max=5, y_max=5, pixel_scale=0.1, mask=None):
        """

        Parameters
        ----------
        pixel_scale : float
            The arcsecond (") size of each pixel
        x_min : float
            The minimum x bound
        y_min : float
            The minimum y bound
        x_max : float
            The maximum x bound
        y_max : float
            The maximum y bound

        Returns
        -------
        array
            A numpy array illustrating this deflection angles for this profile between the given bounds
        """
        return array_for_function(self.compute_deflection_angle, x_min, y_min, x_max, y_max, pixel_scale, mask)

    # noinspection PyMethodMayBeStatic
    def compute_deflection_angle(self, coordinates):
        raise AssertionError("Compute deflection angles should be overridden")

    @subgrid
    def compute_deflection_angle_subgridded(self, coordinates):
        return self.compute_deflection_angle(coordinates)


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


class EllipticalPowerLawMassProfile(EllipticalProfile, MassProfile):
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

        from scipy.integrate import quad

        # TODO : Unit tests missing - need to sort out scipy.integrate

        coordinates = self.coordinates_rotate_to_elliptical(coordinates)

        defl = {}

        npow = 0.0
        defl[0] = quad(self.defl_func, a=0.0, b=1.0, args=(coordinates, npow))[0]
        defl[0] = self.einstein_radius_rescaled*defl[0]*coordinates[0] / 4.0

        npow = 1.0
        defl[1] = quad(self.defl_func, a=0.0, b=1.0, args=(coordinates, npow))[0]
        defl[1] = self.einstein_radius_rescaled*defl[1]*coordinates[1] / 4.0

        # TODO: implement a numerical integrator for this profile using scipy and / or c++

        # defl_elliptical = scipy.integrate(coordinates_elliptical, kappa_power_law)
        # defl_angles = self.coordinates_back_to_cartesian(coordinates_elliptical=defl_elliptical)
        # defl_angles = self.normalization*defl_angles
        # return defl_angles

        return defl


    def defl_func(self, u, coordinates, npow):
        eta = (u*((coordinates[0]**2) + (coordinates[1]**2/(1-(1-self.axis_ratio**2)*u))))**0.5
        return eta**(-(self.slope-1))/((1-(1-self.axis_ratio**2)*u)**(npow+0.5))

    @property
    def einstein_radius_rescaled(self):
        return ((3 - self.slope) / (1 + self.axis_ratio)) * self.einstein_radius


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
    def normalization(self):
        return self.einstein_radius_rescaled * self.axis_ratio / (math.sqrt(1 - self.axis_ratio ** 2))

    def compute_deflection_angle(self, coordinates):
        """
        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The deflection angles at these coordinates
        """

        # TODO : Need to rotate the deflection angles computed belo back to othe Cartesian image coordinates

        coordinates = self.coordinates_rotate_to_elliptical(coordinates)
        psi = math.sqrt((self.axis_ratio ** 2) * (coordinates[0] ** 2) + coordinates[1] ** 2)

        # TODO: This line sometimes throws a division by zero error. May need to check value of psi, try/except or even
        # TODO: throw an assertion error if the inputs causing the error are invalid?
        defl_x = self.normalization * math.atan((math.sqrt(1 - self.axis_ratio ** 2) * coordinates[0]) / psi)
        defl_y = self.normalization * math.atanh((math.sqrt(1 - self.axis_ratio ** 2) * coordinates[1]) / psi)
        return defl_x, defl_y

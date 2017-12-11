import math
import numpy as np
from functools import wraps


def avg(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Parameters
        ----------
        Returns
        -------
            The logical average of that collection
        """
        results = func(*args, **kwargs)
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
    def wrapper(coordinates, pixel_scale=0.1, grid_size=1):
        """

        Parameters
        ----------
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

        half = pixel_scale / 2
        step = pixel_scale / (grid_size + 1)
        results = []
        for x in range(grid_size):
            for y in range(grid_size):
                x1 = coordinates[0] - half + (x + 1) * step
                y1 = coordinates[1] - half + (y + 1) * step
                results.append(func((x1, y1)))
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
    def wrapper(coordinates, pixel_scale=0.1, threshold=0.0001):
        """

        Parameters
        ----------
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
            next_result = subgrid_func(coordinates, pixel_scale=pixel_scale, grid_size=grid_size)
            if last_result is not None and abs(next_result - last_result) / last_result < threshold:
                return next_result
            last_result = next_result
            grid_size += 1

    return wrapper


def array_function(func):
    """

    Parameters
    ----------
    func : function(coordinates)
            A function that takes coordinates and returns a value

    Returns
    -------
        A function that takes bounds, a pixel scale and mask and returns an array
    """

    @wraps(func)
    def wrapper(x_min=-5, y_min=-5, x_max=5, y_max=5, pixel_scale=0.1, mask=None):
        """

        Parameters
        ----------
        mask : Mask
            An object that has an is_masked method which returns True if (x, y) coordinates should be masked (i.e. not
            return a value)
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

    return wrapper


def side_length(dim_min, dim_max, pixel_scale):
    return int((dim_max - dim_min) / pixel_scale)


def pixel_to_coordinate(dim_min, pixel_scale, pixel_coordinate):
    return dim_min + pixel_coordinate * pixel_scale


class TransformedCoordinates(tuple):
    """Coordinates that have been transformed to the coordinate system of the profile"""

    def __init__(self, coordinates):
        super(TransformedCoordinates, self).__init__(coordinates)


def transform_coordinates(func):
    """
    Wrap the function in a function that checks whether the coordinates have been transformed. If they have not been
    transformed then they are transformed. If coordinates are returned they are returned in the coordinate system in
    which they were passed in.
    Parameters
    ----------
    func : function
        A function that requires transformed coordinates

    Returns
    -------
        A function that can except cartesian or transformed coordinates

    """

    @wraps(func)
    def wrapper(profile, coordinates, *args, **kwargs):
        """

        Parameters
        ----------
        profile : Profile
            The profile that owns the function
        coordinates : TransformedCoordinates or (float, float)
            Coordinates in either cartesian or profile coordinate system
        args
        kwargs

        Returns
        -------
            A value or coordinates in the same coordinate system as those passed ins
        """
        if not isinstance(coordinates, TransformedCoordinates):
            result = func(profile, profile.transform_to_reference_frame(coordinates), *args, **kwargs)
            if isinstance(result, TransformedCoordinates):
                result = profile.transform_from_reference_frame(result)
            return result
        return func(profile, coordinates, *args, **kwargs)

    return wrapper


class CoordinatesException(Exception):
    """Exception thrown when coordinates assertion fails"""

    def __init__(self, message):
        super(CoordinatesException, self).__init__(message)


class Profile(object):
    """Abstract Profile, describing an object with x, y cartesian coordinates"""

    def __init__(self, centre):
        self.centre = centre

    # noinspection PyMethodMayBeStatic
    def transform_to_reference_frame(self, coordinates):
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
        raise AssertionError("Transform to reference frame should be overridden")

    # noinspection PyMethodMayBeStatic
    def transform_from_reference_frame(self, coordinates):
        """

        Parameters
        ----------
        coordinates: TransformedCoordinates
            Coordinates that have been transformed to the reference frame of the profile
        Returns
        -------
        coordinates: (float, float)
            Coordinates that are back in the original reference frame
        """
        raise AssertionError("Transform from reference frame should be overridden")

    @property
    def x_cen(self):
        return self.centre[0]

    @property
    def y_cen(self):
        return self.centre[1]

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


class EllipticalProfile(Profile):
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
        super(EllipticalProfile, self).__init__(centre)

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
        Determine the sin and cosine of the angle between the profile ellipse and positive x-axis, \
        defined counter-clockwise from x.

        Returns
        -------
        The sin and cosine of the angle
        """
        phi_radians = math.radians(self.phi)
        return math.cos(phi_radians), math.sin(phi_radians)

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

        shifted_coordinates = self.transform_to_reference_frame(coordinates)
        return math.sqrt(self.axis_ratio) * math.sqrt(
            shifted_coordinates[0] ** 2 + (shifted_coordinates[1] / self.axis_ratio) ** 2)

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

    def coordinates_angle_from_x(self, coordinates):
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
        shifted_coordinates = self.coordinates_to_centre(coordinates)
        theta_from_x = math.degrees(np.arctan2(shifted_coordinates[1], shifted_coordinates[0]))
        return theta_from_x

    def transform_from_reference_frame(self, coordinates_elliptical):
        """
        Rotate elliptical coordinates back to the original Cartesian grid (for a circular profile this
        returns the input coordinates)

        Parameters
        ----------
        coordinates_elliptical : TransformedCoordinates(float, float)
            The x and y coordinates of the image translated to the elliptical coordinate system

        Returns
        ----------
        The coordinates (typically deflection angles) on a regular Cartesian grid
        """

        if not isinstance(coordinates_elliptical, TransformedCoordinates):
            raise CoordinatesException("Can't return cartesian coordinates to cartesian coordinates. Did you remember"
                                       " to explicitly make the elliptical coordinates TransformedCoordinates?")

        x_elliptical = coordinates_elliptical[0]
        x = (x_elliptical * self.cos_phi - coordinates_elliptical[1] * self.sin_phi)
        y = (+x_elliptical * self.sin_phi + coordinates_elliptical[1] * self.cos_phi)
        return x, y

    def transform_to_reference_frame(self, coordinates):
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

        if isinstance(coordinates, TransformedCoordinates):
            raise CoordinatesException("Trying to transform already transformed coordinates")

        # Compute distance of coordinates to the lens profile centre
        radius = self.coordinates_to_radius(coordinates)

        # Compute the angle between the coordinates and x-axis
        theta_from_x = self.coordinates_angle_from_x(coordinates)

        # Compute the angle between the coordinates and profile ellipse
        cos_theta, sin_theta = self.coordinates_angle_to_profile(theta_from_x)

        # Multiply by radius to get their x / y distance from the profile centre in this elliptical unit system
        return TransformedCoordinates((radius * cos_theta, radius * sin_theta))


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

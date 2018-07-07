import numpy as np
from functools import wraps
import inspect
from src import exc


def nan_tuple(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ZeroDivisionError:
            return np.nan, np.nan

    return wrapper


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
        coordinates
            A coordinate pair
        pixel_scale : float
            The scale of a pixel
        grid_size : int
            The side length of the subgrid (i.e. there will be grid_size^2 data_to_image)
        Returns
        -------
        result : [value] or [(value, value)]
            A list of nlo
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
    Decorator to iteratively increase the grid_coords size until the difference between nlo reaches a defined threshold
    Parameters
    ----------
    subgrid_func : function(coordinates, pixel_scale, grid_size) -> value
        A function decorated with subgrid and average
    Returns
    -------
        A function that will iteratively increase grid_coords size until a desired accuracy is reached
    """

    @wraps(subgrid_func)
    def wrapper(coordinates, pixel_scale=0.1, threshold=0.0001):
        """

        Parameters
        ----------
        coordinates
            x, y coordinates in coordinates space
        pixel_scale : float
            The size of a pixel
        threshold : float
            The minimum difference between the result at two different grid_coords sizes
        Returns
        -------
            The last result calculated once the difference between two nlo becomes lower than the threshold
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


def side_length(dim_min, dim_max, pixel_scale):
    return int((dim_max - dim_min) / pixel_scale)


def pixel_to_coordinate(dim_min, pixel_scale, pixel_coordinate):
    return dim_min + pixel_coordinate * pixel_scale


class TransformedCoordinates(tuple):

    # noinspection PyUnusedLocal
    def __init__(self, coordinates):
        """This class tracks whether a set of coordinates have been transformed to the center and rotation angle of a \
        geometry profile.

        If they haven't, the code will transform them using the profile's geometry (centre, phi) if it is necessary \
        for the calculation (e.g. for computing the intensity or deflection angles of a profile.

        If the coordinates have already been transformed (and are therefore an instance of this class), the code
        knows not to perform the transformation on them again."""
        super(TransformedCoordinates, self).__init__()


def transform_coordinates(func):
    """
    Wrap the function in a function that checks whether the coordinates have been transformed. If they have not been \
    transformed then they are transformed. If coordinates are returned they are returned in the coordinate system in \
    which they were passed in.

    Parameters
    ----------
    func : (profiles, *args, **kwargs) -> Object
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
            The profiles that owns the function
        coordinates : TransformedCoordinates or ndarray
            PlaneCoordinates in either cartesian or profiles coordinate system
        args
        kwargs

        Returns
        -------
            A value or coordinate in the same coordinate system as those passed in.
        """
        if not isinstance(coordinates, TransformedCoordinates):
            result = func(profile, profile.transform_to_reference_frame(coordinates), *args, **kwargs)
            if isinstance(result, TransformedCoordinates):
                result = profile.transform_from_reference_frame(result)
            return np.asarray(result)
        return func(profile, coordinates, *args, **kwargs)

    return wrapper


def transform_grid(func):
    """
    Wrap the function in a function that checks whether the coordinates have been transformed. If they have not been \
    transformed then they are transformed. If coordinates are returned they are returned in the coordinate system in \
    which they were passed in.

    Parameters
    ----------
    func : (profiles, *args, **kwargs) -> Object
        A function that requires transformed coordinates

    Returns
    -------
        A function that can except cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(profile, grid, *args, **kwargs):
        """

        Parameters
        ----------
        profile : Profile
            The profiles that owns the function
        grid : ndarray
            PlaneCoordinates in either cartesian or profiles coordinate system
        args
        kwargs

        Returns
        -------
            A value or coordinate in the same coordinate system as those passed in.
        """
        if not isinstance(grid, TransformedGrid):
            result = func(profile, profile.transform_grid_to_reference_frame(grid), *args, **kwargs)
            if isinstance(result, TransformedGrid):
                result = profile.transform_grid_from_reference_frame(result)
            return np.asarray(result)
        return func(profile, grid, *args, **kwargs)

    return wrapper


class TransformedGrid(np.ndarray):
    pass


class Profile(object):

    def __init__(self, centre=(0.0, 0.0)):
        """Abstract Profile, describing an object with x, y cartesian coordinates"""
        self.centre = centre

    @property
    def parameter_labels(self):
        return ['x', 'y']

    # noinspection PyMethodMayBeStatic
    def transform_to_reference_frame(self, coordinates):
        """ Transform Cartesian coordinates to the profiles's reference frame.

        Parameters
        ----------
        coordinates
            The x and y coordinates of the coordinates

        Returns
        ----------
        The coordinates after the elliptical translation
        """
        raise NotImplemented("Transform to reference frame should be overridden")

    def transform_grid_to_reference_frame(self, grid):
        raise NotImplemented()

    def transform_grid_from_reference_frame(self, grid):
        raise NotImplemented()

    @classmethod
    def from_profile(cls, profile, **kwargs):
        """ Creates any profiles from any other profiles, keeping all attributes from the original profiles that can
        then be passed into the constructor of the new profiles. Any none optional attributes required by the new \
        profiles's constructor which are not available as attributes of the original profiles must be passed in as \
        key word arguments. Arguments matching attributes in the original profiles may be passed in to override \
        those attributes.

        Examples
        ----------
        p = profiles.Profile(centre=(1, 1))
        elliptical_profile = profiles.EllipticalProfile.from_profile(p, axis_ratio=1, phi=2)

        elliptical_profile = profiles.EllipticalProfile(1, 2)
        profiles.Profile.from_profile(elliptical_profile).__class__ == profiles.Profile

        Parameters
        ----------
        profile: Profile
            A child of the profiles class
        kwargs
            Key word constructor arguments for the new profiles
        """
        arguments = vars(profile)
        arguments.update(kwargs)
        init_args = inspect.getfullargspec(cls.__init__).args
        arguments = {argument[0]: argument[1] for argument in arguments.items() if argument[0] in init_args}
        return cls(**arguments)

    def transform_from_reference_frame(self, coordinates):
        """ Transform coordinates from a profile's reference frame to the original Cartesian grid_coords.

        Parameters
        ----------
        coordinates: TransformedCoordinates
            PlaneCoordinates that have been transformed to the reference frame of the profiles

        Returns
        -------
        coordinates: (float, float)
            Coordinates that are in the original reference frame.
        """
        raise NotImplementedError("Transform from reference frame should be overridden")

    @property
    def x_cen(self):
        return self.centre[0]

    @property
    def y_cen(self):
        return self.centre[1]

    def coordinates_to_centre(self, coordinates):
        """ Converts coordinates to the profiles's centre.

        This is performed via a translation, which subtracts the profile centre from the coordinates.

        Parameters
        ----------
        coordinates
            The (x, y) coordinates of the profile.

        Returns
        ----------
        The coordinates at the profile's centre.
        """
        return np.subtract(coordinates, self.centre)

    def coordinates_from_centre(self, coordinates):
        """ Translate shifted coordinates back to the profile's original centre.

        This is performed via a translation, which adds the coordinates and centre.

        Parameters
        ----------
        coordinates
            The (x, y) coordinates of the profile.

        Returns
        ----------
        The coordinates at the their original centre.
        """
        return np.add(coordinates, self.centre)

    def coordinates_to_radius(self, coordinates):
        """ Convert coordinates to the radial distance from the profile centre.

        The coordinates are first shifted to the profile centre and the radius is computed via Pythagoras.

        Parameters
        ----------
        coordinates 
            The (x, y) coordinates of the profile.

        Returns
        -------
        The radial distance of the coordinates from the profile centre.
        """
        shifted_coordinates = self.coordinates_to_centre(coordinates)
        return np.sqrt(np.sum(shifted_coordinates ** 2.0))

    def __repr__(self):
        return '{}\n{}'.format(self.__class__.__name__,
                               '\n'.join(["{}: {}".format(k, v) for k, v in self.__dict__.items()]))


class EllipticalProfile(Profile):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0):
        """ Generic elliptical profiles class to contain functions shared by light and mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        """
        super(EllipticalProfile, self).__init__(centre)

        self.axis_ratio = axis_ratio
        self.phi = phi

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi']

    @property
    def cos_phi(self):
        return self.cos_and_sin_from_x_axis()[0]

    @property
    def sin_phi(self):
        return self.cos_and_sin_from_x_axis()[1]

    def cos_and_sin_from_x_axis(self):
        """ Determine the sin and cosine of the angle between the profile's ellipse and the positive x-axis, \
        counter-clockwise.

        Returns
        -------
        The sin and cosine of the angle, counter-clockwise from the positive x-axis.
        """
        phi_radians = np.radians(self.phi)
        return np.cos(phi_radians), np.sin(phi_radians)

    def cos_and_sin_of_angle_to_profile(self, theta):
        """ Compute the sin and cosine of an angle (e.g. of the shifted coordinates to the positive x-axis) and an \
        elliptical profile.

        Parameters
        ----------
        theta : float
            The angle of the shifted coordinates, defined from the positive x-axis.

        Returns
        ----------
        The sin and cos of the angle between the shifted coordinates and profile's ellipse.
        """
        theta_coordinate_to_profile = np.radians(theta - self.phi)
        return np.cos(theta_coordinate_to_profile), np.sin(theta_coordinate_to_profile)

    def grid_angle_to_profile(self, theta_grid):
        theta_coordinate_to_profile = np.radians(np.add(theta_grid, - self.phi))
        return np.cos(theta_coordinate_to_profile), np.sin(theta_coordinate_to_profile)

    def coordinates_angle_from_x(self, coordinates):
        """ Compute the angle between the coordinates (shifted to the profile centre) and the positive x-axis, \
        defined counter-clockwise.

        Elliptical profiles are symmetric about 180 degrees, therefore angles above 180 are converted to their \
        equivalent value from 0. (e.g. 225 degrees counter-clockwise from the x-axis is converted to 45 degrees)

        Parameters
        ----------
        coordinates
            The (x, y) coordinates of the profile.

        Returns
        ----------
        The angle between the shifted coordinates and the positive x-axis (defined counter-clockwise).
        """
        shifted_coordinates = self.coordinates_to_centre(coordinates)
        return np.degrees(np.arctan2(shifted_coordinates[1], shifted_coordinates[0]))

    def rotate_coordinates_from_profile(self, coordinates_elliptical):
        """ Rotate elliptical coordinates from the reference frame of the profile back to the coordinates original \
         Cartesian grid_coords (coordinates are not shifted back to their original centre).

        Parameters
        ----------
        coordinates_elliptical : TransformedCoordinates(ndarray)
            The (x, y) coordinates in the reference frame of an elliptical profile.

        Returns
        ----------
        The coordinates rotated back to their original Cartesian grid_coords.
         """
        x_elliptical = coordinates_elliptical[0]
        x = (x_elliptical * self.cos_phi - coordinates_elliptical[1] * self.sin_phi)
        y = (+x_elliptical * self.sin_phi + coordinates_elliptical[1] * self.cos_phi)
        return x, y

    def rotate_grid_from_profile(self, grid_elliptical):
        """ Rotate elliptical coordinates from the reference frame of the profile back to the coordinates original \
         Cartesian grid_coords (coordinates are not shifted back to their original centre).

        Parameters
        ----------
        grid_elliptical : TransformedGrid(ndarray)
            The (x, y) coordinates in the reference frame of an elliptical profile.

        Returns
        ----------
        The coordinates rotated back to their original Cartesian grid_coords.
         """
        x = np.add(np.multiply(grid_elliptical[:, 0], self.cos_phi), - np.multiply(grid_elliptical[:, 1], self.sin_phi))
        y = np.add(np.multiply(grid_elliptical[:, 0], self.sin_phi), np.multiply(grid_elliptical[:, 1], self.cos_phi))
        return np.vstack((x, y)).T

    @transform_coordinates
    def coordinates_to_elliptical_radius(self, coordinates):
        """
        Convert coordinates to an elliptical radius.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        coordinates
            The (x, y) coordinates of the profile.

        Returns
        -------
        The radius at those coordinates
        """
        return np.sqrt(coordinates[0] ** 2 + (coordinates[1] / self.axis_ratio) ** 2)

    @transform_grid
    def grid_to_elliptical_radius(self, grid):
        """
        Convert coordinates to an elliptical radius.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid

        Returns
        -------
        The radius at those coordinates
        """
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(np.divide(grid[:, 1], self.axis_ratio))))

    @transform_coordinates
    def coordinates_to_eccentric_radius(self, coordinates):
        """ Convert the coordinates to an eccentric radius.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        The eccentric radius is defined as (1.0/axis_ratio) * elliptical radius. It is used in light profiles such \
        that the effective radius is defined as the circular effective radius.

        Parameters
        ----------
        coordinates
            The (x, y) coordinates of the profile.

        Returns
        -------
        The radius at those coordinates
        """

        return np.sqrt(self.axis_ratio) * np.sqrt(coordinates[0] ** 2 + (coordinates[1] / self.axis_ratio) ** 2)

    def coordinates_radius_to_x_and_y(self, coordinates, radius):
        """Decomposed a coordinate at a given radial distance r into its x and y vectors

        Parameters
        ----------
        coordinates
            The (x, y) coordinates of the profile.
        radius : float
            The radial distance r from the centre of the coordinate reference frame.

        Returns
        ----------
        The coordinates after decomposition in their x and y components
        """
        theta_from_x = np.degrees(np.arctan2(coordinates[1], coordinates[0]))
        cos_theta, sin_theta = self.cos_and_sin_of_angle_to_profile(theta_from_x)
        return radius * cos_theta, radius * sin_theta

    def grid_radius_to_cartesian(self, grid, radius):
        theta_grid = np.degrees(np.arctan2(grid[:, 1], grid[:, 0]))
        cos_theta, sin_theta = self.grid_angle_to_profile(theta_grid)
        return np.multiply(radius[:, None], np.vstack((cos_theta, sin_theta)).T)

    def transform_from_reference_frame(self, coordinates_elliptical):
        """ Transform elliptical coordinates from their profile's reference frame back to the original Cartesian \
        grid_coords.

        This routine checks the coordinates are an instance of the TransformedCoordinates class, thus ensuring they \
        have already been translated to the profile's reference frame.

        Parameters
        ----------
        coordinates_elliptical : TransformedCoordinates(ndarray)
            The (x, y) coordinates which have already been transformed to that of the profile.

        Returns
        ----------
        The coordinates (typically deflection angles) on a regular Cartesian grid_coords
        """

        if not isinstance(coordinates_elliptical, TransformedCoordinates):
            raise exc.CoordinatesException(
                "Can't return cartesian coordinates to cartesian coordinates. Did you remember"
                " to explicitly make the elliptical coordinates TransformedCoordinates?")

        x, y = self.rotate_coordinates_from_profile(coordinates_elliptical)
        return self.coordinates_from_centre((x, y))

    def transform_grid_to_reference_frame(self, grid):
        shifted_coordinates = np.subtract(grid, self.centre)
        radius = np.sqrt(np.sum(shifted_coordinates ** 2.0, 1))
        theta_coordinate_to_profile = np.radians(
            np.degrees(np.arctan2(shifted_coordinates[:, 1], shifted_coordinates[:, 0])) - self.phi)
        transformed = np.vstack(
            (radius * np.cos(theta_coordinate_to_profile), radius * np.sin(theta_coordinate_to_profile))).T
        return transformed.view(TransformedGrid)

    def transform_grid_from_reference_frame(self, grid):
        x = np.add(np.add(np.multiply(grid[:, 0], self.cos_phi), - np.multiply(grid[:, 1], self.sin_phi)),
                   self.centre[0])
        y = np.add(
            np.add(np.multiply(grid[:, 0], self.sin_phi), np.multiply(grid[:, 1], self.cos_phi) - self.centre[1]),
            self.centre[1])
        return np.vstack((x, y)).T

    def transform_to_reference_frame(self, coordinates):
        """ Translate Cartesian coordinates to the profiles's reference frame.

        The translated coordinates become an instance of the TransformedCoordinates class.

        Parameters
        ----------
        coordinates
            The (x, y) coordinates on the Cartesian grid_coords..

        Returns
        ----------
        The coordinates after the elliptical profile transformation.
        """

        if isinstance(coordinates, TransformedCoordinates):
            raise exc.CoordinatesException("Trying to transform already transformed coordinates")

        # Compute distance of coordinates to the lens profiles centre
        radius = self.coordinates_to_radius(coordinates)

        # Compute the angle between the coordinates and x-axis
        theta_from_x = self.coordinates_angle_from_x(coordinates)

        # Compute the angle between the coordinates and profiles ellipse
        cos_theta, sin_theta = self.cos_and_sin_of_angle_to_profile(theta_from_x)

        # Multiply by radius to get their x / y distance from the profiles centre in this elliptical unit system
        return TransformedCoordinates((radius * cos_theta, radius * sin_theta))

    def eta_u(self, u, coordinates):
        return np.sqrt((u * ((coordinates[0] ** 2) + (coordinates[1] ** 2 / (1 - (1 - self.axis_ratio ** 2) * u)))))


class SphericalProfile(EllipticalProfile):

    def __init__(self, centre=(0.0, 0.0)):
        """ Generic circular profiles class to contain functions shared by light and mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profile.
        """
        super(SphericalProfile, self).__init__(centre, 1.0, 0.0)

    @property
    def parameter_labels(self):
        return ['x', 'y']

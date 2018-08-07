import numpy as np
from functools import wraps
import inspect


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
            result = func(profile, profile.transform_grid_to_reference_frame_jitted(grid), *args, **kwargs)
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

    def transform_grid_to_reference_frame_jitted(self, grid):
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

    @property
    def x_cen(self):
        return self.centre[0]

    @property
    def y_cen(self):
        return self.centre[1]

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
    def phi_radians(self):
        return np.radians(self.phi)

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

    def grid_angle_to_profile(self, theta_grid):
        theta_coordinate_to_profile = np.add(theta_grid, - self.phi_radians)
        return np.cos(theta_coordinate_to_profile), np.sin(theta_coordinate_to_profile)

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

    @transform_grid
    def grid_to_radius(self, grid):
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
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))

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

    @transform_grid
    def grid_to_eccentric_radii(self, grid):
        return np.multiply(np.sqrt(self.axis_ratio),
                           np.sqrt(np.add(np.square(grid[:, 0]),
                                          np.square(np.divide(grid[:, 1], self.axis_ratio))))).view(np.ndarray)

    def grid_radius_to_cartesian(self, grid, radius):
        theta_grid = np.arctan2(grid[:, 1], grid[:, 0])
        cos_theta, sin_theta = self.grid_angle_to_profile(theta_grid)
        return np.multiply(radius[:, None], np.vstack((cos_theta, sin_theta)).T)

    def transform_grid_to_reference_frame_jitted(self, grid):
        shifted_coordinates = np.subtract(grid, self.centre)
        radius = np.sqrt(np.sum(shifted_coordinates ** 2.0, 1))
        theta_coordinate_to_profile = np.arctan2(shifted_coordinates[:, 1],
                                                 shifted_coordinates[:, 0]) - self.phi_radians
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

    def transform_grid_to_reference_frame_jitted(self, grid):
        transformed = np.subtract(grid, self.centre)
        return transformed.view(TransformedGrid)

    def grid_angle_to_profile(self, theta_grid):
        return np.cos(theta_grid), np.sin(theta_grid)

    def grid_radius_to_cartesian(self, grid, radius):
        theta_grid = np.arctan2(grid[:, 1], grid[:, 0])
        cos_theta, sin_theta = self.grid_angle_to_profile(theta_grid)
        return np.multiply(radius[:, None], np.vstack((cos_theta, sin_theta)).T)

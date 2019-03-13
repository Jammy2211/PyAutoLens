import numpy as np
from functools import wraps


def transform_grid(func):
    """Wrap the function in a function that checks whether the coordinates have been transformed. If they have not \ 
    been transformed then they are transformed.

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
        profile : GeometryProfile
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
            return func(profile, profile.transform_grid_to_reference_frame(grid), *args, **kwargs)
        else:
            return func(profile, grid, *args, **kwargs)

    return wrapper


def cache(func):
    """
    Caches results of a call to a grid function. If a grid that evaluates to the same byte value is passed into the same
    function of the same instance as previously then the cached result is returned.

    Parameters
    ----------
    func
        Some instance method that takes a grid as its argument

    Returns
    -------
    result
        Some result, either newly calculated or recovered from the cache
    """

    def wrapper(instance: GeometryProfile, grid: np.ndarray):
        if not hasattr(instance, "cache"):
            instance.cache = {}
        key = (func.__name__, grid.tobytes())
        if key not in instance.cache:
            instance.cache[key] = func(instance, grid)
        return instance.cache[key]

    return wrapper


class TransformedGrid(np.ndarray):
    pass


class GeometryProfile(object):

    def __init__(self, centre=(0.0, 0.0)):
        """An abstract geometry profile, which describes profiles with y and x centre Cartesian coordinates
        
        Parameters
        -----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        """
        self.centre = centre

    def transform_grid_to_reference_frame(self, grid):
        raise NotImplemented()

    def transform_grid_from_reference_frame(self, grid):
        raise NotImplemented()

    def __repr__(self):
        return '{}\n{}'.format(self.__class__.__name__,
                               '\n'.join(["{}: {}".format(k, v) for k, v in self.__dict__.items()]))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class SphericalProfile(GeometryProfile):

    def __init__(self, centre=(0.0, 0.0)):
        """ A spherical profile, which describes profiles with y and x centre Cartesian coordinates.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        """
        super(SphericalProfile, self).__init__(centre)

    @transform_grid
    def grid_to_grid_radii(self, grid):
        """Convert a grid of (y, x) coordinates to a grid of their circular radii.

        If the coordinates have not been transformed to the profile's centre, this is performed automatically.

        Parameters
        ----------
        grid : TransformedGrid(ndarray)
            The (y, x) coordinates in the reference frame of the profile.
        """
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))

    def grid_angle_to_profile(self, grid_thetas):
        """The angle between each (y,x) coordinate on the grid and the profile, in radians.
        
        Parameters
        -----------
        grid_thetas : ndarray
            The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
        """
        return np.cos(grid_thetas), np.sin(grid_thetas)

    def grid_to_grid_cartesian(self, grid, radius):
        """
        Convert a grid of (y,x) coordinates with their specified circular radii to their original (y,x) Cartesian 
        coordinates.

        Parameters
        ----------
        grid : TransformedGrid(ndarray)
            The (y, x) coordinates in the reference frame of the profile.
        radius : ndarray
            The circular radius of each coordinate from the profile center.
        """
        grid_thetas = np.arctan2(grid[:, 0], grid[:, 1])
        cos_theta, sin_theta = self.grid_angle_to_profile(grid_thetas=grid_thetas)
        return np.multiply(radius[:, None], np.vstack((sin_theta, cos_theta)).T)

    def transform_grid_to_reference_frame(self, grid):
        """Transform a grid of (y,x) coordinates to the reference frame of the profile, including a translation to \
        its centre.

        Parameters
        ----------
        grid : ndarray
            The (y, x) coordinates in the original reference frame of the grid.
        """
        transformed = np.subtract(grid, self.centre)
        return transformed.view(TransformedGrid)

    def transform_grid_from_reference_frame(self, grid):
        """Transform a grid of (y,x) coordinates from the reference frame of the profile to the original observer \
        reference frame, including a translation from the profile's centre.

        Parameters
        ----------
        grid : TransformedGrid(ndarray)
            The (y, x) coordinates in the reference frame of the profile.
        """
        transformed = np.add(grid, self.centre)
        return transformed.view(TransformedGrid)


class EllipticalProfile(SphericalProfile):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0):
        """ An elliptical profile, which describes profiles with y and x centre Cartesian coordinates, an axis-ratio \
        and rotational angle phi.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotation angle of profiles ellipse counter-clockwise from positive x-axis
        """
        super(EllipticalProfile, self).__init__(centre)
        self.axis_ratio = axis_ratio
        self.phi = phi

    @property
    def phi_radians(self):
        return np.radians(self.phi)

    @property
    def cos_phi(self):
        return self.cos_and_sin_from_x_axis()[0]

    @property
    def sin_phi(self):
        return self.cos_and_sin_from_x_axis()[1]

    def cos_and_sin_from_x_axis(self):
        """ Determine the sin and cosine of the angle between the profile's ellipse and the positive x-axis, \
        counter-clockwise. """
        phi_radians = np.radians(self.phi)
        return np.cos(phi_radians), np.sin(phi_radians)

    def grid_angle_to_profile(self, grid_thetas):
        """The angle between each angle theta on the grid and the profile, in radians.

        Parameters
        -----------
        grid_thetas : ndarray
            The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
        """
        theta_coordinate_to_profile = np.add(grid_thetas, - self.phi_radians)
        return np.cos(theta_coordinate_to_profile), np.sin(theta_coordinate_to_profile)

    def rotate_grid_from_profile(self, grid_elliptical):
        """ Rotate a grid of elliptical (y,x) coordinates from the reference frame of the profile back to the \
        unrotated coordinate grid reference frame (coordinates are not shifted back to their original centre).

        This routine is used after computing deflection angles in the reference frame of the profile, so that the \
        deflection angles can be re-rotated to the frame of the original coordinates before performing ray-tracing.

        Parameters
        ----------
        grid_elliptical : TransformedGrid(ndarray)
            The (y, x) coordinates in the reference frame of an elliptical profile.
        """
        y = np.add(np.multiply(grid_elliptical[:, 1], self.sin_phi), np.multiply(grid_elliptical[:, 0], self.cos_phi))
        x = np.add(np.multiply(grid_elliptical[:, 1], self.cos_phi), - np.multiply(grid_elliptical[:, 0], self.sin_phi))
        return np.vstack((y, x)).T

    @transform_grid
    def grid_to_elliptical_radii(self, grid):
        """ Convert a grid of (y,x) coordinates to an elliptical radius.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid : TransformedGrid(ndarray)
            The (y, x) coordinates in the reference frame of the elliptical profile.
        """
        return np.sqrt(np.add(np.square(grid[:, 1]), np.square(np.divide(grid[:, 0], self.axis_ratio))))

    @transform_grid
    def grid_to_eccentric_radii(self, grid):
        """Convert a grid of (y,x) coordinates to an eccentric radius, which is (1.0/axis_ratio) * elliptical radius \
        and used to define light profile half-light radii using circular radii.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid : TransformedGrid(ndarray)
            The (y, x) coordinates in the reference frame of the elliptical profile.
        """
        return np.multiply(np.sqrt(self.axis_ratio), self.grid_to_elliptical_radii(grid)).view(np.ndarray)

    def transform_grid_to_reference_frame(self, grid):
        """Transform a grid of (y,x) coordinates to the reference frame of the profile, including a translation to \
        its centre and a rotation to it orientation.

        Parameters
        ----------
        grid : ndarray
            The (y, x) coordinates in the original reference frame of the grid.
        """
        if self.__class__.__name__.startswith("Spherical"):
            return super().transform_grid_to_reference_frame(grid)
        shifted_coordinates = np.subtract(grid, self.centre)
        radius = np.sqrt(np.sum(shifted_coordinates ** 2.0, 1))
        theta_coordinate_to_profile = np.arctan2(shifted_coordinates[:, 0],
                                                 shifted_coordinates[:, 1]) - self.phi_radians
        transformed = np.vstack(
            (radius * np.sin(theta_coordinate_to_profile), radius * np.cos(theta_coordinate_to_profile))).T
        return transformed.view(TransformedGrid)

    def transform_grid_from_reference_frame(self, grid):
        """Transform a grid of (y,x) coordinates from the reference frame of the profile to the original observer \
        reference frame, including a rotation to its original orientation and a translation from the profile's centre.

        Parameters
        ----------
        grid : TransformedGrid(ndarray)
            The (y, x) coordinates in the reference frame of the profile.
        """
        if self.__class__.__name__.startswith("Spherical"):
            return super().transform_grid_from_reference_frame(grid)

        y = np.add(np.add(np.multiply(grid[:, 1], self.sin_phi), np.multiply(grid[:, 0], self.cos_phi)), self.centre[0])
        x = np.add(np.add(np.multiply(grid[:, 1], self.cos_phi), - np.multiply(grid[:, 0], self.sin_phi)),
                   self.centre[1])
        return np.vstack((y, x)).T

    def eta_u(self, u, coordinates):
        return np.sqrt((u * ((coordinates[1] ** 2) + (coordinates[0] ** 2 / (1 - (1 - self.axis_ratio ** 2) * u)))))

import itertools
import math
import numpy as np
from profile import profile


# TODO: This class seems to share some ideas with the generic profile. We should be careful not to over-integrate but
# TODO: for now I think it makes sense to leverage the profile module in other areas as we may be able to reuse some of
# TODO: its functionality.
class SourcePlaneGeometry(profile.Profile):
    """Stores the source-plane geometry, to ensure different components of the source-plane share the
    same geometry"""

    def __init__(self, centre=(0, 0)):
        super(SourcePlaneGeometry, self).__init__(centre)

    def coordinates_angle_from_x(self, coordinates):
        """
        Compute the angle between the coordinates and source-plane positive x-axis, defined counter-clockwise.

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the source-plane.

        Returns
        ----------
        The angle between the coordinates and the x-axis.
        """
        shifted_coordinates = self.coordinates_to_centre(coordinates)
        theta_from_x = math.degrees(np.arctan2(shifted_coordinates[1], shifted_coordinates[0]))
        if theta_from_x < 0.0:
            theta_from_x += 360.
        return theta_from_x


class SourcePlane(SourcePlaneGeometry):
    """Represents the source-plane and its corresponding traced image sub-coordinates"""

    def __init__(self, sub_coordinates, centre=(0, 0)):
        """

        Parameters
        ----------
        sub_coordinates : [(float, float)]
            The x and y coordinates of each traced image sub-pixel
        centre : (float, float)
            The centre of the source-plane.
        """

        super(SourcePlane, self).__init__(centre)

        self.sub_coordinates = sub_coordinates

    # TODO: You shouldn't have to set up attributes of a class crucial to its function after calling the constructor.
    # TODO: it seems like a mask here doesn't need to be an internal property but does depend on the attributes of the
    # TODO: class, with free attributes of border_mask and polynomial degree. There might be a better route but for now
    # TODO: I've written this method to create a border mask from an existing instance of source plane
    def border_with_mask_and_polynomial_degree(self, border_mask, polynomial_degree):
        return SourcePlaneBorder(list(itertools.compress(self.sub_coordinates, border_mask)), polynomial_degree,
                                 centre=self.centre)

    def relocate_coordinates_outside_border_with_mask_and_polynomial_degree(self, mask, polynomial_degree):
        self.relocate_coordinates_outside_border(self.border_with_mask_and_polynomial_degree(mask, polynomial_degree))

    def relocate_coordinates_outside_border(self, border):
        """ Move all source-plane coordinates outside of its source-plane border to the edge of its border"""
        self.sub_coordinates = list(map(lambda r: border.relocated_coordinate(r), self.sub_coordinates))


class SourcePlaneSparse(SourcePlane):
    """Represents the source-plane, including a sparse coordinate list."""

    # TODO : This is all untested - just experimenting with how best to handle the sparse grid and give you a sense
    # TODO : of the problem. See the Coordinates file for a description.

    def __init__(self, sub_coordinates, sparse_coordinates, sub_to_sparse, centre=(0, 0)):
        """

        Parameters
        ----------
        sub_coordinates : [(float, float)]
            The x and y coordinates of each traced image sub-pixel
        sparse_coordinates : list(float, float)
            The x and y coordinates of each traced image sparse-pixel
        sub_to_sparse : [int]
            The integer entry in sparse_clusters (and sparse_coordinates) that each sub-pixel corresponds to.
        centre : (float, float)
            The centre of the source-plane.
        """

        super(SourcePlaneSparse, self).__init__(sub_coordinates, centre)

        self.sparse_coordinates = sparse_coordinates
        self.sub_to_sparse = sub_to_sparse


class SourcePlaneBorder(SourcePlaneGeometry):
    """Represents the source-plane coordinates on the source-plane border. Each coordinate is stored alongside its
    distance from the source-plane centre (radius) and angle from the x-axis (theta)"""

    # TODO: You shouldn't have to set up attributes of a class crucial to its function after calling the constructor.
    # TODO: Here I've passed polynomial degree into the constructor
    def __init__(self, coordinates, polynomial_degree, centre=(0.0, 0.0)):
        """

        Parameters
        ----------
        coordinates : [(float, float)]
            The x and y coordinates of the source-plane border.
        centre : (float, float)
            The centre of the source-plane.
        """

        super(SourcePlaneBorder, self).__init__(centre)

        self.coordinates = coordinates
        self.thetas = list(map(lambda r: self.coordinates_angle_from_x(r), coordinates))
        self.radii = list(map(lambda r: self.coordinates_to_radius(r), coordinates))
        self.polynomial = np.polyfit(self.thetas, self.radii, polynomial_degree)

    # TODO: "get_" and "set_" are paradigms used for getting and setting a property of a class instance. They're common
    # TODO: in Java but not really used in Python. I prefer not to use generic verbs like that in the method name
    # TODO: because they often don't describe much about what the method does. (As an aside, the property decorated can
    # TODO: be used to implement a getter and setter paradigm which is really useful for computed variables where you
    # TODO: have to calculate a value from some other variable, such as an area from a radius)
    def border_radius_at_theta(self, theta):
        """For a an angle theta from the x-axis, return the border radius via the polynomial fit"""
        return np.polyval(self.polynomial, theta)

    def move_factor(self, coordinate):
        """Get the move factor of a coordinate.
         A move-factor defines how far a coordinate outside the source-plane border must be moved in order to lie on it.
         Coordinates already within the border return a move-factor of 1.0, signifying they are already within the \
         border.

        Parameters
        ----------
        coordinate : (float, float)
            The x and y coordinates of the pixel to have its move-factor computed.
        """
        theta = self.coordinates_angle_from_x(coordinate)
        radius = self.coordinates_to_radius(coordinate)

        border_radius = self.border_radius_at_theta(theta)

        if radius > border_radius:
            return border_radius / radius
        else:
            return 1.0

    def relocated_coordinate(self, coordinate):
        """Get a coordinate relocated to the source-plane border if initially outside of it.

        Parameters
        ----------
        coordinate : (float, float)
            The x and y coordinates of the pixel to have its move-factor computed.
        """
        move_factor = self.move_factor(coordinate)
        return coordinate[0] * move_factor, coordinate[1] * move_factor

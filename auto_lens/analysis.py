import itertools
import math
import numpy as np

class SourcePlaneGeometry(object):
    """Stores the source-plane geometry, to ensure different components of the source-plane share the
    same geometry"""

    def __init__(self, centre=(0, 0)):
        """

        Parameters
        ----------
        centre : (float, float)
            The centre of the source-plane.
        """
        self.centre = centre

    @property
    def x_cen(self):
        return self.centre[0]

    @property
    def y_cen(self):
        return self.centre[1]

    def coordinates_to_centre(self, coordinates):
        """
        Converts source-plane coordinates to centre of source-plane

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the source-plane coordinate

        Returns
        ----------
        The coordinates at the source-plane center
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
            theta_from_x = 360. + theta_from_x
        return theta_from_x

class SourcePlane(SourcePlaneGeometry):
    """Represents the source-plane and its corresponding traced image sub-coordinates"""

    def __init__(self, sub_coordinates, centre=(0, 0)):
        """

        Parameters
        ----------
        sub_coordinates : list(float, float)
            The x and y coordinates of each traced image sub-pixel
        centre : (float, float)
            The centre of the source-plane.
        """

        super(SourcePlane, self).__init__(centre)

        self.sub_coordinates = sub_coordinates

    def setup_border(self, border_mask):
        """ Setup the source-plane border using for this source-plane.

        Parameters
        ----------
        border_mask : list(bool)
            A list indicating for each source-plane coordinate whether it is part of the source-plane border.
            *True* - This pixel is part of the source-plane border.
            *False* - Thiss pixel is not part of the source-plane border

        """
        self.border = SourcePlaneBorder(list(itertools.compress(self.sub_coordinates, border_mask)), centre=self.centre)

    def relocate_coordinates_outside_border(self):
        """ Move all source-plane coordinates outside of its source-plane border to the edge of its border"""
        self.sub_coordinates = list(map(lambda r : self.border.get_relocated_coordinate(r), self.sub_coordinates))

class SourcePlaneSparse(SourcePlane):
    """Represents the source-plane, including a sparse coordinate list."""

    # TODO : This is all untested - just experimenting with how best to handle the sparse grid and give you a sense
    # TODO : of the problem. See the Coordinates file for a description.

    def __init__(self, sub_coordinates, sparse_coordinates, sub_to_sparse, centre=(0, 0)):
        """

        Parameters
        ----------
        sub_coordinates : list(float, float)
            The x and y coordinates of each traced image sub-pixel
        sparse_coordinates : list(float, float)
            The x and y coordinates of each traced image sparse-pixel
        sub_to_sparse : list(int)
            The integer entry in sparse_clusters (and sparse_coordinates) that each sub-pixel corresponds to.
        centre : (float, float)
            The centre of the source-plane.
        """

        super(SourcePlaneSparse, self).__init__(sub_coordinates, centre)

        self.sparse_coordinates = sparse_coordinates
        self.sub_to_sparse= sub_to_sparse

class SourcePlaneBorder(SourcePlaneGeometry):
    """Represents the source-plane coordinates on the source-plane border. Each coordinate is stored alongside its
    distance from the source-plane centre (radius) and angle from the x-axis (theta)"""

    def __init__(self, coordinates, centre=(0.0, 0.0)):
        """

        Parameters
        ----------
        coordinates : list(float, float)
            The x and y coordinates of the source-plane border.
        centre : (float, float)
            The centre of the source-plane.
        """

        super(SourcePlaneBorder, self).__init__(centre)

        self.coordinates = coordinates
        self.thetas = list(map(lambda  r : self.coordinates_angle_from_x(r), coordinates))
        self.radii = list(map(lambda r: self.coordinates_to_radius(r), coordinates))

    def setup_polynomial(self, polynomial_degree=3):
        """The source-plane border is fitted with a polynomial for r as a function of theta. This polynomial is used
        for relocating source-plane coordinates outside the border to its edge

        Parameters
        ----------
        polynomial_degree : int
            The degree of the polynomial to be fitted to the source-plane border
        """
        self.polynomial = np.polyfit(self.thetas, self.radii, polynomial_degree)

    def get_border_radius_at_theta(self, theta):
        """For a an angle theta from the x-axis, return the border radius via the polynomial fit"""
        return np.polyval(self.polynomial, theta)

    def get_move_factor(self, coordinate):
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

        border_radius = self.get_border_radius_at_theta(theta)

        if (radius > border_radius):
            return border_radius/radius
        else:
            return 1.0

    def get_relocated_coordinate(self, coordinate):
        """Get a coordinate relocated to the source-plane border if initially outside of it.

        Parameters
        ----------
        coordinate : (float, float)
            The x and y coordinates of the pixel to have its move-factor computed.
        """
        move_factor = self.get_move_factor(coordinate)
        return coordinate[0]*move_factor, coordinate[1]*move_factor
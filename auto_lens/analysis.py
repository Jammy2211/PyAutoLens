import itertools
import math
import numpy as np
from profile import profile
import sklearn.cluster
import scipy.spatial


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


class PixelizationAdaptive(object):
    """An adaptive source-plane pixelization generated using a k-means clusteriing algorithm"""

    def __init__(self, sparse_coordinates, n_clusters):
        """

        Parameters
        ----------
        sparse_coordinates : list(float, float)
            The x and y coordinates of each traced image sparse-pixel
        n_clusters : int
            The source-plane resolution or number of k-means clusters to be used.
        """

        self.clusters = KMeans(sparse_coordinates, n_clusters)
        self.voronoi = Voronoi(points=self.clusters.cluster_centers_)


# TODO : Should we do away with these classes are just directly put them in the PixelizationAdaptive constructor?
# TODO : I did this so I could mess around with unit tests, happy to just get rid of them but left them for now.

class RegularizationMatrix(np.ndarray):
    """Class used for generating the regularization matrix H, which describes how each source-plane pixel is
    regularized by other source-plane pixels during the source-reconstruction.

    (Python linear algrebra uses ndarray, not matrix, so this inherites from the former."""

    # TODO: All test cases assume one, constant, regularization coefficient (i.e. all regularization_weights = 1.0).
    # TODO : Need to add test cases for different regularization_weights

    def __new__(cls, dimension, regularization_weights, no_vertices, pixel_pairs):
        """
        Setup a new regularization matrix

        Parameters
        ----------
        dimension : int
            The dimensions of the square regularization matrix
        regularization_weights : list(float)
            A vector of regularization weights of each source-pixel
        no_vertices : list(int)
            The number of Voronoi vertices each source-plane pixel shares with other pixels
        pixel_pairs : list(float, float)
            A list of all pixel-pairs in the source-plane, as computed by the Voronoi griding routine.
        """

        obj = np.zeros(shape=(dimension, dimension)).view(cls)
        obj = obj.make_via_pairs(dimension, regularization_weights, no_vertices, pixel_pairs)

        return obj

    @staticmethod
    def make_via_pairs(dimension, regularization_weights, no_vertices, pixel_pairs):
        """
        Setup a new Voronoi adaptive griding regularization matrix, bypassing matrix multiplication by exploiting the
        symmetry in pixel-neighbourings.

        Parameters
        ----------
        dimension : int
            The dimensions of the square regularization matrix
        regularization_weights : list(float)
            A vector of regularization weights of each source-pixel
        no_vertices : list(int)
            The number of Voronoi vertices each source-plane pixel shares with other pixels
        pixel_pairs : list(float, float)
            A list of all pixel-pairs in the source-plane, as computed by the Voronoi gridding routine.
        """

        matrix = np.zeros(shape=(dimension, dimension))

        reg_weight = regularization_weights ** 2

        for i in range(dimension):
            matrix[i, i] += no_vertices[i] * reg_weight[i]

        for j in range(len(pixel_pairs)):
            matrix[pixel_pairs[j, 0], pixel_pairs[j, 1]] -= reg_weight[i]
            matrix[pixel_pairs[j, 1], pixel_pairs[j, 0]] -= reg_weight[i]

        return 2.0 * matrix


class KMeans(sklearn.cluster.KMeans):
    """An adaptive source-plane pixelization generated using a (weighted) k-means clusteriing algorithm"""

    def __init__(self, points, n_clusters):
        super(KMeans, self).__init__(n_clusters=n_clusters)
        self.fit(points)


class Voronoi(scipy.spatial.Voronoi):
    def __init__(self, points):
        super(Voronoi, self).__init__(points, qhull_options='Qbb Qc Qx')

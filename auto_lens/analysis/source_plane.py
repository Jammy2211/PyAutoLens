import itertools
import math
import numpy as np
from ..profile import profile
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

    def __init__(self, coordinates, centre=(0, 0)):
        """

        Parameters
        ----------
        coordinates : [(float, float)]
            The x and y coordinates of each traced image sub-pixel
        centre : (float, float)
            The centre of the source-plane.
        """

        super(SourcePlane, self).__init__(centre)

        self.coordinates = coordinates

    def border_with_mask_and_polynomial_degree(self, border_mask, polynomial_degree):
        return SourcePlaneBorder(list(itertools.compress(self.coordinates, border_mask)), polynomial_degree,
                                 centre=self.centre)

    def relocate_coordinates_outside_border_with_mask_and_polynomial_degree(self, mask, polynomial_degree):
        self.relocate_coordinates_outside_border(self.border_with_mask_and_polynomial_degree(mask, polynomial_degree))

    def relocate_coordinates_outside_border(self, border):
        """ Move all source-plane coordinates outside of its source-plane border to the edge of its border"""
        self.coordinates = list(map(lambda r: border.relocated_coordinate(r), self.coordinates))


class SourcePlaneSparse(SourcePlane):
    """Represents the source-plane, including a sparse coordinate list."""

    # TODO : This is all untested - just experimenting with how best to handle the sparse grid and give you a sense
    # TODO : of the problem. See the Coordinates file for a description.

    def __init__(self, coordinates, sparse_coordinates, sub_to_sparse, centre=(0, 0)):
        """

        Parameters
        ----------
        coordinates : [(float, float)]
            The x and y coordinates of each traced image sub-pixel
        sparse_coordinates : list(float, float)
            The x and y coordinates of each traced image sparse-pixel
        sub_to_sparse : [int]
            The integer entry in sparse_clusters (and sparse_coordinates) that each sub-pixel corresponds to.
        centre : (float, float)
            The centre of the source-plane.
        """

        super(SourcePlaneSparse, self).__init__(coordinates, centre)

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


class KMeans(sklearn.cluster.KMeans):
    """An adaptive source-plane pixelization generated using a (weighted) k-means clusteriing algorithm"""

    def __init__(self, points, n_clusters):
        super(KMeans, self).__init__(n_clusters=n_clusters)
        self.fit(points)


class Voronoi(scipy.spatial.Voronoi):
    def __init__(self, source_pixel_centers):
        super(Voronoi, self).__init__(source_pixel_centers, qhull_options='Qbb Qc Qx Qm')

        self.neighbors = [[] for _ in range(len(source_pixel_centers))]

        for pair in reversed(self.ridge_points):
            self.neighbors[pair[0]].append(pair[1])
            self.neighbors[pair[1]].append(pair[0])

        self.neighbors_total = list(map(lambda x: len(x), self.neighbors))


class RegularizationMatrix(np.ndarray):
    """Class used for generating the regularization matrix H, which describes how each source-plane pixel is
    regularized by other source-plane pixels during the source-reconstruction.

    Python linear algrebra uses ndarray, not matrix, so this inherites from the former."""

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
            matrix[i][i] += no_vertices[i] * reg_weight[i]

        for j in range(len(pixel_pairs)):
            matrix[pixel_pairs[j, 0], pixel_pairs[j, 0]] += reg_weight[pixel_pairs[j, 1]]
            matrix[pixel_pairs[j, 1], pixel_pairs[j, 1]] += reg_weight[pixel_pairs[j, 0]]
            matrix[pixel_pairs[j, 0], pixel_pairs[j, 1]] -= reg_weight[pixel_pairs[j, 0]]
            matrix[pixel_pairs[j, 1], pixel_pairs[j, 0]] -= reg_weight[pixel_pairs[j, 0]]
            matrix[pixel_pairs[j, 0], pixel_pairs[j, 1]] -= reg_weight[pixel_pairs[j, 1]]
            matrix[pixel_pairs[j, 1], pixel_pairs[j, 0]] -= reg_weight[pixel_pairs[j, 1]]

        return matrix


def sub_coordinates_to_source_pixels_via_nearest_neighbour(sub_coordinates, source_pixel_centers):
    """ Match a set of sub image-pixel coordinates to their closest source-pixels, using the source-pixel centers (x,y).

        This method uses a nearest neighbour search between every sub_image-pixel coordinate and set of source-pixel \
        centers, thus it is slow when the number of sub image-pixel coordinates or source-pixels is large. However, it
        is probably the fastest routine for low numbers of sub image-pixels and source-pixels.

        Parameters
        ----------
        sub_coordinates : [(float, float)]
            The x and y sub image-pixel coordinates to be matched to the source-pixel centers.
        source_pixel_centers: [(float, float)
            The source-pixels centers the sub image-pixel coordinates are matched with.

        Returns
        ----------
        sub_image_pixel_to_source_pixel_index : [int]
            The index in source_pixel_centers each sub_coordinate is matched with. (e.g. if the fifth sub_coordinate \
            is closest to the 3rd source-pixel in source_pixel_centers, sub_image_pixel_to_source_pixel_index[4] = 2).

     """

    sub_image_pixel_to_source_pixel_index = []

    for sub_coordinate in sub_coordinates:
        distances = map(lambda centers: compute_squared_separation(sub_coordinate, centers), source_pixel_centers)

        sub_image_pixel_to_source_pixel_index.append(np.argmin(distances))

    return sub_image_pixel_to_source_pixel_index


def sub_coordinates_to_source_pixels_via_sparse_pairs(sub_coordinates, source_pixel_centers, source_pixel_neighbors,
                                                      sub_coordinate_to_sparse_coordinate_index,
                                                      sparse_coordinate_to_source_pixel_index):
    """ Match a set of sub image-pixel coordinates to their closest source-pixel, using the source-pixel centers (x,y).

        This method uses a sparsely sampled grid of sub image-pixel coordinates with known source-pixel pairings and \
        the source-pixels neighbors to speed up the function. This is optimal when the number of sub image-pixels or \
        source-pixels is large. Thus, the sparse grid of sub_coordinates must have had a source pixelization \
        derived (e.g. using the KMeans class) and the neighbors of each source-pixel must be known \
        (e.g. using the Voronoi class). Both must have been performed prior to this function call.

        In a realistic lens analysis, the sparse sub_coordinates will correspond to the center of each image pixel \
        (traced to the source-plane) or an even sparser grid of image-pixels. The sub_coordinates will be the sub \
        image-pixels (again, traced to the source-plane). A benefit of this is the source-pixelization (e.g. using \
        KMeans) will be dervied using significantly fewer sub_coordinates, offering run-time speedup.
        
        In the routine below, some variables and function names refer to a 'sparse_source_pixel'. This term describes a \
        source-pixel that we have paired to a sub_coordinate using the sparse grid of image pixels. Thus, it may not \
        actually be that sub_coordinate's closest source-pixel (the routine will eventually determine this). \
        Therefore, a 'sparse source-pixel' does not refer to a sparse set of source-pixels.

        Parameters
        ----------
        sub_coordinates : [(float, float)]
            The x and y sub_coordinates to be matched to the source_pixel centers.
        source_pixel_centers: [(float, float)
            The source_pixel centers the sub_coordinates are matched with.
        source_pixel_neighbors : [[]]
            The neighboring source_pixels of each source_pixel, computed via the Voronoi grid (e.g. if the fifth source_pixel \
            neighbors source_pixels 7, 9 and 44, source_pixel_neighbors[4] = [6, 8, 43])
        sparse_coordinate_to_coordinates_index : [int]
            The index in sub_coordinates each sparse sub_coordinate is closest too (e.g. if the fifth sparse sub_coordinate \
            is closest to the 3rd sub_coordinate in sub_coordinates, sparse_coordinate_to_coordinates_index[4] = 2).
        sparse_coordinate_to_source_pixel_index : [int]
            The index in source_pixel_centers each sparse sub_coordinate closest too (e.g. if the fifth sparse sub_coordinate \
            is closest to the 3rd source_pixel in source_pixel_centers, sparse_coordinates_to_source_pixel_index[4] = 2).

        Returns
        ----------
        sub_image_pixel_to_source_pixel_index : [int]
            The index in source_pixel_centers each match sub_coordinate is matched with. (e.g. if the fifth match sub_coordinate \
            is closest to the 3rd source_pixel in source_pixel_centers, sub_image_pixel_to_source_pixel_index[4] = 2).

     """

    sub_image_pixel_to_source_pixel_index = []

    for sub_coordinate_index, sub_coordinate in enumerate(sub_coordinates):

        nearest_sparse_coordinate_index = find_index_of_nearest_sparse_coordinate(sub_coordinate_index,
                                                                                  sub_coordinate_to_sparse_coordinate_index)

        nearest_sparse_source_pixel_index = find_index_of_nearest_sparse_source_pixel(nearest_sparse_coordinate_index,
                                                                                      sparse_coordinate_to_source_pixel_index)

        while True:

            separation_of_sub_coordinate_and_sparse_source_pixel = \
                find_separation_of_sub_coordinate_and_nearest_sparse_source_pixel(source_pixel_centers,
                                                                                  sub_coordinate,
                                                                                  nearest_sparse_source_pixel_index)

            neighboring_source_pixel_index, separation_of_sub_coordinate_and_neighboring_source_pixel = \
                find_separation_and_index_of_nearest_neighboring_source_pixel(sub_coordinate, source_pixel_centers,
                                                                              source_pixel_neighbors[
                                                                                  nearest_sparse_source_pixel_index])

            if separation_of_sub_coordinate_and_sparse_source_pixel < separation_of_sub_coordinate_and_neighboring_source_pixel:
                break
            else:
                nearest_sparse_source_pixel_index = neighboring_source_pixel_index

        # If this pixel is closest to the original pixel, it has been paired successfully with its nearest neighbor.
        sub_image_pixel_to_source_pixel_index.append(nearest_sparse_source_pixel_index)

    return sub_image_pixel_to_source_pixel_index


def find_index_of_nearest_sparse_coordinate(index, coordinate_to_sparse_coordinates_index):
    return coordinate_to_sparse_coordinates_index[index]


def find_index_of_nearest_sparse_source_pixel(nearest_sparse_coordinate_index,
                                              sparse_coordinates_to_source_pixel_index):
    return sparse_coordinates_to_source_pixel_index[nearest_sparse_coordinate_index]


def find_separation_of_sub_coordinate_and_nearest_sparse_source_pixel(source_pixel_centers, sub_coordinate,
                                                                      source_pixel_index):
    nearest_sparse_source_pixel_center = source_pixel_centers[source_pixel_index]
    return compute_squared_separation(sub_coordinate, nearest_sparse_source_pixel_center)


def find_separation_and_index_of_nearest_neighboring_source_pixel(sub_coordinate, source_pixel_centers,
                                                                  source_pixel_neighbors):
    """For a given source_pixel, we look over all its adjacent neighbors and find the neighbor whose distance is closest to
    our input coordinaates.
    
        Parameters
        ----------
        sub_coordinate : (float, float)
            The x and y coordinate to be matched with the neighboring set of source_pixels.
        source_pixel_centers: [(float, float)
            The source_pixel centers the coordinates are matched with.
        source_pixel_neighbors : list
            The neighboring source_pixels of the sparse source_pixel the coordinate is currently matched with

        Returns
        ----------
        source_pixel_neighbor_index : int
            The index in source_pixel_centers of the closest source_pixel neighbor.
        source_pixel_neighbor_separation : float
            The separation between the input coordinate and closest source_pixel neighbor
    
    """

    separation_from_neighbor = list(map(lambda neighbors:
                                        compute_squared_separation(sub_coordinate, source_pixel_centers[neighbors]),
                                        source_pixel_neighbors))

    closest_separation_index = min(xrange(len(separation_from_neighbor)), key=separation_from_neighbor.__getitem__)

    return source_pixel_neighbors[closest_separation_index], separation_from_neighbor[closest_separation_index]


def compute_squared_separation(coordinate1, coordinate2):
    """Computes the squared separation of two coordinates (no square root for efficiency)"""
    return (coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2

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
    def __init__(self, cluster_centers):

        super(Voronoi, self).__init__(cluster_centers, qhull_options='Qbb Qc Qx Qm')

        self.neighbors = [[] for _ in range(len(cluster_centers))]

        for pair in reversed(self.ridge_points):
            self.neighbors[pair[0]].append(pair[1])
            self.neighbors[pair[1]].append(pair[0])

        self.neighbors_total = list(map(lambda x : len(x) , self.neighbors))

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
            matrix[i][i] += no_vertices[i] * reg_weight[i]

        for j in range(len(pixel_pairs)):
            matrix[pixel_pairs[j, 0], pixel_pairs[j, 0]] += reg_weight[pixel_pairs[j, 1]]
            matrix[pixel_pairs[j, 1], pixel_pairs[j, 1]] += reg_weight[pixel_pairs[j, 0]]
            matrix[pixel_pairs[j, 0], pixel_pairs[j, 1]] -= reg_weight[pixel_pairs[j, 0]]
            matrix[pixel_pairs[j, 1], pixel_pairs[j, 0]] -= reg_weight[pixel_pairs[j, 0]]
            matrix[pixel_pairs[j, 0], pixel_pairs[j, 1]] -= reg_weight[pixel_pairs[j, 1]]
            matrix[pixel_pairs[j, 1], pixel_pairs[j, 0]] -= reg_weight[pixel_pairs[j, 1]]

        return matrix


def coordinates_to_clusters_via_nearest_neighbour(coordinates, cluster_centers):
    """ Match a set of coordinates to their closest clusters, using the cluster centers (x,y).

        This method uses a nearest neighbour search between every coordinate and set of cluster centers, thus it is \
        slow when the number of coordinates or clusters is large. However, it is probably the fastest routine for low \
        numbers of coordinates.

        Parameters
        ----------
        coordinates : [(float, float)]
            The x and y coordinates to be matched to the cluster centers.
        cluster_centers: [(float, float)
            The cluster centers the coordinates are matched with.

        Returns
        ----------
        coordinates_to_cluster_index : [int]
            The index in cluster_centers each match coordinate is matched with. (e.g. if the fifth match coordinate \
            is closest to the 3rd cluster in cluster_centers, coordinates_to_cluster_index[4] = 2).

     """

    coordinates_to_cluster_index = []

    for coordinate in coordinates:
        distances = map(lambda centers: compute_squared_separation(coordinate, centers), cluster_centers)

        coordinates_to_cluster_index.append(np.argmin(distances))

    return coordinates_to_cluster_index


def coordinates_to_clusters_via_sparse_pairs(coordinates, cluster_centers, cluster_neighbors,
                                             coordinate_to_sparse_coordinate_index,
                                             sparse_coordinate_to_cluster_index):
    """ Match a set of coordinates to their closest clusters, using the cluster centers (x,y).

        This method uses a sparsely sampled grid of coordinates with known cluster pairings and Voronoi vertices to \
        speed up the function for cases where the number of coordinates or clusters is large. Thus, the sparse grid of \
        coordinates must have had a clustering alogirthm and Voronoi gridding performed prior to this function.

        In a realistic lens analysis, the sparse coordinates will correspond to something like the center of each \
        image pixel (traced to the source-plane), whereas the coordinates to be matched will be the subgridded \
        image-pixels (again, already traced to the source-plane). An important benefit of this is therefore that the \
        clusteiring alogirthm is also sped up by being run on fewer coordinates.
        
        In the routine below, some variables and function names refer to a 'sparse cluster'. This term describes a \
        cluster that we have paired to a coordinate to using the sparse grid of image pixels. Thus, it may not \
        actually be that coordinate's closest cluster. \
        Therefore, a 'sparse cluster' does not refer to a sparse set of clusters.

        Parameters
        ----------
        coordinates : [(float, float)]
            The x and y coordinates to be matched to the cluster centers.
        cluster_centers: [(float, float)
            The cluster centers the coordinates are matched with.
        cluster_neighbors : [[]]
            The neighboring clusters of each cluster, computed via the Voronoi grid (e.g. if the fifth cluster \
            neighbors clusters 7, 9 and 44, cluster_neighbors[4] = [6, 8, 43])
        sparse_coordinate_to_coordinates_index : [int]
            The index in coordinates each sparse coordinate is closest too (e.g. if the fifth sparse coordinate \
            is closest to the 3rd coordinate in coordinates, sparse_coordinate_to_coordinates_index[4] = 2).
        sparse_coordinate_to_cluster_index : [int]
            The index in cluster_centers each sparse coordinate closest too (e.g. if the fifth sparse coordinate \
            is closest to the 3rd cluster in cluster_centers, sparse_coordinates_to_cluster_index[4] = 2).

        Returns
        ----------
        coordinates_to_cluster_index : [int]
            The index in cluster_centers each match coordinate is matched with. (e.g. if the fifth match coordinate \
            is closest to the 3rd cluster in cluster_centers, coordinates_to_cluster_index[4] = 2).

     """

    coordinates_to_cluster_index = []

    for coordinate_index, coordinate in enumerate(coordinates):
        
        nearest_sparse_coordinate_index = find_index_of_nearest_sparse_coordinate(coordinate_index, 
                                                                               coordinate_to_sparse_coordinate_index)

        nearest_sparse_cluster_index = find_index_of_nearest_sparse_cluster(nearest_sparse_coordinate_index,
                                                                         sparse_coordinate_to_cluster_index)

        while True:

            separation_of_coordinate_and_sparse_cluster = find_separation_of_coordinate_and_nearest_sparse_cluster(cluster_centers,
                                                          coordinate, nearest_sparse_cluster_index)

            neighboring_cluster_index, separation_of_coordinate_and_neighboring_cluster = \
                find_separation_and_index_of_nearest_neighboring_cluster(coordinate, cluster_centers, cluster_neighbors[
                    nearest_sparse_cluster_index])

            if separation_of_coordinate_and_sparse_cluster < separation_of_coordinate_and_neighboring_cluster:
                break
            else:
                nearest_sparse_cluster_index = neighboring_cluster_index

        # If this pixel is closest to the original pixel, it has been paired successfully with its nearest neighbor.
        coordinates_to_cluster_index.append(nearest_sparse_cluster_index)

    return coordinates_to_cluster_index

def find_index_of_nearest_sparse_coordinate(index, coordinate_to_sparse_coordinates_index):
    return coordinate_to_sparse_coordinates_index[index]

def find_index_of_nearest_sparse_cluster(nearest_sparse_coordinate_index, sparse_coordinates_to_cluster_index):
    return sparse_coordinates_to_cluster_index[nearest_sparse_coordinate_index]

def find_separation_of_coordinate_and_nearest_sparse_cluster(cluster_centers, coordinate, cluster_index):
    nearest_sparse_cluster_center = cluster_centers[cluster_index]
    return compute_squared_separation(coordinate, nearest_sparse_cluster_center)

def find_separation_and_index_of_nearest_neighboring_cluster(coordinate, cluster_centers, cluster_neighbors):
    """For a given cluster, we look over all its adjacent neighbors and find the neighbor whose distance is closest to
    our input coordinaates.
    
        Parameters
        ----------
        coordinate : (float, float)
            The x and y coordinate to be matched with the neighboring set of clusters.
        cluster_centers: [(float, float)
            The cluster centers the coordinates are matched with.
        cluster_neighbors : list
            The neighboring clusters of the sparse cluster the coordinate is currently matched with

        Returns
        ----------
        cluster_neighbor_index : int
            The index in cluster_centers of the closest cluster neighbor.
        cluster_neighbor_separation : float
            The separation between the input coordinate and closest cluster neighbor
    
    """

    separation_from_neighbor = list(map(lambda neighbors :
                               compute_squared_separation(coordinate, cluster_centers[neighbors]), cluster_neighbors))

    closest_separation_index = min(xrange(len(separation_from_neighbor)), key=separation_from_neighbor.__getitem__)

    return cluster_neighbors[closest_separation_index], separation_from_neighbor[closest_separation_index]

def compute_squared_separation(coordinate1, coordinate2):
    """Computes the squared separation of two coordinates (no square root for efficiency)"""
    return (coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2
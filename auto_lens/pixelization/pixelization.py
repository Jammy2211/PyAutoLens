import numpy as np
import sklearn.cluster
import scipy.spatial


class Pixelization(object):

    def __init__(self, pixels):
        """
        Abstract base class for a source-plane Pixelization, which for an input set of Plane coordinates sets up the \
        source-plane pixelization, mappings between (sub-)image and source pixels and the regularization matrix.

        Parameters
        ----------
        pixels : int
            The number of source pixels in the pixelization.
        """
        self.pixels = pixels

    def create_mapping_matrix(self, sub_to_source):
        """
        Create a new mapping matrix, which describes the fractional unit surface brightness counts between each \
        image-pixel and source pixel. The mapping matrix is denoted 'f_ij' in Warren & Dye 2003,
        Nightingale & Dye 2015 and Nightingale, Dye & Massey 2018.

        The matrix is dimensions [image_pixels, source_pixels], where a non-zero entry represents an \
        image-pixel to source-pixel mapping. For example, if image-pixel 0 maps to source-pixel 2, element \
        (0, 2) of the mapping matrix will = 1.

        The mapping matrix is created using sub-gridding. Here, each observed image-pixel is divided into a finer \
        sub_grid. For example, if the sub-grid is size 4x4, each image-pixel is split into a uniform 4 x 4 \
        sub grid and all 16 sub-pixels are individually traced to the source-plane and paired with a source-pixel.

        The entries in the mapping matrix are fractional surface brightness values representing the number of \
        sub-pixel to source-pixel mappings. For example if 3 sub-pixels from image pixel 4 map to source pixel 2,
        then element (4,2) of the mapping matrix will = 3.0 * (1/grid_size_sub**2) = 3/16 = 0.1875. The unit tests \
        show more examples of this.

        Parameters
        ----------
        sub_to_source : [int, int]
            The source_pixel index each image and sub-image pixel is matched with. (e.g. if the fifth
            sub_coordinate of the third image_grid pixel is closest to the 3rd source-pixel in source_pixel_centers,
            sub_to_source[2,4] = 2).
        """

        image_pixels = sub_to_source.shape[0]
        sub_pixels = sub_to_source.shape[1]
        sub_grid_fraction = (1.0 / sub_pixels)

        mapping_matrix = np.zeros((image_pixels, self.pixels))

        for image_pixel in range(image_pixels):
            for sub_pixel in range(sub_pixels):
                mapping_matrix[image_pixel, sub_to_source[image_pixel, sub_pixel]] += sub_grid_fraction

        return mapping_matrix

    def create_regularization_matrix(self, regularization_weights, source_neighbors):
        """
        Setup a new regularization matrix, bypassing matrix multiplication by exploiting a list of source pixel \
        neighbors.

        Parameters
        ----------
        regularization_weights : list(float)
            The regularization weight of each source-pixel
        source_neighbors : [[]]
            A list of the neighbors of each source pixel.
        """

        regularization_matrix = np.zeros(shape=(self.pixels, self.pixels))

        reg_weight = regularization_weights ** 2

        for i in range(self.pixels):
            for j in source_neighbors[i]:
                regularization_matrix[i, i] += reg_weight[j]
                regularization_matrix[j, j] += reg_weight[j]
                regularization_matrix[i, j] -= reg_weight[j]
                regularization_matrix[j, i] -= reg_weight[j]

        return regularization_matrix


class SquarePixelization(Pixelization):
    # TODO: Implement me
    pass


class VoronoiPixelization(Pixelization):

    def __init__(self, pixels, regularization_coefficient=1.0):
        """
        A (non-adaptive) voronoi pixelization, which represents a source-plane composed of Voronoi cells built via \
        a k-means clsutering algorithm.

        Parameters
        ----------
        pixels : int
            The number of source pixels in the pixelization.
        """
        super(VoronoiPixelization, self).__init__(pixels)
        self.regularization_coefficient = regularization_coefficient

    def compute_mapping_matrix(self, image_coordinates, sub_coordinates, mapper_cluster):
        """
        Compute the mapping matrix of the pixelization. For a VoronoiPixelization this involves the following steps:

        1) Extract the cluster-coordinates, which are a sparse set of source-coordinates used to perform efficient \
        k-means clustering.
        2) Performs kmeans clustering on the cluster-coordinates.
        3) Derive a Voronoi grid of the cluster-coordinates.
        4) Compute the mapping between the complete set of sub-coordinates and source-pixels.
        5) Use these mappings to compute the mapping matrix.

        Parameters
        ----------
        image_coordinates : [[float, float]]
            The x and y image-coordinates which are used to extract the cluster coordinates.
        sub_coordinates : [[float, float]]
            The x and y sub-coordinates which are matched to the voronoi source-pixels.
        mapper_cluster : auto_lens.imaging.grids.GridMapperCluster
            The mapping between cluster-pixels and image / sub pixels.
        """

        cluster_coordinates = image_coordinates[mapper_cluster.cluster_to_image]
        kmeans = self.kmeans_cluster(cluster_coordinates)
        voronoi = self.compute_voronoi_grid(kmeans.cluster_centers_)
        neighbors = self.compute_source_neighbors(voronoi.ridge_points)
        sub_to_source = self.compute_sub_to_source(sub_coordinates, kmeans.cluster_centers_, neighbors,
                                                   mapper_cluster.image_to_cluster, kmeans.labels_)

        return self.create_mapping_matrix(sub_to_source)

    def kmeans_cluster(self, cluster_coordinates):
        """Perform k-means clustering on the cluster_coordinates to compute the k-means clusters which represent \
        source-pixels.

        Parameters
        ----------
        cluster_coordinates : ndarray
            The x and y cluster-coordinates which are used to derive the k-means pixelization.
        """
        kmeans = sklearn.cluster.KMeans(self.pixels)
        return kmeans.fit(cluster_coordinates)

    @staticmethod
    def compute_voronoi_grid(source_coordinates):
        """Compute the Voronoi grid of the source pixelization, using the source-pixel centers.

        A list of each Voronoi cell's.

        Parameters
        ----------
        source_coordinates : ndarray
            The x and y image_grid to derive the Voronoi grid_coords.
        """
        return scipy.spatial.Voronoi(source_coordinates, qhull_options='Qbb Qc Qx Qm')

    def compute_source_neighbors(self, ridge_points):
        """Compute the neighbors of every source-pixel, where the neighbor is a list of every source-pixel a given \
        source-pixel shares a vertex with.
        
        For the Voronoi pixelization, the ridge points are used to derive this.

        Parameters
        ----------
        ridge_points : scipy.spatial.Voronoi.ridge_points
            Each Voronoi-ridge (two indexes representing a source-pixel mapping).
        """
        source_neighbors = [[] for _ in range(self.pixels)]

        for pair in reversed(ridge_points):
            source_neighbors[pair[0]].append(pair[1])
            source_neighbors[pair[1]].append(pair[0])

        return source_neighbors

    @staticmethod
    def compute_sub_to_source(sub_coordinates, source_centers, source_neighbors, image_to_cluster, cluster_to_source):
        """ Compute the mappings between a set of sub-coordinates and source-pixels.

        For the Voronoi grid, the sparse set of 'cluster_coordinates' are fed to the kmeans algorithm and thus provide \
        the mappings between only a sub-set of image-pixels and source-pixels. To get the full set of sub-to-source \
        mappings, we therefore need to use these sparse mappings to derive the full set of mappings. This would be \
        slow using a basic nearest neighbor algoirthm, therefore the source-pixel neighbors are used to localize \
        the nearest neighbor search.

        In this routine, some variables and function names refer to a 'sparse_source_'. This term describes a \
        source-pixel that we have paired to a sub_coordinate using the sparse_coordinate of an image coordinate. \
        Thus, it may not actually be that sub_coordinate's closest source-pixel (the routine will eventually
        determine this).

        Parameters
        ----------
        sub_coordinates : [[float, float]]
            The x and y sub_grid to be matched to the source_pixel centers.
        image_total : int
            The total number of image_grid image_to_pixel in the image_grid.
        sub_total : int
            The total number of sub_grid image_to_pixel in the image_grid sub_grid-grid_coords.
        source_centers: [[float, float]]
            The source_pixel centers the sub_grid are matched with.
        source_neighbors : [[]]
            The neighboring source_pixels of each source_pixel, computed via the Voronoi grid_coords (e.g. if the fifth source_pixel \
            neighbors source_pixels 7, 9 and 44, source_neighbors[4] = [6, 8, 43])
        sub_to_cluster : [int]
            The index in sub_grid each sparse_grid sub_coordinate is closest too (e.g. if the fifth sparse_grid sub_coordinate \
            is closest to the 3rd sub_coordinate in sub_grid, sub_to_sparse[4] = 2).
        cluster_to_source : [int]
            The index in source_pixel_centers each sparse_grid sub_coordinate closest too (e.g. if the fifth sparse_grid sub_coordinate \
            is closest to the 3rd source_pixel in source_pixel_centers, sparse_to_source[4] = 2).

        Returns
        ----------
        sub_to_source : [int, int]
            The source_pixel index each image and sub-image pixel is matched with. (e.g. if the fifth
            sub_coordinate of the third image_grid pixel is closest to the 3rd source-pixel in source_pixel_centers,
            sub_to_source[2,4] = 2).

         """

        image_pixels = sub_coordinates.shape[0]
        sub_pixels = sub_coordinates.shape[1]

        sub_to_source = np.zeros((image_pixels, sub_pixels))

        for image_index in range(len(sub_coordinates)):
            sub_index = 0
            for sub_coordinate in sub_coordinates[image_index]:

                nearest_sparse = image_to_cluster[image_index]
                nearest_sparse_source = cluster_to_source[nearest_sparse]

                while True:

                    sub_to_sparse_source_distance = compute_sub_to_nearest_sparse_source(source_centers,
                                                                                         sub_coordinate,
                                                                                         nearest_sparse_source)

                    neighboring_source_index, sub_to_neighboring_source_distance = \
                        compute_nearest_neighboring_source_and_distance(sub_coordinate, source_centers,
                                                                        source_neighbors[nearest_sparse_source])

                    if sub_to_sparse_source_distance < sub_to_neighboring_source_distance:
                        break
                    else:
                        nearest_sparse_source = neighboring_source_index

                # If this pixel is closest to the original pixel, it has been paired successfully with its nearest neighbor.
                sub_to_source[image_index, sub_index] = nearest_sparse_source
                sub_index += 1

        return sub_to_source


def compute_sub_to_nearest_sparse_source(source_centers, sub_coordinate, source_pixel):
    nearest_sparse_source_pixel_center = source_centers[source_pixel]
    return compute_squared_separation(sub_coordinate, nearest_sparse_source_pixel_center)


def compute_nearest_neighboring_source_and_distance(sub_coordinate, source_centers, source_neighbors):
    """For a given source_pixel, we look over all its adjacent neighbors and find the neighbor whose distance is closest to
    our input coordinaates.

    Parameters
    ----------
    sub_coordinate : (float, float)
        The x and y coordinate to be matched with the neighboring set of source_pixels.
    source_centers: [(float, float)
        The source_pixel centers the image_grid are matched with.
    source_neighbors : list
        The neighboring source_pixels of the sparse_grid source_pixel the coordinate is currently matched with

    Returns
    ----------
    source_pixel_neighbor_index : int
        The index in source_pixel_centers of the closest source_pixel neighbor.
    source_pixel_neighbor_separation : float
        The separation between the input coordinate and closest source_pixel neighbor

    """

    separation_from_neighbor = list(map(lambda neighbors:
                                        compute_squared_separation(sub_coordinate, source_centers[neighbors]),
                                        source_neighbors))

    closest_separation_index = min(range(len(separation_from_neighbor)), key=separation_from_neighbor.__getitem__)

    return source_neighbors[closest_separation_index], separation_from_neighbor[closest_separation_index]


def compute_squared_separation(coordinate1, coordinate2):
    """Computes the squared separation of two image_grid (no square root for efficiency)"""
    return (coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2

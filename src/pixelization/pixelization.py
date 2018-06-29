import numpy as np
import sklearn.cluster
import scipy.spatial

from src import exc


class Pixelization(object):

    def __init__(self, pixels, regularization_coefficients=(1.0,), source_signal_scale=1.0):
        """
        Abstract base class for a pixelization, which discretizes a set of image (sub-)coordinates into groups of \
        source-pixels thus determining the mappings between image (sub-)coordinates and source-pixels.

        The regularization matrix of the pixeliztion is also computed, which is used to enforces smoothness a on the \
        source-reconstruction.

        A number of 1D and 2D arrays are used to represent mappings betwen image, sub, source and cluster pixels. The \
        nomenclature here follows grid_to_grid, such that it maps the index of a value on one grid to another. For \
        example:

        - source_to_image[2] = 5 tells us that the 3rd source-pixel maps to the 6th image-pixel.
        - sub_to_source[4,2] = 2 tells us that the 3rd sub-pixel in the 5th image-pixel maps to the 3rd source-pixel.

        NOTE: To make it intuitive, this documentation assumes a Pixelization is always applied to a source-plane and \
        therefore represents the mappings between the image and source planes. However, in principle, a Pixelization \
        could be applied to the image-plane.

        Parameters
        ----------
        pixels : int
            The number of source pixels in the pixelization.
        """
        self.pixels = pixels
        self.regularization_coefficients = regularization_coefficients
        self.source_signal_scale = source_signal_scale

    def compute_mapping_and_regularization_matrix(self, source_coordinates, source_sub_coordinates, mapper_cluster):
        raise exc.PixelizationException('compute_mapping_matrix must be over-riden by a Pixelization.')

    def create_mapping_matrix(self, sub_to_source):
        """
        Create a new mapping matrix, which describes the fractional unit surface brightness counts between each \
        image-pixel and source pixel. The mapping matrix is denoted 'f_ij' in Warren & Dye 2003,
        Nightingale & Dye 2015 and Nightingale, Dye & Massey 2018.

        The matrix has dimensions [image_pixels, source_pixels] and non-zero entries represents an \
        image-pixel to source-pixel mapping. For example, if image-pixel 0 maps to source-pixel 2, element \
        [0,2] of the mapping matrix will = 1.

        The mapping matrix is created using sub-gridding. Here, each observed image-pixel is divided into a finer \
        sub_grid. For example, if the sub-grid is size 4x4, each image-pixel is split into a uniform 4 x 4 \
        sub grid and all 16 sub-pixels are individually traced to the source-plane and paired with a source-pixel.

        The entries in the mapping matrix therefore become fractional surface brightness values, representing the \
        number of sub-pixel to source-pixel mappings. For example if 3 sub-pixels from image-pixel 4 map to \
        source-pixel 2, then element [4,2] of the mapping matrix will = 3.0 * (1/grid_size_sub**2) = 3/16 = 0.1875. \
        See test_pixelization.py for clearer examples of this.

        Parameters
        ----------
        sub_to_source : [int, int]
            The source_pixel index each image and sub-image pixel is matched with. (e.g. if the fifth
            sub-pixel of the third image-pixel is matched with the 3rd source-pixel, sub_to_source[2,4] = 2).
        """

        image_pixels = sub_to_source.shape[0]
        sub_pixels = sub_to_source.shape[1]
        sub_grid_fraction = (1.0 / sub_pixels)

        mapping_matrix = np.zeros((image_pixels, self.pixels))

        for image_pixel in range(image_pixels):
            for sub_pixel in range(sub_pixels):
                mapping_matrix[image_pixel, sub_to_source[image_pixel, sub_pixel]] += sub_grid_fraction

        return mapping_matrix

    def create_constant_regularization_matrix(self, source_neighbors):
        """
        Setup a constant regularization matrix, where source-pixels are regularized with one another in 1 direction
        with 1 constant regularization coefficient.

        Matrix multiplication is bypassed by exploiting a list of source pixel neighbors.

        Parameters
        ----------
        source_neighbors : [[]]
            A list of the neighbors of each source pixel.
        """

        regularization_matrix = np.zeros(shape=(self.pixels, self.pixels))

        reg_coeff = self.regularization_coefficients[0] ** 2.0

        for i in range(self.pixels):
            regularization_matrix[i, i] += 1e-8
            for j in source_neighbors[i]:
                regularization_matrix[i, i] += reg_coeff
                regularization_matrix[i, j] -= reg_coeff

        return regularization_matrix

    def compute_source_signals(self, image_to_source, galaxy_image):
        """Compute the (scaled) signal in each source-pixel, where the signal is the sum of its image-pixel fluxes. \
        These source-signals are then used to compute the effective regularization weight of each source-pixel.

        The source signals are scaled in the following ways:

        1) Divided by the number of image pixels in the source-pixel, to ensure all source-pixels have the same \
        'relative' signal (i.e. a source pixel with 10 images pixels doesn't have x2 the signal of one with 5).

        2) Divided by the maximum source-signal, so that all signals vary between 0 and 1. This ensures that the \
        regularizations weights they're used to compute are well defined.

        3) Raised to the power of the hyper-parameter *source_signal_scale*, so the method can control the relative \
        contribution of the diffrent regions of regularization.
        """

        source_signals = np.zeros((self.pixels))
        source_sizes = np.zeros((self.pixels))

        for image_pixel in range(galaxy_image.shape[0]):
            source_signals[image_to_source[image_pixel]] += galaxy_image[image_pixel]
            source_sizes[image_to_source[image_pixel]] += 1

        source_signals /= source_sizes
        source_signals /= max(source_signals)

        return source_signals ** self.source_signal_scale

    def compute_regularization_weights(self, source_signals):
        """Compute the regularization weights, which represent the effective regularization coefficient of every \
        source-pixel. These are computed using the (scaled) source-signal in each source-pixel.

        Two regularization coefficients are used which map to:

        1) source_signals - This regularizes source-plane pixels with a high source-signal (i.e. where the source is).
        2) 1.0 - source_signals - This regularizes source-plane pixels with a low source-signal (i.e. background sky)
        """
        return (self.regularization_coefficients[0] * source_signals +
                self.regularization_coefficients[1] * (1.0 - source_signals)) ** 2.0

    def create_weighted_regularization_matrix(self, regularization_weights, source_neighbors):
        """
        Setup a weighted regularization matrix, where all source-pixels are regularized with one another in both
        directions different effective regularization coefficients.

        Matrix multiplication is bypassed by exploiting a list of source pixel neighbors.

        Parameters
        ----------
        regularization_weights : list(float)
            The regularization weight of each source-pixel
        source_neighbors : [[]]
            A list of the neighbors of each source pixel.
        """

        regularization_matrix = np.zeros(shape=(self.pixels, self.pixels))

        reg_weight = regularization_weights ** 2.0

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

    def __init__(self, pixels, regularization_coefficients=(1.0,)):
        """
        Abstract base class for a Voronoi pixelization, which represents source-pixels as a set of centers where \
        all of the nearest-neighbor source-coordinates (i.e. traced image-pixels) are mapped to them.

        This forms a Voronoi grid source-plane, the properties of which are used for fast calculations, defining the \
        regularization matrix and visualization.

        Parameters
        ----------
        pixels : int
            The number of source pixels in the pixelization.
        """

        super(VoronoiPixelization, self).__init__(pixels, regularization_coefficients)

    @staticmethod
    def compute_voronoi_grid(source_coordinates):
        """Compute the Voronoi grid of the pixelization, using the source-pixel centers.

        Parameters
        ----------
        source_coordinates : ndarray
            The x and y image_grid to derive the Voronoi grid_coords.
        """
        return scipy.spatial.Voronoi(source_coordinates, qhull_options='Qbb Qc Qx Qm')

    def compute_source_neighbors(self, ridge_points):
        """Compute the neighbors of every source-pixel, where the neighbor is a list of every source-pixel a given \
        source-pixel shares a vertex with. The ridge points of the Voronoi grid are used to derive this.

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
    def compute_sub_to_source(source_sub_coordinates, source_centers, source_neighbors, image_to_cluster,
                              source_to_cluster):
        """ Compute the mappings between a set of source sub-pixels and source-pixels, using the sub-coordinates and
        source-pixel centers.

        For the Voronoi pixelizations, a sparse set of 'cluster-pixels' are used to determine the source pixelization. \
        These provide the mappings between only a sub-set of sub-pixels / image-pixels and source-pixels.

        To determine the complete set of sub-pixel to source-pixel mappings, we must therefore pair every sub-pixel to \
        its nearest source-pixel (using the sub-pixel's source-plane coordinate and source-pixel center). Using a full \
        nearest neighbor search to do this is slow, thus the source-pixel neighbors (derived via the Voronoi grid) \
        is used to localize each nearest neighbor search.

        In this routine, some variables and function names refer to a 'sparse_source_'. This term describes a \
        source-pixel that we have paired to a sub_coordinate using the sparse_coordinate of an image coordinate. \
        Thus, it may not actually be that sub_coordinate's closest source-pixel (the routine will eventually
        determine this).

        Parameters
        ----------
        source_sub_coordinates : [[float, float]]
            The x and y source sub-grid coordinates which are to be matched with their closest source-pixels.
        source_centers: [[float, float]]
            The coordinate of the center of every source-pixel.
        source_neighbors : [[]]
            The neighboring source_pixels of each source_pixel, computed via the Voronoi grid_coords. \
            (e.g. if the fifth source_pixel neighbors source_pixels 7, 9 and 44, source_neighbors[4] = [6, 8, 43])
        image_to_cluster : [int]
            The index in the image-grid each sparse cluster-pixel is closest too (e.g. if the fifth image-pixel \
            is closest to the 3rd cluster-pixel, image_to_sparse[4] = 2).
        source_to_cluster : [int]
            The mapping between every source-pixel and cluster-pixel (e.g. if the fifth source-pixel maps to \
            the 3rd cluster_pixel, source_to_cluster[4] = 2).

        Returns
        ----------
        sub_to_source : [int, int]
            The mapping between every sub-pixel and source-pixel. (e.g. if the fifth sub-pixel of the third \
            image-pixel maps to the 3rd source-pixel, sub_to_source[2,4] = 2).

         """

        def compute_source_sub_to_nearest_sparse_source(source_centers, sub_coordinate, source_pixel):
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

            closest_separation_index = min(range(len(separation_from_neighbor)),
                                           key=separation_from_neighbor.__getitem__)

            return source_neighbors[closest_separation_index], separation_from_neighbor[closest_separation_index]

        def compute_squared_separation(coordinate1, coordinate2):
            """Computes the squared separation of two image_grid (no square root for efficiency)"""
            return (coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2

        image_pixels = source_sub_coordinates.shape[0]
        sub_pixels = source_sub_coordinates.shape[1]

        sub_to_source = np.zeros((image_pixels, sub_pixels), dtype=int)

        for image_index in range(len(source_sub_coordinates)):
            sub_index = 0
            for sub_coordinate in source_sub_coordinates[image_index]:

                nearest_sparse = image_to_cluster[image_index]
                nearest_sparse_source = source_to_cluster[nearest_sparse]

                while True:

                    source_sub_to_sparse_source_distance = compute_source_sub_to_nearest_sparse_source(source_centers,
                                                                                                       sub_coordinate,
                                                                                                       nearest_sparse_source)

                    neighboring_source_index, sub_to_neighboring_source_distance = \
                        compute_nearest_neighboring_source_and_distance(sub_coordinate, source_centers,
                                                                        source_neighbors[nearest_sparse_source])

                    if source_sub_to_sparse_source_distance < sub_to_neighboring_source_distance:
                        break
                    else:
                        nearest_sparse_source = neighboring_source_index

                # If this pixel is closest to the original pixel, it has been paired successfully with its nearest neighbor.
                sub_to_source[image_index, sub_index] = nearest_sparse_source
                sub_index += 1

        return sub_to_source


class ClusterPixelization(VoronoiPixelization):

    def __init__(self, pixels, regularization_coefficients=(1.0,)):
        """
        A cluster pixelization, which represents source-pixels as a set of centers where all of the nearest-neighbor \
        source-coordinates (i.e. traced image-pixels) are mapped to them.

        For this pixelization, a set of cluster-pixels (defined in the image-plane as a sparse uniform grid of \
        image-pixels) determine the source-pixel centers .

        Parameters
        ----------
        pixels : int
            The number of source pixels in the pixelization.
        """
        super(ClusterPixelization, self).__init__(pixels, regularization_coefficients)

    def compute_mapping_and_regularization_matrix(self, source_coordinates, source_sub_coordinates, mapper_cluster):
        """
        Compute the mapping matrix of the cluster pixelization by following these steps:

        1) Extract the cluster-coordinates (see grids.GridMapperCluster) from the source-plane and use these as the \
        source-pixel centres.
        3) Derive a Voronoi grid using these source-pixel centres.
        4) Compute the mapping between all image sub-coordinates and source-pixels.
        5) Use these mappings to compute the mapping matrix.

        Parameters
        ----------
        source_coordinates : [[float, float]]
            The x and y source-coordinates.
        source_sub_coordinates : [[float, float]]
            The x and y sub-coordinates.
        mapper_cluster : auto_lens.imaging.grids.GridMapperCluster
            The mapping between cluster-pixels and image / source pixels.
        """

        if self.pixels is not len(mapper_cluster.cluster_to_image):
            raise exc.PixelizationException('ClusteringPixelization - The input number of pixels in the constructor'
                                            'is not the same as the length of the cluster_to_image mapper')

        source_centers = source_coordinates[mapper_cluster.cluster_to_image]
        source_to_image = np.arange(0, self.pixels)
        voronoi = self.compute_voronoi_grid(source_centers)
        source_neighbors = self.compute_source_neighbors(voronoi.ridge_points)
        sub_to_source = self.compute_sub_to_source(source_sub_coordinates, source_centers, source_neighbors,
                                                   mapper_cluster.image_to_cluster, source_to_image)

        # TODO : WE have to compute regularization matrix here as tey both use source_neighbors. Can we make source
        # TODO : neigbors a class property so these are separate functions (that doon't repeat the calculation?)

        return self.create_mapping_matrix(sub_to_source), self.create_constant_regularization_matrix(source_neighbors)


class AmorphousPixelization(VoronoiPixelization):

    def __init__(self, pixels, regularization_coefficients=(1.0, 1.0, 2.0)):
        """
        An amorphous pixelization, which represents source-pixels as a set of centers where all of the \
        nearest-neighbor source-coordinates (i.e. traced image-pixels) are mapped to them.

        For this pixelization, a set of cluster-pixels (defined in the image-plane as a sparse uniform grid of \
        image-pixels) are used to determine a set of source-plane coordinates. These coordinates are then fed into a \
        weighted k-means clustering algorithm, such that the source-pixel centers adapt to the unlensed source \
        surface-brightness profile.

        Parameters
        ----------
        pixels : int
            The number of source pixels in the pixelization.
        """
        super(AmorphousPixelization, self).__init__(pixels, regularization_coefficients)

    def compute_mapping_and_regularization_matrix(self, source_coordinates, source_sub_coordinates, mapper_cluster):
        """
        Compute the mapping matrix of the amorphous pixelization by following these steps:

        1) Extract the cluster-coordinates (see grids.GridMapperCluster) from the source-plane.
        2) Performs weighted kmeans clustering on these cluster-coordinates to compute the source-pixel centres.
        3) Derive a Voronoi grid using these source-pixel centres.
        4) Compute the mapping between all image sub-coordinates and source-pixels.
        5) Use these mappings to compute the mapping matrix.

        Parameters
        ----------
        source_coordinates : [[float, float]]
            The x and y source-coordinates.
        source_sub_coordinates : [[float, float]]
            The x and y sub-coordinates.
        mapper_cluster : auto_lens.imaging.grids.GridMapperCluster
            The mapping between cluster-pixels and image / source pixels.
        """

        cluster_coordinates = source_coordinates[mapper_cluster.cluster_to_image]
        source_centers, source_to_cluster = self.kmeans_cluster(cluster_coordinates)
        voronoi = self.compute_voronoi_grid(source_centers)
        source_neighbors = self.compute_source_neighbors(voronoi.ridge_points)
        sub_to_source = self.compute_sub_to_source(source_sub_coordinates, source_centers, source_neighbors,
                                                   mapper_cluster.image_to_cluster, source_to_cluster)

        return self.create_mapping_matrix(sub_to_source), self.create_constant_regularization_matrix(source_neighbors)

    def kmeans_cluster(self, cluster_coordinates):
        """Perform k-means clustering on the cluster_coordinates to compute the k-means clusters which represent \
        source-pixels.

        Parameters
        ----------
        cluster_coordinates : ndarray
            The x and y cluster-coordinates which are used to derive the k-means pixelization.
        """
        kmeans = sklearn.cluster.KMeans(self.pixels)
        km = kmeans.fit(cluster_coordinates)
        return km.cluster_centers_, km.labels_

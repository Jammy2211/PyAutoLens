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
        regularization_coefficients : (float,)
            The regularization coefficients used to smooth the source reconstruction.
        source_signal_scale : float
            A hyper-parameter which scales the signal attributed to each source-pixel, used for weight regularization.
        """
        self.pixels = pixels
        self.regularization_coefficients = regularization_coefficients
        self.source_signal_scale = source_signal_scale

    def compute_pixelization_matrices(self, source_coordinates, source_sub_coordinates, sub_to_image,
                                      image_pixels, sub_grid_size, mapper_cluster=None):
        raise exc.PixelizationException('compute_mapping_matrix must be over-riden by a Pixelization.')

    def create_mapping_matrix(self, sub_to_source, sub_to_image, image_pixels, sub_grid_size):
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
            The source-pixel index each image and sub-image pixel is matched with. (e.g. if the fifth
            sub-pixel is matched with the 3rd source-pixel, sub_to_source[4] = 2).
        sub_to_image : [int, int]
            The image-pixel index each image pixel and sub-image pixel is matched with (e.g. if the fifth sub-pixel is
            inside the 2nd image pixel, sub_to_image[4] = 1)
        image_pixels : int
            The number of image pixels in the masked image the pixelization is fitting.
        sub_grid_size : int
            The size of the sub-grid.
        """

        sub_pixels = sub_to_source.shape[0]
        sub_grid_fraction = (1.0 / sub_grid_size ** 2.0)

        mapping_matrix = np.zeros((image_pixels, self.pixels))

        for sub_index in range(sub_pixels):
            mapping_matrix[sub_to_image[sub_index], sub_to_source[sub_index]] += sub_grid_fraction

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

        for image_index in range(galaxy_image.shape[0]):
            source_signals[image_to_source[image_index]] += galaxy_image[image_index]
            source_sizes[image_to_source[image_index]] += 1

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

    def compute_source_neighbors(self):
        raise NotImplementedError("compute_source_neighbors must be over-riden by a Pixelization")


class RectangularPixelization(Pixelization):

    def __init__(self, shape=(3,3), regularization_coefficients=(1.0,)):
        """A rectangular and Cartesian pixelization, which represents source-pixels on a uniform rectangular grid
        of size x_pixels x y_pixels.

        Like image's, the indexing of the rectangular grid begins in the top-left corner and goes right and down.

        Parameters
        -----------
        shape : (int, int)
            The dimensions of the rectangular grid of pixels (x_pixels, y_pixel)
        regularization_coefficients : (float,)
            The regularization coefficients used to smooth the source reconstruction.
        """
        
        if shape[0] <= 2 or shape[1] <= 2:
            raise exc.PixelizationException('The recentgular pixelization must be at least dimensions 3x3')

        super(RectangularPixelization, self).__init__(shape[0]*shape[1], regularization_coefficients)

        self.shape = shape

    class Geometry(object):

        def __init__(self, y_min, y_max, x_min, x_max, y_pixel_scale, x_pixel_scale):
            self.y_min = y_min
            self.y_max = y_max
            self.x_min = x_min
            self.x_max = x_max
            self.y_pixel_scale = y_pixel_scale
            self.x_pixel_scale = x_pixel_scale

        def arc_second_to_pixel_index_x(self, coordinate):
            return np.floor((coordinate - self.x_min) / self.x_pixel_scale)

        def arc_second_to_pixel_index_y(self, coordinate):
            return np.floor((coordinate - self.y_min) / self.y_pixel_scale)

    def compute_geometry(self, coordinates, buffer=1e-8):

        y_min = np.min(coordinates[:,0]) - buffer
        y_max = np.max(coordinates[:,0]) + buffer
        x_min = np.min(coordinates[:,1]) - buffer
        x_max = np.max(coordinates[:,1]) + buffer
        y_pixel_scale = (y_max - y_min) / self.shape[0]
        x_pixel_scale = (x_max - x_min) / self.shape[1]

        return self.Geometry(y_min, y_max, x_min, x_max, y_pixel_scale, x_pixel_scale)

    def compute_source_neighbors(self):
        """Compute the neighbors of every source-pixel as a list of the source-pixel index's each source-pixel \
        shares a vertex with.

        The regular uniform grid geometry is used to compute this.
        """

        def compute_corner_neighbors(source_neighbors):

            source_neighbors[0] = [1, self.shape[1]]
            source_neighbors[self.shape[1]-1] = [self.shape[1]-2, self.shape[1]+self.shape[1]-1]
            source_neighbors[self.pixels-self.shape[1]] = [self.pixels-self.shape[1]*2 ,self.pixels-self.shape[1]+1]
            source_neighbors[self.pixels-1] = [self.pixels-self.shape[1]-1, self.pixels-2]

            return source_neighbors

        def compute_top_edge_neighbors(source_neighbors):

            for pix in range(1, self.shape[1]-1):
                pixel_index = pix
                source_neighbors[pixel_index] = [pixel_index-1, pixel_index+1, pixel_index+self.shape[1]]

            return source_neighbors

        def compute_left_edge_neighbors(source_neighbors):

            for pix in range(1, self.shape[0]-1):
                pixel_index = pix*self.shape[1]
                source_neighbors[pixel_index] = [pixel_index-self.shape[1], pixel_index+1, pixel_index+self.shape[1]]

            return source_neighbors

        def compute_right_edge_neighbors(source_neighbors):

            for pix in range(1, self.shape[0] - 1):
                pixel_index = pix*self.shape[1] + self.shape[1] - 1
                source_neighbors[pixel_index] = [pixel_index-self.shape[1], pixel_index-1, pixel_index+self.shape[1]]
                
            return source_neighbors

        def compute_bottom_edge_neighbors(source_neighbors):

            for pix in range(1, self.shape[1]-1):
                pixel_index = self.pixels - pix - 1
                source_neighbors[pixel_index] = [pixel_index-self.shape[1], pixel_index-1, pixel_index+1]

            return source_neighbors
        
        def compute_central_neighbors(source_neighbors):
            
            for y in range(1, self.shape[1]-1):
                for x in range(1, self.shape[0]-1):


                    pixel_index = x*self.shape[1] + y
                    source_neighbors[pixel_index] = [pixel_index-self.shape[1], pixel_index-1, pixel_index+1,
                                                     pixel_index+self.shape[1]]

            return source_neighbors

        source_neighbors = [[] for _ in range(self.pixels)]
    
        source_neighbors = compute_corner_neighbors(source_neighbors)
        source_neighbors = compute_top_edge_neighbors(source_neighbors)
        source_neighbors = compute_left_edge_neighbors(source_neighbors)
        source_neighbors = compute_right_edge_neighbors(source_neighbors)
        source_neighbors = compute_bottom_edge_neighbors(source_neighbors)
        source_neighbors = compute_central_neighbors(source_neighbors)
        
        return source_neighbors

    def compute_pixel_to_source(self, source_coordinates, geometry):
        """Compute the mappings between a set of image pixels (or sub-pixels) and source-pixels, using the image's
        traced source-plane coordinates (or sub-coordinates) and the uniform rectangular pixelization's geometry.

        Parameters
        ----------
        source_coordinates : [[float, float]]
            The x and y source coordinates (or sub-coordinaates) which are to be matched with their source-pixels.
        geometry : Geometry
            The rectangular pixel grid's geometry.
        """
        pixel_to_source = np.zeros(source_coordinates.shape[0], dtype='int')

        for index, source_coordinate in enumerate(source_coordinates):

            y_pixel = geometry.arc_second_to_pixel_index_y(source_coordinate[0])
            x_pixel = geometry.arc_second_to_pixel_index_x(source_coordinate[1])

            pixel_to_source[index] = y_pixel*self.shape[1] + x_pixel

        return pixel_to_source

    def compute_pixelization_matrices(self, source_coordinates, source_sub_coordinates, sub_to_image,
                                      image_pixels, sub_grid_size, mapper_cluster=None):
        """
        Compute the pixelization matrices of the rectangular pixelization by following these steps:

        1) Setup the rectangular grid geometry, by making its corner appear at the higher / lowest x and y source sub-
        coordinates.
        2) Pair image and sub-image pixels to the rectangular grid using their traced coordinates and its geometry.

        Parameters
        ----------
        source_coordinates : [[float, float]]
            The x and y source-coordinates.
        source_sub_coordinates : [[float, float]]
            The x and y sub-coordinates.
        mapper_cluster : auto_lens.imaging.grids.GridMapperCluster
            The mapping between cluster-pixels and image / source pixels.
        """

        geometry = self.compute_geometry(source_sub_coordinates)
        source_neighbors = self.compute_source_neighbors()
        image_to_source = self.compute_pixel_to_source(source_coordinates, geometry)
        sub_to_source = self.compute_pixel_to_source(source_sub_coordinates, geometry)

        mapping_matrix =  self.create_mapping_matrix(sub_to_source, sub_to_image, image_pixels, sub_grid_size)
        regularization_matrix = self.create_constant_regularization_matrix(source_neighbors)

        return PixelizationMatrices(mapping_matrix, regularization_matrix)

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
        regularization_coefficients : (float,)
            The regularization coefficients used to smooth the source reconstruction.
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
        """Compute the neighbors of every source-pixel as a list of the source-pixel index's each source-pixel \
        shares a vertex with.

        The ridge points of the Voronoi grid are used to derive this.

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

    def compute_image_to_source(self, source_coordinates, source_centers, source_neighbors, image_to_cluster,
                                source_to_cluster):
        """ Compute the mappings between a set of image pixels and source-pixels, using the image's traced \
        source-plane coordinates and the source-pixel centers.

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

        image_to_source = np.zeros((source_coordinates.shape[0]), dtype=int)

        for image_index, source_coordinate in enumerate(source_coordinates):

            nearest_cluster = image_to_cluster[image_index]

            image_to_source[image_index] = self.pair_coordinate_and_pixel(source_coordinate, nearest_cluster,
                                                         source_centers, source_neighbors, source_to_cluster)

        return image_to_source

    def compute_sub_to_source(self, source_sub_coordinates, source_centers, source_neighbors,
                              sub_to_image, image_to_cluster, source_to_cluster):
        """ Compute the mappings between a set of sub-image pixels and source-pixels, using the image's traced \
        source-plane sub-coordinates and the source-pixel centers. This uses the source-neighbors to perform a graph \
        search when pairing pixels, for efficiency.

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
        sub_to_image : [int, int]
            The image-pixel index each image pixel and sub-image pixel is matched with (e.g. if the fifth sub-pixel is
            inside the 2nd image pixel, sub_to_image[4] = 1)
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

        sub_to_source = np.zeros((source_sub_coordinates.shape[0]), dtype=int)

        for sub_index, sub_coordinate in enumerate(source_sub_coordinates):

            nearest_cluster = image_to_cluster[sub_to_image[sub_index]]

            sub_to_source[sub_index] = self.pair_coordinate_and_pixel(sub_coordinate, nearest_cluster, source_centers,
                                                                      source_neighbors, source_to_cluster)

        return sub_to_source

    def pair_coordinate_and_pixel(self, coordinate, nearest_cluster, source_centers, source_neighbors, source_to_cluster):
        """ Compute the mappings between a set of sub-image pixels and source-pixels, using the image's traced \
        source-plane sub-coordinates and the source-pixel centers. This uses the source-neighbors to perform a graph \
        search when pairing pixels, for efficiency.

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
        coordinate : [float, float]
            The x and y source sub-grid coordinates which are to be matched with their closest source-pixels.
        nearest_cluster : int
            The nearest source-pixel defined on the cluster-pixel grid.
        source_centers: [[float, float]]
            The coordinate of the center of every source-pixel.
        source_neighbors : [[]]
            The neighboring source_pixels of each source_pixel, computed via the Voronoi grid_coords. \
            (e.g. if the fifth source_pixel neighbors source_pixels 7, 9 and 44, source_neighbors[4] = [6, 8, 43])
        source_to_cluster : [int]
            The mapping between every source-pixel and cluster-pixel (e.g. if the fifth source-pixel maps to \
            the 3rd cluster_pixel, source_to_cluster[4] = 2).
         """

        nearest_sparse_source = source_to_cluster[nearest_cluster]

        while True:

            source_sub_to_sparse_source_distance = self.compute_distance_to_nearest_sparse_source(source_centers,
                                                                                                  coordinate,
                                                                                                  nearest_sparse_source)

            neighboring_source_index, sub_to_neighboring_source_distance = \
                self.compute_nearest_neighboring_source_and_distance(coordinate, source_centers,
                                                                     source_neighbors[nearest_sparse_source])

            if source_sub_to_sparse_source_distance < sub_to_neighboring_source_distance:
                return nearest_sparse_source
            else:
                nearest_sparse_source = neighboring_source_index

    def compute_distance_to_nearest_sparse_source(self, source_centers, coordinate, source_pixel):
        nearest_sparse_source_pixel_center = source_centers[source_pixel]
        return self.compute_squared_separation(coordinate, nearest_sparse_source_pixel_center)

    def compute_nearest_neighboring_source_and_distance(self, coordinate, source_centers, source_neighbors):
        """For a given source_pixel, we look over all its adjacent neighbors and find the neighbor whose distance is closest to
        our input coordinaates.

        Parameters
        ----------
        coordinate : (float, float)
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
                                            self.compute_squared_separation(coordinate, source_centers[neighbors]),
                                            source_neighbors))

        closest_separation_index = min(range(len(separation_from_neighbor)),
                                       key=separation_from_neighbor.__getitem__)

        return source_neighbors[closest_separation_index], separation_from_neighbor[closest_separation_index]

    @staticmethod
    def compute_squared_separation(coordinate1, coordinate2):
        """Computes the squared separation of two image_grid (no square root for efficiency)"""
        return (coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2


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
        regularization_coefficients : (float,)
            The regularization coefficients used to smooth the source reconstruction.
        """
        super(ClusterPixelization, self).__init__(pixels, regularization_coefficients)

    def compute_pixelization_matrices(self, source_coordinates, source_sub_coordinates, mapper_cluster, sub_to_image,
                                      image_pixels, sub_grid_size):
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
                                                   sub_to_image, mapper_cluster.image_to_cluster, source_to_image)

        mapping_matrix =  self.create_mapping_matrix(sub_to_source, sub_to_image, image_pixels, sub_grid_size)
        regularization_matrix = self.create_constant_regularization_matrix(source_neighbors)

        return PixelizationMatrices(mapping_matrix, regularization_matrix)


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
        regularization_coefficients : (float,)
            The regularization coefficients used to smooth the source reconstruction.
        """
        super(AmorphousPixelization, self).__init__(pixels, regularization_coefficients)

    def compute_pixelization_matrices(self, source_coordinates, source_sub_coordinates, mapper_cluster, sub_to_image,
                                      image_pixels, sub_grid_size):
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
                                                   sub_to_image, mapper_cluster.image_to_cluster, source_to_cluster)
        mapping_matrix =  self.create_mapping_matrix(sub_to_source, sub_to_image, image_pixels, sub_grid_size)
        regularization_matrix = self.create_constant_regularization_matrix(source_neighbors)

        return PixelizationMatrices(mapping_matrix, regularization_matrix)

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


class PixelizationMatrices(object):

    def __init__(self, mapping, regularization):

        self.mapping = mapping
        self.regularization = regularization
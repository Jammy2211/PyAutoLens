import numpy as np
import sklearn.cluster
import scipy.spatial

from src import exc
from src.pixelization import covariance_matrix

from src.imaging import mask


class Pixelization(object):

    def __init__(self, pixels, regularization_coefficients=(1.0,), pix_signal_scale=1.0):
        """
        Abstract base class for a pixelization, which discretizes a set of image and sub grid grid into \ 
        pixels. These pixels then fit a  weighted_data-set using a linear inversion, where their regularization matrix
        enforces smoothness between pixel values.

        A number of 1D and 2D arrays are used to represent mappings betwen image, sub, pix, and cluster pixels. The \
        nomenclature here follows grid_to_grid, such that it maps the index of a value on one grid to another. For \
        example:

        - pix_to_image[2] = 5 tells us that the 3rd pixelization-pixel maps to the 6th image-pixel.
        - sub_to_pix[4,2] = 2 tells us that the 5th sub-pixel maps to the 3rd pixelization-pixel.

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        regularization_coefficients : (float,)
            The regularization coefficients used to smooth the pix reconstruction.
        pix_signal_scale : float
            A hyper-parameter which scales the signal attributed to each pixel, used for weighted regularization.
        """
        self.pixels = pixels
        self.regularization_coefficients = regularization_coefficients
        self.pix_signal_scale = pix_signal_scale

    def mapping_matrix_from_sub_to_pix(self, sub_to_pix, grids):
        """
        Create a new mapping matrix, which describes the fractional unit surface brightness counts between each \
        image-pixel and pixel. The mapping matrix is denoted 'f_ij' in Warren & Dye 2003,
        Nightingale & Dye 2015 and Nightingale, Dye & Massey 2018.

        The matrix has dimensions [image_pixels, pix_pixels] and non-zero entries represents an \
        image-pixel to pixel mapping. For example, if image-pixel 0 maps to pixel 2, element \
        [0,2] of the mapping matrix will = 1.

        The mapping matrix is created using sub-gridding. Here, each observed image-pixel is divided into a finer \
        sub_grid. For example, if the sub-grid is sub_grid_size=4, each image-pixel is split into a uniform 4 x 4 \
        sub grid and all 16 sub-pixels are individually paired with pixels.

        The entries in the mapping matrix therefore become fractional surface brightness values, representing the \
        number of sub-pixel to pixel mappings. For example if 3 sub-pixels from image-pixel 4 map to \
        pixel 2, then element [4,2] of the mapping matrix will = 3.0 * (1/sub_grid_size**2) = 3/16 = 0.1875.

        Parameters
        ----------
        grids
        sub_to_pix : [int, int]
            The pixel index each image and sub-image pixel is matched with. (e.g. if the fifth
            sub-pixel is matched with the 3rd pixel, sub_to_pix[4] = 2).

        """

        sub_grid = grids.sub

        mapping_matrix = np.zeros((grids.image.no_pixels, self.pixels))

        for sub_index in range(sub_grid.no_pixels):
            mapping_matrix[sub_grid.sub_to_image[sub_index], sub_to_pix[sub_index]] += sub_grid.sub_grid_fraction

        return mapping_matrix

    def constant_regularization_matrix_from_pix_neighbors(self, pix_neighbors):
        """
        Setup a pixelization's constant regularization matrix (see test_pixelization.py)

        Parameters
        ----------
        pix_neighbors : [[]]
            A list of the neighbors of each pixel.
        """

        regularization_matrix = np.zeros(shape=(self.pixels, self.pixels))

        reg_coeff = self.regularization_coefficients[0] ** 2.0

        for i in range(self.pixels):
            regularization_matrix[i, i] += 1e-8
            for j in pix_neighbors[i]:
                regularization_matrix[i, i] += reg_coeff
                regularization_matrix[i, j] -= reg_coeff

        return regularization_matrix

    def pix_signals_from_images(self, image_to_pix, galaxy_image):
        """Compute the (scaled) signal in each pixel, where the signal is the sum of its image-pixel fluxes. \
        These pix-signals are then used to compute the effective regularization weight of each pixel.

        The pix signals are scaled in the following ways:

        1) Divided by the number of image-pixels in the pixel, to ensure all pixels have the same \
        'relative' signal (i.e. a pixel with 10 images-pixels doesn't have x2 the signal of one with 5).

        2) Divided by the maximum pix-signal, so that all signals vary between 0 and 1. This ensures that the \
        regularizations weights they're used to compute are defined identically for all image units / SNR's.

        3) Raised to the power of the hyper-parameter *pix_signal_scale*, so the method can control the relative \
        contribution of the different regions of regularization.
        """

        pix_signals = np.zeros((self.pixels,))
        pix_sizes = np.zeros((self.pixels,))

        for image_index in range(galaxy_image.shape[0]):
            pix_signals[image_to_pix[image_index]] += galaxy_image[image_index]
            pix_sizes[image_to_pix[image_index]] += 1

        pix_signals /= pix_sizes
        pix_signals /= max(pix_signals)

        return pix_signals ** self.pix_signal_scale

    def regularization_weights_from_pix_signals(self, pix_signals):
        """Compute the regularization weights, which represent the effective regularization coefficient of every \
        pixel. These are computed using the (scaled) pix-signal in each pixel.

        Two regularization coefficients are used which map to:

        1) pix_signals - This regularizes pix-plane pixels with a high pix-signal (i.e. where the pix is).
        2) 1.0 - pix_signals - This regularizes pix-plane pixels with a low pix-signal (i.e. background sky)
        """
        return (self.regularization_coefficients[0] * pix_signals +
                self.regularization_coefficients[1] * (1.0 - pix_signals)) ** 2.0

    def weighted_regularization_matrix_from_pix_neighbors(self, regularization_weights, pix_neighbors):
        """
        Setup a weighted regularization matrix, where all pixels are regularized with one another in both \
        directions using a different effective regularization coefficient.

        Parameters
        ----------
        regularization_weights : list(float)
            The regularization weight of each pixel
        pix_neighbors : [[]]
            A list of the neighbors of each pixel.
        """

        regularization_matrix = np.zeros(shape=(self.pixels, self.pixels))

        reg_weight = regularization_weights ** 2.0

        for i in range(self.pixels):
            for j in pix_neighbors[i]:
                regularization_matrix[i, i] += reg_weight[j]
                regularization_matrix[j, j] += reg_weight[j]
                regularization_matrix[i, j] -= reg_weight[j]
                regularization_matrix[j, i] -= reg_weight[j]

        return regularization_matrix


class RectangularPixelization(Pixelization):

    def __init__(self, shape=(3, 3), regularization_coefficients=(1.0,)):
        """A rectangular pixelization where pixels appear on a Cartesian, uniform and rectangular grid \
        of  shape (rows, columns).

        Like an image grid, the indexing of the rectangular grid begins in the top-left corner and goes right and down.

        Parameters
        -----------
        shape : (int, int)
            The dimensions of the rectangular grid of pixels (x_pixels, y_pixel)
        regularization_coefficients : (float,)
            The regularization coefficients used to smooth the pix reconstruction.
        """

        if shape[0] <= 2 or shape[1] <= 2:
            raise exc.PixelizationException('The rectangular pixelization must be at least dimensions 3x3')

        super(RectangularPixelization, self).__init__(shape[0] * shape[1], regularization_coefficients)

        self.shape = shape

    class Geometry(object):

        def __init__(self, y_min, y_max, x_min, x_max, y_pixel_scale, x_pixel_scale):
            """The geometry of a rectangular grid, defining where the grids top-left, top-right, bottom-left and \
            bottom-right corners are in arc seconds. The arc-second size of each rectangular pixel is also computed.

            Parameters
            -----------

            """
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

    def geometry_from_pix_sub_grid(self, pix_sub_grid, buffer=1e-8):
        """Determine the geometry of the rectangular grid, by alligning it with the outer-most pix_grid grid \
        plus a small buffer.

        Parameters
        -----------
        pix_sub_grid : [[float, float]]
            The x and y pix grid (or sub-coordinaates) which are to be matched with their pixels.
        buffer : float
            The size the grid-geometry is extended beyond the most exterior grid.
        """
        y_min = np.min(pix_sub_grid[:, 0]) - buffer
        y_max = np.max(pix_sub_grid[:, 0]) + buffer
        x_min = np.min(pix_sub_grid[:, 1]) - buffer
        x_max = np.max(pix_sub_grid[:, 1]) + buffer
        y_pixel_scale = (y_max - y_min) / self.shape[0]
        x_pixel_scale = (x_max - x_min) / self.shape[1]

        return self.Geometry(y_min, y_max, x_min, x_max, y_pixel_scale, x_pixel_scale)

    def neighbors_from_pixelization(self):
        """Compute the neighbors of every pixel as a list of the pixel index's each pixel shares a vertex with.

        The uniformity of the rectangular grid's geometry is used to compute this.
        """

        def compute_corner_neighbors(pix_neighbors):

            pix_neighbors[0] = [1, self.shape[1]]
            pix_neighbors[self.shape[1] - 1] = [self.shape[1] - 2, self.shape[1] + self.shape[1] - 1]
            pix_neighbors[self.pixels - self.shape[1]] = [self.pixels - self.shape[1] * 2,
                                                          self.pixels - self.shape[1] + 1]
            pix_neighbors[self.pixels - 1] = [self.pixels - self.shape[1] - 1, self.pixels - 2]

            return pix_neighbors

        def compute_top_edge_neighbors(pix_neighbors):

            for pix in range(1, self.shape[1] - 1):
                pixel_index = pix
                pix_neighbors[pixel_index] = [pixel_index - 1, pixel_index + 1, pixel_index + self.shape[1]]

            return pix_neighbors

        def compute_left_edge_neighbors(pix_neighbors):

            for pix in range(1, self.shape[0] - 1):
                pixel_index = pix * self.shape[1]
                pix_neighbors[pixel_index] = [pixel_index - self.shape[1], pixel_index + 1, pixel_index + self.shape[1]]

            return pix_neighbors

        def compute_right_edge_neighbors(pix_neighbors):

            for pix in range(1, self.shape[0] - 1):
                pixel_index = pix * self.shape[1] + self.shape[1] - 1
                pix_neighbors[pixel_index] = [pixel_index - self.shape[1], pixel_index - 1, pixel_index + self.shape[1]]

            return pix_neighbors

        def compute_bottom_edge_neighbors(pix_neighbors):

            for pix in range(1, self.shape[1] - 1):
                pixel_index = self.pixels - pix - 1
                pix_neighbors[pixel_index] = [pixel_index - self.shape[1], pixel_index - 1, pixel_index + 1]

            return pix_neighbors

        def compute_central_neighbors(pix_neighbors):

            for y in range(1, self.shape[1] - 1):
                for x in range(1, self.shape[0] - 1):
                    pixel_index = x * self.shape[1] + y
                    pix_neighbors[pixel_index] = [pixel_index - self.shape[1], pixel_index - 1, pixel_index + 1,
                                                  pixel_index + self.shape[1]]

            return pix_neighbors

        pixel_neighbors = [[] for _ in range(self.pixels)]

        pixel_neighbors = compute_corner_neighbors(pixel_neighbors)
        pixel_neighbors = compute_top_edge_neighbors(pixel_neighbors)
        pixel_neighbors = compute_left_edge_neighbors(pixel_neighbors)
        pixel_neighbors = compute_right_edge_neighbors(pixel_neighbors)
        pixel_neighbors = compute_bottom_edge_neighbors(pixel_neighbors)
        pixel_neighbors = compute_central_neighbors(pixel_neighbors)

        return pixel_neighbors

    def compute_grid_to_pix(self, grid, geometry):
        """Compute the mappings between a set of image pixels (or sub-pixels) and pixels, using the image's
        traced pix-plane grid (or sub-grid) and the uniform rectangular pixelization's geometry.

        Parameters
        ----------
        grid : [[float, float]]
            The x and y pix grid (or sub-coordinates) which are to be matched with their pixels.
        geometry : Geometry
            The rectangular pixel grid's geometry.
        """
        grid_to_pix = np.zeros(grid.shape[0], dtype='int')

        for index, pix_coordinate in enumerate(grid):
            y_pixel = geometry.arc_second_to_pixel_index_y(pix_coordinate[0])
            x_pixel = geometry.arc_second_to_pixel_index_x(pix_coordinate[1])

            grid_to_pix[index] = y_pixel * self.shape[1] + x_pixel

        return grid_to_pix

    # TODO : RectangularPixelization doesnt need sparse mask, but equivalent functions elsewhere do. Change to *kwrgs?

    def inversion_from_pix_grids(self, grids, sparse_mask=None):
        """
        Compute the pixelization matrices of the rectangular pixelization by following these steps:

        1) Setup the rectangular grid geometry, by making its corner appear at the higher / lowest x and y pix sub-
        grid.
        2) Pair image and sub-image pixels to the rectangular grid using their traced grid and its geometry.

        Parameters
        ----------

        """

        geometry = self.geometry_from_pix_sub_grid(grids.sub)
        pix_neighbors = self.neighbors_from_pixelization()
        image_to_pix = self.compute_grid_to_pix(grids.image, geometry)
        sub_to_pix = self.compute_grid_to_pix(grids.sub, geometry)

        mapping = self.mapping_matrix_from_sub_to_pix(sub_to_pix, grids)
        regularization = self.constant_regularization_matrix_from_pix_neighbors(pix_neighbors)

        return Inversion(mapping, regularization, image_to_pix, sub_to_pix)


class VoronoiPixelization(Pixelization):

    def __init__(self, pixels, regularization_coefficients=(1.0,)):
        """
        Abstract base class for a Voronoi pixelization, which represents pixels as a set of centers where \
        all of the nearest-neighbor pix-grid (i.e. traced image-pixels) are mapped to them.

        This forms a Voronoi grid pix-plane, the properties of which are used for fast calculations, defining the \
        regularization matrix and visualization.

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        regularization_coefficients : (float,)
            The regularization coefficients used to smooth the pix reconstruction.
        """

        super(VoronoiPixelization, self).__init__(pixels, regularization_coefficients)

    @staticmethod
    def voronoi_from_cluster_grid(cluster_grid):
        """Compute the Voronoi grid of the pixelization, using the pixel centers.

        Parameters
        ----------
        cluster_grid : ndarray
            The x and y image_grid to derive the Voronoi grid_coords.
        """
        return scipy.spatial.Voronoi(cluster_grid, qhull_options='Qbb Qc Qx Qm')

    def neighbors_from_pixelization(self, ridge_points):
        """Compute the neighbors of every pixel as a list of the pixel index's each pixel \
        shares a vertex with.

        The ridge points of the Voronoi grid are used to derive this.

        Parameters
        ----------
        ridge_points : scipy.spatial.Voronoi.ridge_points
            Each Voronoi-ridge (two indexes representing a pixel mapping).
        """
        pix_neighbors = [[] for _ in range(self.pixels)]

        for pair in reversed(ridge_points):
            pix_neighbors[pair[0]].append(pair[1])
            pix_neighbors[pair[1]].append(pair[0])

        return pix_neighbors

    def image_to_pix_from_pixelization(self, grids, pix_centers, pix_neighbors, cluster_to_pix, sparse_mask):
        """ Compute the mappings between a set of image pixels and pixels, using the image's traced \
        pix-plane grid and the pixel centers.

        For the Voronoi pixelizations, a cluster set of 'cluster-pixels' are used to determine the pixelization. \
        These provide the mappings between only a sub-set of sub-pixels / image-pixels and pixels.

        To determine the complete set of sub-pixel to pixel mappings, we must therefore pair every sub-pixel to \
        its nearest pixel (using the sub-pixel's pix-plane coordinate and pixel center). Using a full \
        nearest neighbor search to do this is slow, thus the pixel neighbors (derived via the Voronoi grid) \
        is used to localize each nearest neighbor search.

        In this routine, some variables and function names refer to a 'cluster_pix_'. This term describes a \
        pixel that we have paired to a sub_coordinate using the cluster_coordinate of an image coordinate. \
        Thus, it may not actually be that sub_coordinate's closest pixel (the routine will eventually
        determine this).

        Parameters
        ----------

        sparse_mask
        grids
        pix_centers: [[float, float]]
            The coordinate of the center of every pixel.
        pix_neighbors : [[]]
            The neighboring pix_pixels of each pix_pixel, computed via the Voronoi grid_coords. \
            (e.g. if the fifth pix_pixel neighbors pix_pixels 7, 9 and 44, pix_neighbors[4] = [6, 8, 43])
        cluster_to_pix : [int]
            The mapping between every pixel and cluster-pixel (e.g. if the fifth pixel maps to \
            the 3rd cluster_pixel, cluster_to_pix[4] = 2).

        Returns
        ----------
        sub_to_pix : [int, int]
            The mapping between every sub-pixel and pixel. (e.g. if the fifth sub-pixel of the third \
            image-pixel maps to the 3rd pixel, sub_to_pix[2,4] = 2).

         """

        image_to_pix = np.zeros((grids.image.shape[0]), dtype=int)

        for image_index, pix_coordinate in enumerate(grids.image):
            nearest_cluster = sparse_mask.image_to_sparse[image_index]

            image_to_pix[image_index] = self.pair_image_and_pix(pix_coordinate, nearest_cluster,
                                                                pix_centers, pix_neighbors, cluster_to_pix)

        return image_to_pix

    def sub_to_pix_from_pixelization(self, grids, pix_centers, pix_neighbors, cluster_to_pix, sparse_mask):
        """ Compute the mappings between a set of sub-image pixels and pixels, using the image's traced \
        pix-plane sub-grid and the pixel centers. This uses the pix-neighbors to perform a graph \
        search when pairing pixels, for efficiency.

        For the Voronoi pixelizations, a cluster set of 'cluster-pixels' are used to determine the pixelization. \
        These provide the mappings between only a sub-set of sub-pixels / image-pixels and pixels.

        To determine the complete set of sub-pixel to pixel mappings, we must therefore pair every sub-pixel to \
        its nearest pixel (using the sub-pixel's pix-plane coordinate and pixel center). Using a full \
        nearest neighbor search to do this is slow, thus the pixel neighbors (derived via the Voronoi grid) \
        is used to localize each nearest neighbor search.

        In this routine, some variables and function names refer to a 'cluster_pix_'. This term describes a \
        pixel that we have paired to a sub_coordinate using the cluster_coordinate of an image coordinate. \
        Thus, it may not actually be that sub_coordinate's closest pixel (the routine will eventually
        determine this).

        Parameters
        ----------

        grids: mask.CoordinateCollection
            A collection of coordinates for the masked image, subgrid and blurring grid
        sparse_mask: mask.SparseMask
            A mask describing the image pixels that should be used in pixel clustering
        pix_centers: [[float, float]]
            The coordinate of the center of every pixel.
        pix_neighbors : [[]]
            The neighboring pix_pixels of each pix_pixel, computed via the Voronoi grid_coords. \
            (e.g. if the fifth pix_pixel neighbors pix_pixels 7, 9 and 44, pix_neighbors[4] = [6, 8, 43])
        cluster_to_pix : [int]
            The mapping between every pixel and cluster-pixel (e.g. if the fifth pixel maps to \
            the 3rd cluster_pixel, pix_to_cluster[4] = 2).

        Returns
        ----------
        sub_to_pix : [int, int]
            The mapping between every sub-pixel and pixel. (e.g. if the fifth sub-pixel of the third \
            image-pixel maps to the 3rd pixel, sub_to_pix[2,4] = 2).

         """

        sub_to_pix = np.zeros((grids.sub.no_pixels,), dtype=int)

        for sub_index, sub_coordinate in enumerate(grids.sub):
            nearest_cluster = sparse_mask.image_to_sparse[grids.sub.sub_to_image[sub_index]]

            sub_to_pix[sub_index] = self.pair_image_and_pix(sub_coordinate, nearest_cluster, pix_centers,
                                                            pix_neighbors, cluster_to_pix)

        return sub_to_pix

    def pair_image_and_pix(self, coordinate, nearest_cluster, pix_centers, pix_neighbors, cluster_to_pix):
        """ Compute the mappings between a set of sub-image pixels and pixels, using the image's traced \
        pix-plane sub-grid and the pixel centers. This uses the pix-neighbors to perform a graph \
        search when pairing pixels, for efficiency.

        For the Voronoi pixelizations, a cluster set of 'cluster-pixels' are used to determine the pixelization. \
        These provide the mappings between only a sub-set of sub-pixels / image-pixels and pixels.

        To determine the complete set of sub-pixel to pixel mappings, we must therefore pair every sub-pixel to \
        its nearest pixel (using the sub-pixel's pix-plane coordinate and pixel center). Using a full \
        nearest neighbor search to do this is slow, thus the pixel neighbors (derived via the Voronoi grid) \
        is used to localize each nearest neighbor search.

        In this routine, some variables and function names refer to a 'cluster_pix_'. This term describes a \
        pixel that we have paired to a sub_coordinate using the cluster_coordinate of an image coordinate. \
        Thus, it may not actually be that sub_coordinate's closest pixel (the routine will eventually
        determine this).

        Parameters
        ----------
        coordinate : [float, float]
            The x and y pix sub-grid grid which are to be matched with their closest pixels.
        nearest_cluster : int
            The nearest pixel defined on the cluster-pixel grid.
        pix_centers: [[float, float]]
            The coordinate of the center of every pixel.
        pix_neighbors : [[]]
            The neighboring pix_pixels of each pix_pixel, computed via the Voronoi grid_coords. \
            (e.g. if the fifth pix_pixel neighbors pix_pixels 7, 9 and 44, pix_neighbors[4] = [6, 8, 43])
        cluster_to_pix : [int]
            The mapping between every cluster-pixel and pixel (e.g. if the fifth pixel maps to \
            the 3rd cluster_pixel, cluster_to_pix[4] = 2).
         """

        nearest_pix = cluster_to_pix[nearest_cluster]

        while True:

            pix_to_cluster_distance = self.distance_to_nearest_cluster_pix(coordinate, pix_centers, nearest_pix)

            neighboring_pix_index, sub_to_neighboring_pix_distance = \
                self.nearest_neighboring_pix_and_distance(coordinate, pix_centers,
                                                          pix_neighbors[nearest_pix])

            if pix_to_cluster_distance < sub_to_neighboring_pix_distance:
                return nearest_pix
            else:
                nearest_pix = neighboring_pix_index

    def distance_to_nearest_cluster_pix(self, coordinate, pix_centers, nearest_pix):
        nearest_cluster_pix_center = pix_centers[nearest_pix]
        return self.compute_squared_separation(coordinate, nearest_cluster_pix_center)

    def nearest_neighboring_pix_and_distance(self, coordinate, pix_centers, pix_neighbors):
        """For a given pix_pixel, we look over all its adjacent neighbors and find the neighbor whose distance is closest to
        our input coordinates.

        Parameters
        ----------
        coordinate : (float, float)
            The x and y coordinate to be matched with the neighboring set of pix_pixels.
        pix_centers: [(float, float)
            The pix_pixel centers the image_grid are matched with.
        pix_neighbors : []
            The neighboring pix_pixels of the cluster_grid pix_pixel the coordinate is currently matched with

        Returns
        ----------
        pix_neighbors_index : int
            The index in pix_pixel_centers of the closest pix_pixel neighbor.
        separation_from_neighbor : float
            The separation between the input coordinate and closest pix_pixel neighbor

        """

        separation_from_neighbor = list(map(lambda neighbors:
                                            self.compute_squared_separation(coordinate, pix_centers[neighbors]),
                                            pix_neighbors))

        closest_separation_index = min(range(len(separation_from_neighbor)),
                                       key=separation_from_neighbor.__getitem__)

        return pix_neighbors[closest_separation_index], separation_from_neighbor[closest_separation_index]

    @staticmethod
    def compute_squared_separation(coordinate1, coordinate2):
        """Computes the squared separation of two image_grid (no square root for efficiency)"""
        return (coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2


class ClusterPixelization(VoronoiPixelization):

    def __init__(self, pixels, regularization_coefficients=(1.0,)):
        """
        A cluster pixelization, which represents pixels as a set of centers where all of the nearest-neighbor \
        pix-grid (i.e. traced image-pixels) are mapped to them.

        For this pixelization, a set of cluster-pixels (defined in the image-plane as a cluster uniform grid of \
        image-pixels) determine the pixel centers .

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        regularization_coefficients : (float,)
            The regularization coefficients used to smooth the pix reconstruction.
        """
        super(ClusterPixelization, self).__init__(pixels, regularization_coefficients)

    def inversion_from_pix_grids(self, grids, sparse_mask):
        """
        Compute the mapping matrix of the cluster pixelization by following these steps:

        1) Extract the cluster-grid (see grids.GridMapperCluster) from the pix-plane and use these as the \
        pixel centres.
        3) Derive a Voronoi grid using these pixel centres.
        4) Compute the mapping between all image sub-grid and pixels.
        5) Use these mappings to compute the mapping matrix.

        Parameters
        ----------
        grids: mask.CoordinateCollection
            A collection of coordinates for the masked image, subgrid and blurring grid
        sparse_mask: mask.SparseMask
            A mask describing the image pixels that should be used in pixel clustering
        """

        if self.pixels is not len(sparse_mask.sparse_to_image):
            raise exc.PixelizationException('ClusteringPixelization - The input number of pixels in the constructor'
                                            'is not the same as the length of the cluster_to_image mapper')

        pix_centers = grids.image[sparse_mask.sparse_to_image]
        cluster_to_pix = np.arange(0, self.pixels)
        voronoi = self.voronoi_from_cluster_grid(pix_centers)
        pix_neighbors = self.neighbors_from_pixelization(voronoi.ridge_points)
        image_to_pix = self.image_to_pix_from_pixelization(grids, pix_centers, pix_neighbors, cluster_to_pix,
                                                           sparse_mask)
        sub_to_pix = self.sub_to_pix_from_pixelization(grids, pix_centers, pix_neighbors, cluster_to_pix, sparse_mask)

        mapping_matrix = self.mapping_matrix_from_sub_to_pix(sub_to_pix, grids)
        regularization_matrix = self.constant_regularization_matrix_from_pix_neighbors(pix_neighbors)

        return Inversion(mapping_matrix, regularization_matrix, image_to_pix, sub_to_pix)


class AmorphousPixelization(VoronoiPixelization):

    def __init__(self, pixels, regularization_coefficients=(1.0, 1.0, 2.0)):
        """
        An amorphous pixelization, which represents pixels as a set of centers where all of the \
        nearest-neighbor pix-grid (i.e. traced image-pixels) are mapped to them.

        For this pixelization, a set of cluster-pixels (defined in the image-plane as a cluster uniform grid of \
        image-pixels) are used to determine a set of pix-plane grid. These grid are then fed into a \
        weighted k-means clustering algorithm, such that the pixel centers adapt to the unlensed pix \
        surface-brightness profile.

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        regularization_coefficients : (float,)
            The regularization coefficients used to smooth the pix reconstruction.
        """
        super(AmorphousPixelization, self).__init__(pixels, regularization_coefficients)

    def inversion_from_pix_grids(self, grids, sparse_mask):
        """
        Compute the mapping matrix of the amorphous pixelization by following these steps:

        1) Extract the cluster-grid (see grids.GridMapperCluster) from the pix-plane.
        2) Performs weighted kmeans clustering on these cluster-grid to compute the pixel centres.
        3) Derive a Voronoi grid using these pixel centres.
        4) Compute the mapping between all image sub-grid and pixels.
        5) Use these mappings to compute the mapping matrix.

        Parameters
        ----------
        grids: mask.CoordinateCollection
            A collection of coordinates for the masked image, subgrid and blurring grid
        sparse_mask: mask.SparseMask
            A mask describing the image pixels that should be used in pixel clustering
        """

        cluster_grid = grids.image[sparse_mask.sparse_to_image]
        pix_centers, cluster_to_pix = self.kmeans_cluster(cluster_grid)
        voronoi = self.voronoi_from_cluster_grid(pix_centers)
        pix_neighbors = self.neighbors_from_pixelization(voronoi.ridge_points)
        image_to_pix = self.image_to_pix_from_pixelization(grids, pix_centers, pix_neighbors, cluster_to_pix,
                                                           sparse_mask)
        sub_to_pix = self.sub_to_pix_from_pixelization(grids, pix_centers, pix_neighbors, cluster_to_pix, sparse_mask)

        mapping_matrix = self.mapping_matrix_from_sub_to_pix(sub_to_pix, grids)
        regularization_matrix = self.constant_regularization_matrix_from_pix_neighbors(pix_neighbors)

        return Inversion(mapping_matrix, regularization_matrix, image_to_pix, sub_to_pix)

    def kmeans_cluster(self, cluster_grid):
        """Perform k-means clustering on the cluster_grid to compute the k-means clusters which represent \
        pixels.

        Parameters
        ----------
        cluster_grid : ndarray
            The x and y cluster-grid which are used to derive the k-means pixelization.
        """
        kmeans = sklearn.cluster.KMeans(self.pixels)
        km = kmeans.fit(cluster_grid)
        return km.cluster_centers_, km.labels_


# TODO : Split into Inversion, InversionBlurred and InversionFitted.

class Inversion(object):

    def __init__(self, mapping, regularization, image_to_pix, sub_to_pix):
        """The matrices and mappings used to linearly invert and fit a data-set.

        Parameters
        -----------
        mapping : ndarray
            The matrix representing the mapping between reconstruction-pixels and weighted_data-pixels.
        regularization : ndarray
            The matrix defining how the reconstruction's pixels are regularized with one another when fitting the
            weighted_data.
        image_to_pix : ndarray
            The mapping between each image-grid pixel and pixelization-grid pixel.
        sub_to_pix : ndarray
            The mapping between each sub-grid pixel and pixelization-grid sub-pixel.
        """

        self.mapping = mapping
        self.regularization = regularization
        self.image_to_pix = image_to_pix
        self.sub_to_pix = sub_to_pix

    def fit_image_via_inversion(self, image, noise, kernel_convolver):
        """Fit the image data using the inversion."""

        # TODO : Do faster / more cleanly

        blurred_mapping = np.zeros(self.mapping.shape)
        for i in range(self.mapping.shape[1]):
            blurred_mapping[:, i] = kernel_convolver.convolve_array(self.mapping[:, i])

        # TODO : Use fast routines once ready.

        covariance = covariance_matrix.compute_covariance_matrix_exact(blurred_mapping, noise)
        weighted_data = covariance_matrix.compute_d_vector_exact(blurred_mapping, image, noise)
        cov_reg = covariance + self.regularization
        reconstruction = np.linalg.solve(cov_reg, weighted_data)

        return InversionFitted(weighted_data, blurred_mapping, self.regularization, covariance, cov_reg, reconstruction)


class InversionFitted(object):

    def __init__(self, weighted_data, blurred_mapping, regularization, covariance, covariance_regularization,
                 reconstruction):
        """The matrices, mappings which have been used to linearly invert and fit a data-set.

        Parameters
        -----------
        weighted_data : ndarray | None
            The 1D vector representing the data, weighted by its noise in a chi squared sense, which is fitted by the \
            inversion (D).
        blurred_mapping : ndarray | None
            The matrix representing the mapping between reconstruction-pixels and data-pixels, including a \
            blurring operation (f).
        regularization : ndarray | None
            The matrix defining how the reconstruction's pixels are regularized with one another (H).
        covariance : ndarray | None
            The covariance between each reconstruction pixel and all other reconstruction pixels (F).
        covariance_regularization : ndarray | None
            The covariance + regularizationo matrix.
        reconstruction : ndarray | None
            The vector containing the reconstructed fit of the data.
        """
        self.weighted_data = weighted_data
        self.blurred_mapping = blurred_mapping
        self.regularization = regularization
        self.covariance = covariance
        self.covariance_regularization = covariance_regularization
        self.reconstruction = reconstruction

    def model_image_from_reconstruction(self):
        """ Map the reconstruction pix s_vector back to the image-plane to compute the pixelization's model-image.
        """
        model_image = np.zeros(self.blurred_mapping.shape[0])
        for i in range(self.blurred_mapping.shape[0]):
            for j in range(len(self.reconstruction)):
                model_image[i] += self.reconstruction[j] * self.blurred_mapping[i, j]

        return model_image

    # TODO : Speed this up using pix_pixel neighbors list to skip sparsity (see regularization matrix calculation)
    def regularization_term_from_reconstruction(self):
        """ Compute the regularization term of a pixelization's Bayesian likelihood function. This represents the sum \
         of the difference in fluxes between every pair of neighboring pixels. This is computed as:

         s_T * H * s = s_vector.T * regularization_matrix * s_vector

         The term is referred to as 'G_l' in Warren & Dye 2003, Nightingale & Dye 2015.

         The above works include the regularization coefficient (lambda) in this calculation. In PyAutoLens, this is  \
         already in the regularization matrix and thus included in the matrix multiplication.
         """
        return np.matmul(self.reconstruction.T, np.matmul(self.regularization, self.reconstruction))

    # TODO : Cholesky decomposition can also use pixel neighbors list to skip sparsity.
    @staticmethod
    def log_determinant_of_matrix_cholesky(matrix):
        """There are two terms in the pixelization's Bayesian likelihood function which require the log determinant of \
        a matrix. These are (Nightingale & Dye 2015, Nightingale, Dye and Massey 2018):

        ln[det(F + H)] = ln[det(cov_reg_matrix)]
        ln[det(H)]     = ln[det(regularization_matrix)]

        The cov_reg_matrix is positive-definite, which means its log_determinant can be computed efficiently \
        (compared to using np.det) by using a Cholesky decomposition first and summing the log of each diagonal term.

        Parameters
        -----------
        matrix : ndarray
            The positive-definite matrix the log determinant is computed for.
        """
        return 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(matrix))))

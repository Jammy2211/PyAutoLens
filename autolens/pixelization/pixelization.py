import numpy as np
import sklearn.cluster
import scipy.spatial

from autolens import exc
from autolens.pixelization import regularization
from autolens.pixelization import reconstruction
import numba

from autolens.imaging import mask


class PixelizationGrid(object):

    def __init__(self, shape=(3,3)):
        self.shape = shape

    def coordinate_grid_within_annulus(self, inner_radius, outer_radius, centre=(0., 0.)):

        x_pixel_scale = 2.0*outer_radius / self.shape[0]
        y_pixel_scale = 2.0*outer_radius / self.shape[1]

        central_pixel = float(self.shape[0] - 1) / 2, float(self.shape[1] - 1) / 2

        pixel_count = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):

                x_arcsec = ((x - central_pixel[0]) * x_pixel_scale) - centre[0]
                y_arcsec = ((y - central_pixel[1]) * y_pixel_scale) - centre[1]
                radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

                if radius_arcsec < outer_radius or radius_arcsec > inner_radius:

                    pixel_count += 1

        coordinates_array = np.zeros((pixel_count, 2))

        pixel_count = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):

                x_arcsec = ((x - central_pixel[0]) * x_pixel_scale) - centre[0]
                y_arcsec = ((y - central_pixel[1]) * y_pixel_scale) - centre[1]
                radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

                if radius_arcsec < outer_radius or radius_arcsec > inner_radius:

                    coordinates_array[pixel_count, 0] = x_arcsec
                    coordinates_array[pixel_count, 1] = y_arcsec

                    pixel_count += 1

        return coordinates_array





class Pixelization(object):

    def __init__(self, pixels=100, regularization_coefficients=(1.0,)):
        """
        Abstract base class for a pixelization, which discretizes a set of image and sub grid grid into \
        pixels. These pixels fit an image using a linear inversion, where a regularization matrix
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
        """
        self.pixels = pixels
        self.regularization_coefficients = regularization_coefficients

    @property
    def subscript(self):
        return 'pix'

    def mapping_matrix_from_sub_to_pix_jit(self, sub_to_pix, grids):
        return self.mapping_matrix_from_sub_to_pix_jitted(sub_to_pix, self.pixels, grids.image, grids.sub.sub_to_image,
                                                          grids.sub.sub_grid_fraction)

    @staticmethod
    @numba.jit(nopython=True)
    def mapping_matrix_from_sub_to_pix_jitted(sub_to_pix, pixels, grid_image, sub_to_image, sub_grid_fraction):

        mapping_matrix = np.zeros((grid_image.shape[0], pixels))

        for sub_index in range(sub_to_image.shape[0]):
            mapping_matrix[sub_to_image[sub_index], sub_to_pix[sub_index]] += sub_grid_fraction

        return mapping_matrix


class Rectangular(Pixelization):

    def __init__(self, shape=(3,3), regularization_coefficients=(1.0,)):
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

        super(Rectangular, self).__init__(shape[0] * shape[1], regularization_coefficients)

        self.shape = shape

    class Geometry(object):

        def __init__(self, x_min, x_max, x_pixel_scale, y_min, y_max, y_pixel_scale):
            """The geometry of a rectangular grid, defining where the grids top-left, top-right, bottom-left and \
            bottom-right corners are in arc seconds. The arc-second size of each rectangular pixel is also computed.

            Parameters
            -----------

            """
            self.x_min = x_min
            self.x_max = x_max
            self.x_pixel_scale = x_pixel_scale
            self.y_min = y_min
            self.y_max = y_max
            self.y_pixel_scale = y_pixel_scale

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
        x_min = np.min(pix_sub_grid[:, 0]) - buffer
        x_max = np.max(pix_sub_grid[:, 0]) + buffer
        y_min = np.min(pix_sub_grid[:, 1]) - buffer
        y_max = np.max(pix_sub_grid[:, 1]) + buffer
        x_pixel_scale = (x_max - x_min) / self.shape[0]
        y_pixel_scale = (y_max - y_min) / self.shape[1]

        return self.Geometry(x_min, x_max, x_pixel_scale, y_min, y_max, y_pixel_scale)

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

            for x in range(1, self.shape[0] - 1):
                for y in range(1, self.shape[1] - 1):
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

    def grid_to_pix_from_grid_jit(self, grid, geometry):
        """Compute the mappings between a set of image pixels (or sub-pixels) and pixels, using the image's
        traced pix-plane grid (or sub-grid) and the uniform rectangular pixelization's geometry.

        Parameters
        ----------
        grid : [[float, float]]
            The x and y pix grid (or sub-coordinates) which are to be matched with their pixels.
        geometry : Geometry
            The rectangular pixel grid's geometry.
        """
        return self.grid_to_pix_from_grid_jitted(grid, geometry.x_min, geometry.x_pixel_scale, geometry.y_min,
                                                 geometry.y_pixel_scale, self.shape[1]).astype(dtype='int')

    @staticmethod
    @numba.jit(nopython=True)
    def grid_to_pix_from_grid_jitted(grid, x_min, x_pixel_scale, y_min, y_pixel_scale, y_shape):

        grid_to_pix = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):

            x_pixel = np.floor((grid[i,0] - x_min) / x_pixel_scale)
            y_pixel = np.floor((grid[i,1] - y_min) / y_pixel_scale)

            grid_to_pix[i] = x_pixel * y_shape + y_pixel

        return grid_to_pix

    # TODO : RectangularRegWeight doesnt need sparse mask, but equivalent functions elsewhere do. Change to *kwrgs?

    def reconstructor_from_pix_grids(self, grids, borders, cluster_mask=None):
        """
        Compute the pixelization matrices of the rectangular pixelization by following these steps:

        1) Setup the rectangular grid geometry, by making its corner appear at the higher / lowest x and y pix sub-
        grid.
        2) Pair image and sub-image pixels to the rectangular grid using their traced grid and its geometry.

        Parameters
        ----------

        """
        relocated_grids = borders.relocated_grids_from_grids(grids)
        geometry = self.geometry_from_pix_sub_grid(relocated_grids.sub)
        pix_neighbors = self.neighbors_from_pixelization()
        image_to_pix = self.grid_to_pix_from_grid_jit(relocated_grids.image, geometry)
        sub_to_pix = self.grid_to_pix_from_grid_jit(relocated_grids.sub, geometry)

        mapping = self.mapping_matrix_from_sub_to_pix_jit(sub_to_pix, grids)
        regularization = self.regularization_matrix_from_pix_neighbors(pix_neighbors)

        return reconstruction.Reconstructor(mapping, regularization, image_to_pix, sub_to_pix)


class RectangularRegConst(Rectangular, regularization.RegularizationConstant):
    
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
        super(RectangularRegConst, self).__init__(shape, regularization_coefficients)

    @property
    def parameter_labels(self):
        return ['x', 'y', r'\lambda']

class RectangularRegWeighted(Rectangular, regularization.RegularizationWeighted):

    def __init__(self, shape=(3, 3), regularization_coefficients=(1.0, 1.0), pix_signal_scale=1.0):
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
        super(RectangularRegWeighted, self).__init__(shape, regularization_coefficients)
        self.pix_signal_scale = pix_signal_scale

    @property
    def parameter_labels(self):
        return ['x', 'y', r'\lambda1', r'\lambda2', 'V']

class Voronoi(Pixelization):

    def __init__(self, pixels=100, regularization_coefficients=(1.0,)):
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

        super(Voronoi, self).__init__(pixels, regularization_coefficients)

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

    def image_to_pix_from_pixelization(self, grids, pix_centers, pix_neighbors, cluster_to_pix, cluster_mask):
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

        cluster_mask
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
            nearest_cluster = cluster_mask.image_to_sparse[image_index]

            image_to_pix[image_index] = self.pair_image_and_pix(pix_coordinate, nearest_cluster,
                                                                pix_centers, pix_neighbors, cluster_to_pix)

        return image_to_pix

    def sub_to_pix_from_pixelization(self, grids, pix_centers, pix_neighbors, cluster_to_pix, cluster_mask):
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

        grids: mask.GridCollection
            A collection of coordinates for the masked image, subgrid and blurring grid
        cluster_mask: mask.SparseMask
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
            nearest_cluster = cluster_mask.image_to_sparse[grids.sub.sub_to_image[sub_index]]

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


class Cluster(Voronoi):

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
        super(Cluster, self).__init__(pixels, regularization_coefficients)

    def reconstructor_from_pix_grids(self, grids, borders, cluster_mask):
        """
        Compute the mapping matrix of the cluster pixelization by following these steps:

        1) Extract the cluster-grid (see grids.GridMapperCluster) from the pix-plane and use these as the \
        pixel centres.
        3) Derive a Voronoi grid using these pixel centres.
        4) Compute the mapping between all image sub-grid and pixels.
        5) Use these mappings to compute the mapping matrix.

        Parameters
        ----------
        grids: mask.GridCollection
            A collection of coordinates for the masked image, subgrid and blurring grid
        cluster_mask: mask.SparseMask
            A mask describing the image pixels that should be used in pixel clustering
        """

        relocated_grids = borders.relocated_grids_from_grids(grids)

        if self.pixels is not len(cluster_mask.sparse_to_image):
            raise exc.PixelizationException('ClusteringPixelization - The input number of pixels in the constructor'
                                            'is not the same as the length of the cluster_to_image mapper')

        pix_centers = grids.image[cluster_mask.sparse_to_image]
        cluster_to_pix = np.arange(0, self.pixels)
        voronoi = self.voronoi_from_cluster_grid(pix_centers)
        pix_neighbors = self.neighbors_from_pixelization(voronoi.ridge_points)
        image_to_pix = self.image_to_pix_from_pixelization(relocated_grids, pix_centers, pix_neighbors, cluster_to_pix,
                                                           cluster_mask)
        sub_to_pix = self.sub_to_pix_from_pixelization(relocated_grids, pix_centers, pix_neighbors, cluster_to_pix,
                                                       cluster_mask)

        mapping_matrix = self.mapping_matrix_from_sub_to_pix_jit(sub_to_pix, relocated_grids)
        regularization_matrix = self.regularization_matrix_from_pix_neighbors(pix_neighbors)

        return reconstruction.Reconstructor(mapping_matrix, regularization_matrix, image_to_pix, sub_to_pix)


class ClusterRegConst(Cluster, regularization.RegularizationConstant):

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
        super(ClusterRegConst, self).__init__(pixels, regularization_coefficients)

    @property
    def parameter_labels(self):
        return ['N', r'\lambda1']

class ClusterRegWeighted(Cluster, regularization.RegularizationWeighted):

    def __init__(self, pixels, regularization_coefficients=(1.0, 1.0), pix_signal_scale=1.0):
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
        super(ClusterRegWeighted, self).__init__(pixels, regularization_coefficients)
        self.pix_signal_scale = pix_signal_scale

    @property
    def parameter_labels(self):
        return ['N', r'\lambda1', r'\lambda2', 'V']

class Amorphous(Voronoi):

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
        super(Amorphous, self).__init__(pixels, regularization_coefficients)

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

    def reconstructor_from_pix_grids(self, grids, borders, cluster_mask):
        """
        Compute the mapping matrix of the amorphous pixelization by following these steps:

        1) Extract the cluster-grid (see grids.GridMapperCluster) from the pix-plane.
        2) Performs weighted kmeans clustering on these cluster-grid to compute the pixel centres.
        3) Derive a Voronoi grid using these pixel centres.
        4) Compute the mapping between all image sub-grid and pixels.
        5) Use these mappings to compute the mapping matrix.

        Parameters
        ----------
        grids: mask.GridCollection
            A collection of coordinates for the masked image, subgrid and blurring grid
        cluster_mask: mask.SparseMask
            A mask describing the image pixels that should be used in pixel clustering
        """

        relocated_grids = borders.relocated_grids_from_grids(grids)

        cluster_grid = grids.image[cluster_mask.sparse_to_image]
        pix_centers, cluster_to_pix = self.kmeans_cluster(cluster_grid)
        voronoi = self.voronoi_from_cluster_grid(pix_centers)
        pix_neighbors = self.neighbors_from_pixelization(voronoi.ridge_points)
        image_to_pix = self.image_to_pix_from_pixelization(relocated_grids, pix_centers, pix_neighbors, cluster_to_pix,
                                                           cluster_mask)
        sub_to_pix = self.sub_to_pix_from_pixelization(relocated_grids, pix_centers, pix_neighbors, cluster_to_pix, cluster_mask)

        mapping_matrix = self.mapping_matrix_from_sub_to_pix_jit(sub_to_pix, relocated_grids)
        regularization_matrix = self.regularization_matrix_from_pix_neighbors(pix_neighbors)

        return reconstruction.Reconstructor(mapping_matrix, regularization_matrix, image_to_pix, sub_to_pix)


class AmorphousRegConst(Amorphous, regularization.RegularizationConstant):

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
        super(AmorphousRegConst, self).__init__(pixels, regularization_coefficients)

    @property
    def parameter_labels(self):
        return ['N', r'\lambda1']

class AmorphousRegWeighted(Amorphous, regularization.RegularizationWeighted):

    def __init__(self, pixels, regularization_coefficients=(1.0, 1.0), pix_signal_scale=1.0):
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
        super(AmorphousRegWeighted, self).__init__(pixels, regularization_coefficients)
        self.pix_signal_scale = pix_signal_scale

    @property
    def parameter_labels(self):
        return ['N', r'\lambda1', r'\lambda2', 'V']
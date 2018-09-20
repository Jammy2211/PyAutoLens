import numpy as np
import numba


class Mapper(object):

    def __init__(self, pixels, grids, pixel_neighbors):
        """
        Abstract base class representing the mapping between the pixels in an observed image of a strong lens and \
        the pixels of a pixelization.

        1D arrays are used to represent these mappings, in particular between the different image-grids (e.g. the \
        image and sub grids, see the *imaging.mask module*). The nomenclature here follows grid_to_grid, whereby the \
        index of a value on one grid equals another. For example:

        - image_to_pixelization[2] = 1 tells us that the 3rd image-pixel maps to the 2nd pixelization-pixel.
        - sub_to_pixelization[4] = 2 tells us that the 5th sub-pixel maps to the 3rd pixelization-pixel.
        - pixelization_to_image[2] = 5 tells us that the 3rd pixelization-pixel maps to the 6th (masked) image-pixel.

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        grids: mask.ImagingGrids
            A collection of grid describing the observed image's pixel coordinates (includes an image and sub grid).
        pixel_neighbors : [[]]
            A list of the neighbors of each pixel.
        """
        self.pixels = pixels
        self.grids = grids
        self.pixel_neighbors = pixel_neighbors

    @property
    def mapping_matrix(self):
        """The mapping matrix is a matrix representing the mapping between every observed image-pixel and \
        pixelization pixel. Non-zero entries signify a mapping, whereas zeros signify no mapping.

        For example, if the observed image has 5 pixels and the pixelization 3 pixels, with the following mappings:

        image pixel 0 -> pixelization pixel 0
        image pixel 1 -> pixelization pixel 0
        image pixel 2 -> pixelization pixel 1
        image pixel 3 -> pixelization pixel 1
        image pixel 4 -> pixelization pixel 2

        The mapping matrix (which is of dimensions image_pixels x source_pixels) would appear as follows:

        [1, 0, 0] [0->0]
        [1, 0, 0] [1->0]
        [0, 1, 0] [2->1]
        [0, 1, 0] [3->1]
        [0, 0, 1] [4->2]

        The mapping matrix is in fact built using the sub-grid of the observed image, where each image-pixel is \
        divided into a regular grid of sub-pixels each of which are paired to pixels in the pixelization. The entires \
        in the mapping matrix now become fractional values dependent on the sub-grid size. For example, for a 2x2 \
        sub-grid in each pixel (which means the fraction value is 1.0/(2.0^2) = 0.25, if we have the following mappings:

        image pixel 0 -> sub pixel 0 -> pixelization pixel 0
        image pixel 0 -> sub pixel 1 -> pixelization pixel 1
        image pixel 0 -> sub pixel 2 -> pixelization pixel 1
        image pixel 0 -> sub pixel 3 -> pixelization pixel 1
        image pixel 1 -> sub pixel 0 -> pixelization pixel 1
        image pixel 1 -> sub pixel 1 -> pixelization pixel 1
        image pixel 1 -> sub pixel 2 -> pixelization pixel 1
        image pixel 1 -> sub pixel 3 -> pixelization pixel 1
        image pixel 2 -> sub pixel 0 -> pixelization pixel 2
        image pixel 2 -> sub pixel 1 -> pixelization pixel 2
        image pixel 2 -> sub pixel 2 -> pixelization pixel 3
        image pixel 2 -> sub pixel 3 -> pixelization pixel 3

        The mapping matrix (which is still of dimensions image_pixels x source_pixels) would appear as follows:

        [0.25, 0.75, 0.0, 0.0] [1 sub-pixel maps to pixel 0, 3 mappers to pixel 1]
        [ 0.0,  1.0, 0.0, 0.0] [All sub-pixels mappers to pixel 1]
        [ 0.0,  0.0, 0.5, 0.5] [2 sub-pixels mappers to pixel 2, 2 mappers to pixel 3]
        """
        return self.mapping_matrix_from_sub_to_pix_jit(self.sub_to_pixelization, self.pixels, self.grids.image.shape[0],
                                                       self.grids.sub.sub_to_image, self.grids.sub.sub_grid_fraction)

    @staticmethod
    @numba.jit(nopython=True)
    def mapping_matrix_from_sub_to_pix_jit(sub_to_pixelization, pixels, image_pixels, sub_to_image, sub_grid_fraction):
        """Computes the mapping matrix, by iterating over the known mappings between the sub-grid and pixelization.

        Parameters
        -----------
        sub_to_pixelization : ndarray
            The mappings between the observed image's sub-pixels and pixelization's pixels.
        pixels : int
            The number of pixels in the pixelization.
        image_pixels : int
            The number of image pixels in the observed image.
        sub_to_image : ndarray
            The mappings between the observed image's sub-pixels and observed image's pixels.
        sub_grid_fraction : float
            The fractional area each sub-pixel takes up in an image-pixel.
        """
        mapping_matrix = np.zeros((image_pixels, pixels))

        for sub_index in range(sub_to_image.shape[0]):
            mapping_matrix[sub_to_image[sub_index], sub_to_pixelization[sub_index]] += sub_grid_fraction

        return mapping_matrix

    @property
    def sub_to_pixelization(self):
        raise NotImplementedError("sub_to_pixelization should be overridden")


class RectangularMapper(Mapper):

    def __init__(self, pixels, grids, pixel_neighbors, shape, geometry):
        """Class representing the mappings between the pixels in an observed image of a strong lens and \
        the pixels of a rectangular pixelization.

        The regular and uniform geometry of the rectangular grid is used to perform efficient pixel pairings.

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        grids: mask.ImagingGrids
            A collection of grid describing the observed image's pixel coordinates (includes an image and sub grid).
        pixel_neighbors : [[]]
            A list of the neighbors of each pixel.
        shape : (int, int)
            The dimensions of the rectangular grid of pixels (x_pixels, y_pixel)
        geometry : pixelization.Rectangular.Geometry
            The geometry (e.g. x / y edge locations, pixel-scales) of the rectangular pixelization.
        """
        self.shape = shape
        self.geometry = geometry
        super(RectangularMapper, self).__init__(pixels, grids, pixel_neighbors)

    @property
    def image_to_pixelization(self):
        """The mappings between a set of image pixels and pixelization pixels."""
        return self.grid_to_pixelization_from_grid_jit(self.grids.image,
                                                       self.geometry.x_min, self.geometry.x_pixel_scale,
                                                       self.geometry.y_min, self.geometry.y_pixel_scale,
                                                       self.shape[1]).astype(dtype='int')

    @property
    def sub_to_pixelization(self):
        """The mappings between a set of sub-pixels and pixelization pixels"""
        return self.grid_to_pixelization_from_grid_jit(self.grids.sub,
                                                       self.geometry.x_min, self.geometry.x_pixel_scale,
                                                       self.geometry.y_min, self.geometry.y_pixel_scale,
                                                       self.shape[1]).astype(dtype='int')

    @staticmethod
    @numba.jit(nopython=True)
    def grid_to_pixelization_from_grid_jit(grid, x_min, x_pixel_scale, y_min, y_pixel_scale, y_shape):

        grid_to_pixelization = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            x_pixel = np.floor((grid[i, 0] - x_min) / x_pixel_scale)
            y_pixel = np.floor((grid[i, 1] - y_min) / y_pixel_scale)

            grid_to_pixelization[i] = x_pixel * y_shape + y_pixel

        return grid_to_pixelization


class VoronoiMapper(Mapper):

    def __init__(self, pixels, grids, pixel_neighbors, pixel_centers, voronoi, voronoi_to_pixelization,
                 image_to_voronoi):
        """Class representing the mappings between the pixels in an observed image of a strong lens and \
        the pixels of a Voronoi pixelization.

        The irregular and non-uniform geometry of the Voronoi grid means efficient pixel pairings requires knowledge \
        of how many different grids mappers to one another (see *pair_image_and_pixel*).

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        grids: mask.ImagingGrids
            A collection of grid describing the observed image's pixel coordinates (includes an image and sub grid).
        pixel_neighbors : [[]]
            A list of the neighbors of each pixel.
        shape : (int, int)
            The dimensions of the rectangular grid of pixels (x_pixels, y_pixel)
        geometry : pixelization.Rectangular.Geometry
            The geometry (e.g. x / y edge locations, pixel-scales) of the rectangular pixelization.
        """
        self.pixel_centers = pixel_centers
        self.voronoi = voronoi
        self.voronoi_to_pixelization = voronoi_to_pixelization
        self.image_to_voronoi = image_to_voronoi
        super(VoronoiMapper, self).__init__(pixels, grids, pixel_neighbors)

    @property
    def image_to_pixelization(self):
        """The mappings between a set of image pixels and pixelization pixels."""

        image_to_pixelization = np.zeros((self.grids.image.shape[0]), dtype=int)

        for image_index, pixel_coordinate in enumerate(self.grids.image):

            nearest_cluster = self.image_to_voronoi[image_index]

            image_to_pixelization[image_index] = self.pair_image_and_pixel(pixel_coordinate, nearest_cluster)

        return image_to_pixelization

    @property
    def sub_to_pixelization(self):
        """ Compute the mappings between a set of sub-masked_image pixels and pixels, using the masked_image's traced \
        pix-plane sub-grid and the pixel centers. This uses the pix-neighbors to perform a graph \
        search when pairing pixels, for efficiency.

        For the Voronoi pixelizations, a cluster set of 'cluster-pixels' are used to determine the pixelization. \
        These provide the mappings between only a sub-set of sub-pixels / masked_image-pixels and pixels.

        To determine the complete set of sub-pixel to pixel mappings, we must therefore pair every sub-pixel to \
        its nearest pixel (using the sub-pixel's pix-plane coordinate and pixel center). Using a full \
        nearest neighbor search to do this is slow, thus the pixel neighbors (derived via the Voronoi grid) \
        is used to localize each nearest neighbor search.

        In this routine, some variables and function names refer to a 'cluster_pix_'. This term describes a \
        pixel that we have paired to a sub_coordinate using the cluster_coordinate of an masked_image coordinate. \
        Thus, it may not actually be that sub_coordinate's closest pixel (the routine will eventually
        determine this).

        Parameters
        ----------

        grids: mask.ImagingGrids
            A collection of coordinates for the masked masked_image, subgrid and blurring grid
        cluster_mask: mask.SparseMask
            A mask describing the masked_image pixels that should be used in pixel clustering
        pixel_centers: [[float, float]]
            The coordinate of the center of every pixel.
        pixel_neighbors : [[]]
            The neighboring pix_pixels of each pix_pixel, computed via the Voronoi grid_coords. \
            (e.g. if the fifth pix_pixel neighbors pix_pixels 7, 9 and 44, pixel_neighbors[4] = [6, 8, 43])
        cluster_to_pixelization : [int]
            The mapping_matrix between every pixel and cluster-pixel (e.g. if the fifth pixel maps to \
            the 3rd cluster_pixel, pix_to_cluster[4] = 2).

        Returns
        ----------
        sub_to_pixelization : [int, int]
            The mapping_matrix between every sub-pixel and pixel. (e.g. if the fifth sub-pixel of the third \
            masked_image-pixel maps to the 3rd pixel, sub_to_pixelization[2,4] = 2).

         """

        sub_to_pixelization = np.zeros((self.grids.sub.total_pixels,), dtype=int)

        for sub_index, sub_coordinate in enumerate(self.grids.sub):
            nearest_cluster = self.image_to_voronoi[self.grids.sub.sub_to_image[sub_index]]

            sub_to_pixelization[sub_index] = self.pair_image_and_pixel(sub_coordinate, nearest_cluster)

        return sub_to_pixelization

    def pair_image_and_pixel(self, coordinate, nearest_cluster):
        """ Compute the mappings between a set of sub-masked_image pixels and pixels, using the masked_image's traced \
        pix-plane sub-grid and the pixel centers. This uses the pix-neighbors to perform a graph \
        search when pairing pixels, for efficiency.

        For the Voronoi pixelizations, a cluster set of 'cluster-pixels' are used to determine the pixelization. \
        These provide the mappings between only a sub-set of sub-pixels / masked_image-pixels and pixels.

        To determine the complete set of sub-pixel to pixel mappings, we must therefore pair every sub-pixel to \
        its nearest pixel (using the sub-pixel's pix-plane coordinate and pixel center). Using a full \
        nearest neighbor search to do this is slow, thus the pixel neighbors (derived via the Voronoi grid) \
        is used to localize each nearest neighbor search.

        In this routine, some variables and function names refer to a 'cluster_pix_'. This term describes a \
        pixel that we have paired to a sub_coordinate using the cluster_coordinate of an masked_image coordinate. \
        Thus, it may not actually be that sub_coordinate's closest pixel (the routine will eventually
        determine this).

        Parameters
        ----------
        coordinate : [float, float]
            The x and y pix sub-grid grid which are to be matched with their closest pixels.
        nearest_cluster : int
            The nearest pixel defined on the cluster-pixel grid.
        pixel_centers: [[float, float]]
            The coordinate of the center of every pixel.
        pixel_neighbors : [[]]
            The neighboring pix_pixels of each pix_pixel, computed via the Voronoi grid_coords. \
            (e.g. if the fifth pix_pixel neighbors pix_pixels 7, 9 and 44, pixel_neighbors[4] = [6, 8, 43])
        cluster_to_pixelization : [int]
            The mapping_matrix between every cluster-pixel and pixel (e.g. if the fifth pixel maps to \
            the 3rd cluster_pixel, cluster_to_pix[4] = 2).
         """

        nearest_pixel = self.voronoi_to_pixelization[nearest_cluster]

        while True:

            pixel_to_cluster_distance = self.distance_to_nearest_cluster_pixel(coordinate, nearest_pixel)

            neighboring_pixel_index, sub_to_neighboring_pixel_distance = \
                self.nearest_neighboring_pixel_and_distance(coordinate, self.pixel_neighbors[nearest_pixel])

            if pixel_to_cluster_distance < sub_to_neighboring_pixel_distance:
                return nearest_pixel
            else:
                nearest_pixel = neighboring_pixel_index

    def distance_to_nearest_cluster_pixel(self, coordinate, nearest_pixel):
        nearest_cluster_pixel_center = self.pixel_centers[nearest_pixel]
        return self.compute_squared_separation(coordinate, nearest_cluster_pixel_center)

    def nearest_neighboring_pixel_and_distance(self, coordinate, pixel_neighbors):
        """For a given pix_pixel, we look over all its adjacent neighbors and find the neighbor whose distance is closest to
        our input coordinates.

        Parameters
        ----------
        coordinate : (float, float)
            The x and y coordinate to be matched with the neighboring set of pix_pixels.
        pixel_centers: [(float, float)
            The pix_pixel centers the image_grid are matched with.
        pixel_neighbors : []
            The neighboring pix_pixels of the cluster_grid pix_pixel the coordinate is currently matched with

        Returns
        ----------
        pix_neighbors_index : int
            The index in pix_pixel_centers of the closest pix_pixel neighbor.
        separation_from_neighbor : float
            The separation between the input coordinate and closest pix_pixel neighbor

        """

        separation_from_neighbor = list(map(lambda neighbors:
                                            self.compute_squared_separation(coordinate, self.pixel_centers[neighbors]),
                                            pixel_neighbors))

        closest_separation_index = min(range(len(separation_from_neighbor)),
                                       key=separation_from_neighbor.__getitem__)

        return pixel_neighbors[closest_separation_index], separation_from_neighbor[closest_separation_index]

    @staticmethod
    def compute_squared_separation(coordinate1, coordinate2):
        """Computes the squared separation of two image_grid (no square root for efficiency)"""
        return (coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2
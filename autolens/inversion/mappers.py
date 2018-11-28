import numba
import numpy as np

from autolens.imaging.util import mapping_util
from autolens.imaging import scaled_array

class Mapper(object):

    def __init__(self, pixels, grids, border, pixel_neighbors):
        """
        Abstract base class representing the mapping between the pixels in an observed image of a strong lens and \
        the pixels of a pixelization.

        1D arrays are used to represent these mappings, in particular between the different image-grids (e.g. the \
        image and sub grids, see the *imaging.masks module*). The nomenclature here follows grid_to_grid, whereby the \
        index of a value on one grid equals another. For howtolens:

        - image_to_pixelization[2] = 1 tells us that the 3rd image-pixel maps to the 2nd pixelization-pixel.
        - sub_to_pixelization[4] = 2 tells us that the 5th sub-pixel maps to the 3rd pixelization-pixel.
        - pixelization_to_image[2] = 5 tells us that the 3rd pixelization-pixel maps to the 6th (masked) image-pixel.

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        grids: masks.ImagingGrids
            A collection of grid describing the observed image's pixel coordinates (includes an image and sub grid).
        pixel_neighbors : [[]]
            A list of the neighbors of each pixel.
        """
        self.pixels = pixels
        self.grids = grids
        self.border = border
        self.pixel_neighbors = pixel_neighbors

    @property
    def mapping_matrix(self):
        """The mapping matrix is a matrix representing the mapping between every observed image-pixel and \
        pixelization pixel. Non-zero entries signify a mapping, whereas zeros signify no mapping.

        For howtolens, if the observed image has 5 pixels and the pixelization 3 pixels, with the following mappings:

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
        in the mapping matrix now become fractional values dependent on the sub-grid size. For howtolens, for a 2x2 \
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
        return self.mapping_matrix_from_sub_to_pix_jit(self.sub_to_pix, self.pixels, self.grids.image.shape[0],
                                                       self.grids.sub.sub_to_image, self.grids.sub.sub_grid_fraction)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def mapping_matrix_from_sub_to_pix_jit(sub_to_pix, pixels, image_pixels, sub_to_image, sub_grid_fraction):
        """Computes the mapping matrix, by iterating over the known mappings between the sub-grid and pixelization.

        Parameters
        -----------
        sub_to_pix : ndarray
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

            mapping_matrix[sub_to_image[sub_index], sub_to_pix[sub_index]] += sub_grid_fraction

        return mapping_matrix

    @property
    def image_to_pix(self):
        raise NotImplementedError("image_to_pixelization should be overridden")

    @property
    def sub_to_pix(self):
        raise NotImplementedError("sub_to_pixelization should be overridden")

    @property
    def pix_to_image(self):

        pix_to_image = [[] for _ in range(self.pixels)]

        for image_pixel, pix_pixel in enumerate(self.image_to_pix):

            pix_to_image[pix_pixel].append(image_pixel)

        return pix_to_image

    @property
    def pix_to_sub(self):

        pix_to_sub = [[] for _ in range(self.pixels)]

        for image_pixel, pix_pixel in enumerate(self.sub_to_pix):
            pix_to_sub[pix_pixel].append(image_pixel)

        return pix_to_sub


class RectangularMapper(Mapper):

    def __init__(self, pixels, grids, border, pixel_neighbors, shape, geometry):
        """Class representing the mappings between the pixels in an observed image of a strong lens and \
        the pixels of a rectangular pixelization.

        The regular and uniform geometry of the rectangular grid is used to perform efficient pixel pairings.

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        grids: masks.ImagingGrids
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
        super(RectangularMapper, self).__init__(pixels, grids, border, pixel_neighbors)

    @property
    def image_to_pix(self):
        """The 1D index mappings between the image pixels and rectangular pixelization pixels."""
        return self.geometry.grid_arc_seconds_to_grid_pixel_indexes(grid_arc_seconds=self.grids.image)

    @property
    def sub_to_pix(self):
        """The 1D index mappings between the sub-pixels and rectangular pixelization pixels"""
        return self.geometry.grid_arc_seconds_to_grid_pixel_indexes(grid_arc_seconds=self.grids.sub)

    def reconstructed_pixelization_from_solution_vector(self, solution_vector):
        recon = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=solution_vector,
                                                                                    shape=self.shape)
        return scaled_array.ScaledRectangularPixelArray(array=recon, pixel_scales=self.geometry.pixel_scales,
                                                        origin=self.geometry.origin)


class VoronoiMapper(Mapper):

    def __init__(self, pixels, grids, border, pixel_neighbors, pixel_centers, voronoi, image_to_nearest_image_pix):
        """Class representing the mappings between the pixels in an observed image of a strong lens and \
        the pixels of a Voronoi pixelization.

        The irregular and non-uniform geometry of the Voronoi grid means efficient pixel pairings requires knowledge \
        of how many different grids mappers to one another (see *pairimage_and_pixel*).

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        grids: masks.ImagingGrids
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
        self.image_to_nearest_image_pix = image_to_nearest_image_pix
        super(VoronoiMapper, self).__init__(pixels, grids, border, pixel_neighbors)

    @property
    def image_to_pix(self):
        """The 1D index mappings between the image pixels and Voronoi pixelization pixels."""

        image_to_pix = np.zeros((self.grids.image.shape[0]), dtype=int)

        for image_index, pixel_coordinate in enumerate(self.grids.image):

            nearest_pixel = self.image_to_nearest_image_pix[image_index]

            image_to_pix[image_index] = self.pair_image_and_pixel(pixel_coordinate, nearest_pixel)

        return image_to_pix

    @property
    def sub_to_pix(self):
        """  The 1D index mappings between the sub pixels and Voronoi pixelization pixels.
        
        To compute these mappings, a set of sub-maskedimage pixels and pixels, using the maskedimage's traced \
        pix-plane sub-grid and the pixel centers. This uses the pix-neighbors to perform a graph \
        search when pairing pixels, for efficiency.

        For the Voronoi pixelizations, a cluster set of 'cluster-pixels' are used to determine the pixelization. \
        These provide the mappings between only a sub-set of sub-pixels / maskedimage-pixels and pixels.

        To determine the complete set of sub-pixel to pixel mappings, we must therefore pair every sub-pixel to \
        its nearest pixel (using the sub-pixel's pix-plane coordinate and pixel center). Using a full \
        nearest neighbor search to do this is slow, thus the pixel neighbors (derived via the Voronoi grid) \
        is used to localize each nearest neighbor search.

        In this routine, some variables and function names refer to a 'cluster_pix_'. This term describes a \
        pixel that we have paired to a sub_coordinate using the cluster_coordinate of an maskedimage coordinate. \
        Thus, it may not actually be that sub_coordinate's closest pixel (the routine will eventually
        determine this).

        Parameters
        ----------

        grids: masks.ImagingGrids
            A collection of coordinates for the masked maskedimage, subgrid and blurring grid
        cluster_mask: masks.SparseMask
            A masks describing the maskedimage pixels that should be used in pixel clustering
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
            maskedimage-pixel maps to the 3rd pixel, sub_to_pixelization[2,4] = 2).

         """

        sub_to_pix = np.zeros((self.grids.sub.total_pixels,), dtype=int)

        for sub_index, sub_coordinate in enumerate(self.grids.sub):

            nearest_pixel = self.image_to_nearest_image_pix[self.grids.sub.sub_to_image[sub_index]]

            sub_to_pix[sub_index] = self.pair_image_and_pixel(sub_coordinate, nearest_pixel)

        return sub_to_pix

    def pair_image_and_pixel(self, coordinate, nearest_pixel):
        """ Compute the mappings between a set of sub-maskedimage pixels and pixels, using the maskedimage's traced \
        pix-plane sub-grid and the pixel centers. This uses the pix-neighbors to perform a graph \
        search when pairing pixels, for efficiency.

        For the Voronoi pixelizations, a cluster set of 'cluster-pixels' are used to determine the pixelization. \
        These provide the mappings between only a sub-set of sub-pixels / maskedimage-pixels and pixels.

        To determine the complete set of sub-pixel to pixel mappings, we must therefore pair every sub-pixel to \
        its nearest pixel (using the sub-pixel's pix-plane coordinate and pixel center). Using a full \
        nearest neighbor search to do this is slow, thus the pixel neighbors (derived via the Voronoi grid) \
        is used to localize each nearest neighbor search.

        In this routine, some variables and function names refer to a 'cluster_pix_'. This term describes a \
        pixel that we have paired to a sub_coordinate using the cluster_coordinate of an maskedimage coordinate. \
        Thus, it may not actually be that sub_coordinate's closest pixel (the routine will eventually
        determine this).

        Parameters
        ----------
        coordinate : [float, float]
            The x and y pix sub-grid grid which are to be matched with their closest pixels.
        nearest_pixel : int
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

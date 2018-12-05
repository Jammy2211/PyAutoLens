import numba
import numpy as np

from autolens.data.array.util import mapping_util
from autolens.data.array import scaled_array
from autolens.model.inversion.util import mapper_util

class Mapper(object):

    def __init__(self, pixels, grids, border):
        """
        Abstract base class representing the mapping between the pixels in an observed regular of a strong lens and \
        the pixels of a pixelization.

        1D arrays are used to represent these mappings, in particular between the different regular-grids (e.g. the \
        regular and sub grids, see the *imaging.masks module*). The nomenclature here follows grid_to_grid, whereby the \
        index of a value on one grid equals another. For howtolens:

        - image_to_pixelization[2] = 1 tells us that the 3rd regular-pixel maps to the 2nd pixelization-pixel.
        - sub_to_pixelization[4] = 2 tells us that the 5th sub-pixel maps to the 3rd pixelization-pixel.
        - pixelization_to_image[2] = 5 tells us that the 3rd pixelization-pixel maps to the 6th (masked) regular-pixel.

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        grids: masks.DataGrids
            A collection of grid describing the observed regular's pixel coordinates (includes an regular and sub grid).
        """
        self.pixels = pixels
        self.grids = grids
        self.border = border

    @property
    def mapping_matrix(self):
        """The mapping matrix is a matrix representing the mapping between every observed regular-pixel and \
        pixelization pixel. Non-zero entries signify a mapping, whereas zeros signify no mapping.

        For howtolens, if the observed regular has 5 pixels and the pixelization 3 pixels, with the following mappings:

        regular pixel 0 -> pixelization pixel 0
        regular pixel 1 -> pixelization pixel 0
        regular pixel 2 -> pixelization pixel 1
        regular pixel 3 -> pixelization pixel 1
        regular pixel 4 -> pixelization pixel 2

        The mapping matrix (which is of dimensions image_pixels x source_pixels) would appear as follows:

        [1, 0, 0] [0->0]
        [1, 0, 0] [1->0]
        [0, 1, 0] [2->1]
        [0, 1, 0] [3->1]
        [0, 0, 1] [4->2]

        The mapping matrix is in fact built using the sub-grid of the observed regular, where each regular-pixel is \
        divided into a regular grid of sub-pixels each of which are paired to pixels in the pixelization. The entires \
        in the mapping matrix now become fractional values dependent on the sub-grid size. For howtolens, for a 2x2 \
        sub-grid in each pixel (which means the fraction value is 1.0/(2.0^2) = 0.25, if we have the following mappings:

        regular pixel 0 -> sub pixel 0 -> pixelization pixel 0
        regular pixel 0 -> sub pixel 1 -> pixelization pixel 1
        regular pixel 0 -> sub pixel 2 -> pixelization pixel 1
        regular pixel 0 -> sub pixel 3 -> pixelization pixel 1
        regular pixel 1 -> sub pixel 0 -> pixelization pixel 1
        regular pixel 1 -> sub pixel 1 -> pixelization pixel 1
        regular pixel 1 -> sub pixel 2 -> pixelization pixel 1
        regular pixel 1 -> sub pixel 3 -> pixelization pixel 1
        regular pixel 2 -> sub pixel 0 -> pixelization pixel 2
        regular pixel 2 -> sub pixel 1 -> pixelization pixel 2
        regular pixel 2 -> sub pixel 2 -> pixelization pixel 3
        regular pixel 2 -> sub pixel 3 -> pixelization pixel 3

        The mapping matrix (which is still of dimensions image_pixels x source_pixels) would appear as follows:

        [0.25, 0.75, 0.0, 0.0] [1 sub-pixel maps to pixel 0, 3 mappers to pixel 1]
        [ 0.0,  1.0, 0.0, 0.0] [All sub-pixels mappers to pixel 1]
        [ 0.0,  0.0, 0.5, 0.5] [2 sub-pixels mappers to pixel 2, 2 mappers to pixel 3]
        """
        return mapper_util.mapping_matrix_from_sub_to_pix(self.sub_to_pix, self.pixels, self.grids.regular.shape[0],
                                                       self.grids.sub.sub_to_regular, self.grids.sub.sub_grid_fraction)

    @property
    def regular_to_pix(self):
        raise NotImplementedError("regular_to_pix should be overridden")

    @property
    def sub_to_pix(self):
        raise NotImplementedError("sub_to_pix should be overridden")

    @property
    def pix_to_regular(self):

        pix_to_regular = [[] for _ in range(self.pixels)]

        for regular_pixel, pix_pixel in enumerate(self.regular_to_pix):

            pix_to_regular[pix_pixel].append(regular_pixel)

        return pix_to_regular

    @property
    def pix_to_sub(self):

        pix_to_sub = [[] for _ in range(self.pixels)]

        for regular_pixel, pix_pixel in enumerate(self.sub_to_pix):
            pix_to_sub[pix_pixel].append(regular_pixel)

        return pix_to_sub


class RectangularMapper(Mapper):

    def __init__(self, pixels, grids, border, shape, geometry):
        """Class representing the mappings between the pixels in an observed regular of a strong lens and \
        the pixels of a rectangular pixelization.

        The regular and uniform geometry of the rectangular grid is used to perform efficient pixel pairings.

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        grids: masks.DataGrids
            A collection of grid describing the observed regular's pixel coordinates (includes an regular and sub grid).
        shape : (int, int)
            The dimensions of the rectangular grid of pixels (x_pixels, y_pixel)
        geometry : pixelization.Rectangular.Geometry
            The geometry (e.g. x / y edge locations, pixel-scales) of the rectangular pixelization.
        """
        self.shape = shape
        self.geometry = geometry
        super(RectangularMapper, self).__init__(pixels, grids, border)

    @property
    def is_image_plane_pixelization(self):
        return False

    @property
    def regular_to_pix(self):
        """The 1D index mappings between the regular pixels and rectangular pixelization pixels."""
        return self.geometry.grid_arc_seconds_to_grid_pixel_indexes(grid_arc_seconds=self.grids.regular)

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

    def __init__(self, pixels, grids, border, voronoi, geometry):
        """Class representing the mappings between the pixels in an observed regular of a strong lens and \
        the pixels of a Voronoi pixelization.

        The irregular and non-uniform geometry of the Voronoi grid means efficient pixel pairings requires knowledge \
        of how many different grids mappers to one another (see *pairregular_and_pixel*).

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        grids: masks.DataGrids
            A collection of grid describing the observed regular's pixel coordinates (includes an regular and sub grid).
        shape : (int, int)
            The dimensions of the rectangular grid of pixels (x_pixels, y_pixel)
        geometry : pixelization.Rectangular.Geometry
            The geometry (e.g. x / y edge locations, pixel-scales) of the rectangular pixelization.
        """
        self.voronoi = voronoi
        self.geometry = geometry
        super(VoronoiMapper, self).__init__(pixels, grids, border)

    @property
    def is_image_plane_pixelization(self):
        return True

    @property
    def regular_to_pix(self):
        """The 1D index mappings between the regular pixels and Voronoi pixelization pixels."""
        return mapper_util.voronoi_regular_to_pix_from_grids_and_geometry(regular_grid=self.grids.regular,
                                       regular_to_nearest_regular_pix=self.grids.pix.regular_to_nearest_regular_pix,
                                       pixel_centres=self.geometry.pixel_centres,
                                       pixel_neighbors=self.geometry.pixel_neighbors,
                                       pixel_neighbors_size=self.geometry.pixel_neighbors_size).astype('int')

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
         """
        return mapper_util.voronoi_sub_to_pix_from_grids_and_geometry(sub_grid=self.grids.sub,
                                   regular_to_nearest_regular_pix=self.grids.pix.regular_to_nearest_regular_pix,
                                   sub_to_regular=self.grids.sub.sub_to_regular,
                                   pixel_centres=self.geometry.pixel_centres,
                                   pixel_neighbors=self.geometry.pixel_neighbors,
                                   pixel_neighbors_size=self.geometry.pixel_neighbors_size).astype('int')
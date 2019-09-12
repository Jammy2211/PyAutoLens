from autolens.array.mapping_util import array_mapping_util
from autolens.array import scaled_array
from autolens.model.inversion.util import mapper_util

import numpy as np


class Mapper(object):
    def __init__(self, pixels, grid, pixelization_grid, hyper_image=None):
        """ Abstract base class representing a mapper, which maps unmasked pixels on a masked 2D array (in the form of \
        a grid, see the *hyper_galaxies.array.grid* module) to discretized pixels in a pixelization.

        1D arrays are used to represent these mappings, for example between the different grid in a grid \
        (e.g. the / sub grid). This follows the syntax grid_to_grid, whereby the index of a value on one grid \
        equals that of another grid, for example:

        - image_to_pix[2] = 1  tells us that the 3rd pixel on a grid maps to the 2nd pixel of a pixelization.
        - sub_to_pix4] = 2  tells us that the 5th sub-pixel of a sub-grid maps to the 3rd pixel of a pixelization.
        - pix_to_image[2] = 5 tells us that the 3rd pixel of a pixelization maps to the 6th (unmasked) pixel of a \
                            grid.

        Parameters
        ----------
        pixels : int
            The number of pixels in the mapper's pixelization.
        grid: gridStack
            A stack of grid's which are mapped to the pixelization (includes an and sub grid).
        border : grid.GridBorder
            The border of the grid's grid.
        hyper_image : ndarray
            A pre-computed hyper_galaxies-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """
        self.pixels = pixels
        self.grid = grid
        self.pixelization_grid = pixelization_grid
        self.hyper_image = hyper_image

    @property
    def mapping_matrix(self):
        """The mapping_util matrix is a matrix representing the mapping_util between every unmasked pixel of a grid and \
        the pixels of a pixelization. Non-zero entries signify a mapping_util, whereas zeros signify no mapping_util.

        For example, if the grid has 5 pixels and the pixelization 3 pixels, with the following mappings:

        pixel 0 -> pixelization pixel 0
        pixel 1 -> pixelization pixel 0
        pixel 2 -> pixelization pixel 1
        pixel 3 -> pixelization pixel 1
        pixel 4 -> pixelization pixel 2

        The mapping_util matrix (which is of dimensions regular_pixels x pixelization_pixels) would appear as follows:

        [1, 0, 0] [0->0]
        [1, 0, 0] [1->0]
        [0, 1, 0] [2->1]
        [0, 1, 0] [3->1]
        [0, 0, 1] [4->2]

        The mapping_util matrix is in fact built using the sub-grid of the grid, whereby each pixel is \
        divided into a grid of sub-pixels which are all paired to pixels in the pixelization. The entires \
        in the mapping_util matrix now become fractional values dependent on the sub-grid size. For example, for a 2x2 \
        sub-grid in each pixel (which means the fraction value is 1.0/(2.0^2) = 0.25, if we have the following mappings:

        pixel 0 -> sub pixel 0 -> pixelization pixel 0
        pixel 0 -> sub pixel 1 -> pixelization pixel 1
        pixel 0 -> sub pixel 2 -> pixelization pixel 1
        pixel 0 -> sub pixel 3 -> pixelization pixel 1
        pixel 1 -> sub pixel 0 -> pixelization pixel 1
        pixel 1 -> sub pixel 1 -> pixelization pixel 1
        pixel 1 -> sub pixel 2 -> pixelization pixel 1
        pixel 1 -> sub pixel 3 -> pixelization pixel 1
        pixel 2 -> sub pixel 0 -> pixelization pixel 2
        pixel 2 -> sub pixel 1 -> pixelization pixel 2
        pixel 2 -> sub pixel 2 -> pixelization pixel 3
        pixel 2 -> sub pixel 3 -> pixelization pixel 3

        The mapping_util matrix (which is still of dimensions regular_pixels x source_pixels) would appear as follows:

        [0.25, 0.75, 0.0, 0.0] [1 sub-pixel maps to pixel 0, 3 map to pixel 1]
        [ 0.0,  1.0, 0.0, 0.0] [All sub-pixels map to pixel 1]
        [ 0.0,  0.0, 0.5, 0.5] [2 sub-pixels map to pixel 2, 2 map to pixel 3]
        """
        return mapper_util.mapping_matrix_from_sub_mask_1d_index_to_pixelization_1d_index(
            sub_mask_1d_index_to_pixelization_1d_index=self.sub_mask_1d_index_to_pixelization_1d_index,
            pixels=self.pixels,
            total_mask_pixels=self.grid.mask.pixels_in_mask,
            sub_mask_1d_index_to_mask_1d_index=self.grid.mapping.sub_mask_1d_index_to_mask_1d_index,
            sub_fraction=self.grid.mask.sub_fraction,
        )

    @property
    def sub_mask_1d_index_to_pixelization_1d_index(self):
        raise NotImplementedError("sub_to_pixelization should be overridden")

    @property
    def pixelization_1d_index_to_all_sub_mask_1d_indexes(self):
        """Compute the mappings between a pixelization's pixels and the unmasked sub-grid pixels. These mappings \
        are determined after the grid is used to determine the pixelization.

        The pixelization's pixels map to different number of sub-grid pixels, thus a list of lists is used to \
        represent these mappings"""

        pixelization_1d_index_to_all_sub_mask_1d_indexes = [
            [] for _ in range(self.pixels)
        ]

        for mask_1d_index, pixelization_1d_index in enumerate(
            self.sub_mask_1d_index_to_pixelization_1d_index
        ):
            pixelization_1d_index_to_all_sub_mask_1d_indexes[
                pixelization_1d_index
            ].append(mask_1d_index)

        return pixelization_1d_index_to_all_sub_mask_1d_indexes


class RectangularMapper(Mapper):
    def __init__(
        self, pixels, grid, pixelization_grid, shape, geometry, hyper_image=None
    ):
        """ Class representing a rectangular mapper, which maps unmasked pixels on a masked 2D array (in the form of \
        a grid, see the *hyper_galaxies.array.grid* module) to pixels discretized on a rectangular grid.

        The and uniform geometry of the rectangular grid is used to perform efficient pixel pairings.

        Parameters
        ----------
        pixels : int
            The number of pixels in the rectangular pixelization (y_pixels*x_pixels).
        grid : gridStack
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        border : grid.GridBorder
            The border of the grid's grid.
        shape : (int, int)
            The dimensions of the rectangular grid of pixels (y_pixels, x_pixel)
        geometry : pixelization.Rectangular.Geometry
            The geometry (e.g. y / x edge locations, pixel-scales) of the rectangular pixelization.
        """
        self.shape = shape
        self.geometry = geometry
        super(RectangularMapper, self).__init__(
            pixels=pixels,
            grid=grid,
            pixelization_grid=pixelization_grid,
            hyper_image=hyper_image,
        )

    @property
    def is_image_plane_pixelization(self):
        return False

    @property
    def sub_mask_1d_index_to_pixelization_1d_index(self):
        """The 1D index mappings between the sub grid's pixels and rectangular pixelization's pixels"""
        return self.geometry.grid_arcsec_1d_to_grid_pixel_indexes_1d(
            grid_arcsec=self.grid
        )

    def reconstructed_pixelization_from_solution_vector(self, solution_vector):
        """Given the solution vector of an inversion (see *inversions.Inversion*), determine the reconstructed \
        pixelization of the rectangular pixelization by using the mapper."""
        recon = array_mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=solution_vector,
            mask=np.full(fill_value=False, shape=self.shape),
            sub_size=1,
        )
        return scaled_array.ScaledRectangularPixelArray(
            array=recon,
            pixel_scales=self.geometry.pixel_scales,
            origin=self.geometry.origin,
        )


class VoronoiMapper(Mapper):
    def __init__(
        self, pixels, grid, pixelization_grid, voronoi, geometry, hyper_image=None
    ):
        """Class representing a Voronoi mapper, which maps unmasked pixels on a masked 2D array (in the form of \
        a grid, see the *hyper_galaxies.array.grid* module) to pixels discretized on a Voronoi grid.

        The irand non-uniform geometry of the Voronoi grid means efficient pixel pairings requires knowledge \
        of how different grid map to one another.

        Parameters
        ----------
        pixels : int
            The number of pixels in the Voronoi pixelization.
        grid : gridStack
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        border : grid.GridBorder
            The border of the grid's grid.
        voronoi : scipy.spatial.Voronoi
            Class storing the Voronoi grid's geometry.
        geometry : pixelization.Voronoi.Geometry
            The geometry (e.g. y / x edge locations, pixel-scales) of the Vornoi pixelization.
        hyper_image : ndarray
            A pre-computed hyper_galaxies-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """
        self.voronoi = voronoi
        self.geometry = geometry
        super(VoronoiMapper, self).__init__(
            pixels=pixels,
            grid=grid,
            pixelization_grid=pixelization_grid,
            hyper_image=hyper_image,
        )

    @property
    def is_image_plane_pixelization(self):
        return True

    @property
    def sub_mask_1d_index_to_pixelization_1d_index(self):
        """  The 1D index mappings between the sub pixels and Voronoi pixelization pixels. """
        return mapper_util.voronoi_sub_mask_1d_index_to_pixeliztion_1d_index_from_grids_and_geometry(
            grid=self.grid,
            mask_1d_index_to_nearest_pixelization_1d_index=self.pixelization_grid.mask_1d_index_to_nearest_pixelization_1d_index,
            sub_mask_1d_index_to_mask_1d_index=self.grid.mapping.sub_mask_1d_index_to_mask_1d_index,
            pixel_centres=self.geometry.pixel_centres,
            pixel_neighbors=self.geometry.pixel_neighbors,
            pixel_neighbors_size=self.geometry.pixel_neighbors_size,
        ).astype(
            "int"
        )

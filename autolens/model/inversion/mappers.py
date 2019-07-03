from autolens.data.array.util import mapping_util
from autolens.data.array import scaled_array
from autolens.model.inversion.util import mapper_util

class Mapper(object):

    def __init__(self, pixels, grid_stack, border, hyper_image=None):
        """ Abstract base class representing a mapper, which maps unmasked pixels on a masked 2D array (in the form of \
        a grid, see the *hyper.array.grid_stack* module) to discretized pixels in a pixelization.

        1D arrays are used to represent these mappings, for example between the different grid_stack in a grid-stack \
        (e.g. the regular / sub grid_stack). This follows the syntax grid_to_grid, whereby the index of a value on one grid \
        equals that of another grid, for example:

        - image_to_pix[2] = 1  tells us that the 3rd pixel on a regular grid maps to the 2nd pixel of a pixelization.
        - sub_to_pix4] = 2  tells us that the 5th sub-pixel of a sub-grid maps to the 3rd pixel of a pixelization.
        - pix_to_image[2] = 5 tells us that the 3rd pixel of a pixelization maps to the 6th (unmasked) pixel of a \
                            regular grid.

        Parameters
        ----------
        pixels : int
            The number of pixels in the mapper's pixelization.
        grid_stack: grid_stack.GridStack
            A stack of grid's which are mapped to the pixelization (includes an regular and sub grid).
        border : grid_stack.RegularGridBorder
            The border of the grid-stack's regular-grid.
        hyper_image : ndarray
            A pre-computed hyper-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """
        self.pixels = pixels
        self.grid_stack = grid_stack
        self.border = border
        self.hyper_image = hyper_image

    @property
    def mapping_matrix(self):
        """The mapping matrix is a matrix representing the mapping between every unmasked pixel of a grid and \
        the pixels of a pixelization. Non-zero entries signify a mapping, whereas zeros signify no mapping.

        For example, if the regular grid has 5 pixels and the pixelization 3 pixels, with the following mappings:

        regular pixel 0 -> pixelization pixel 0
        regular pixel 1 -> pixelization pixel 0
        regular pixel 2 -> pixelization pixel 1
        regular pixel 3 -> pixelization pixel 1
        regular pixel 4 -> pixelization pixel 2

        The mapping matrix (which is of dimensions regular_pixels x pixelization_pixels) would appear as follows:

        [1, 0, 0] [0->0]
        [1, 0, 0] [1->0]
        [0, 1, 0] [2->1]
        [0, 1, 0] [3->1]
        [0, 0, 1] [4->2]

        The mapping matrix is in fact built using the sub-grid of the grid-stack, whereby each regular-pixel is \
        divided into a regular grid of sub-pixels which are all paired to pixels in the pixelization. The entires \
        in the mapping matrix now become fractional values dependent on the sub-grid size. For example, for a 2x2 \
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

        The mapping matrix (which is still of dimensions regular_pixels x source_pixels) would appear as follows:

        [0.25, 0.75, 0.0, 0.0] [1 sub-pixel maps to pixel 0, 3 map to pixel 1]
        [ 0.0,  1.0, 0.0, 0.0] [All sub-pixels map to pixel 1]
        [ 0.0,  0.0, 0.5, 0.5] [2 sub-pixels map to pixel 2, 2 map to pixel 3]
        """
        return mapper_util.mapping_matrix_from_sub_to_pix(
            sub_to_pix=self.sub_to_pixelization, pixels=self.pixels,
            regular_pixels=self.grid_stack.regular.shape[0],
            sub_to_regular=self.grid_stack.sub.sub_to_regular,
            sub_grid_fraction=self.grid_stack.sub.sub_grid_fraction)

    @property
    def regular_to_pixelization(self):
        raise NotImplementedError("regular_to_pixelization should be overridden")

    @property
    def sub_to_pixelization(self):
        raise NotImplementedError("sub_to_pixelization should be overridden")

    @property
    def pixelization_to_regular_all(self):
        """Compute the mappings between a pixelization's pixels and the unmasked regular-grid pixels. These mappings \
        are determined after the regular-grid is used to determine the pixelization.

        The pixelization's pixels map to different number of regular-grid pixels, thus a list of lists is used to \
        represent these mappings"""

        pixelization_to_regular_all = [[] for _ in range(self.pixels)]

        for regular_pixel, pix_pixel in enumerate(self.regular_to_pixelization):

            pixelization_to_regular_all[pix_pixel].append(regular_pixel)

        return pixelization_to_regular_all

    @property
    def pixelization_to_sub_all(self):
        """Compute the mappings between a pixelization's pixels and the unmasked sub-grid pixels. These mappings \
        are determined after the regular-grid is used to determine the pixelization.

        The pixelization's pixels map to different number of sub-grid pixels, thus a list of lists is used to \
        represent these mappings"""

        pixelization_to_sub_all = [[] for _ in range(self.pixels)]

        for regular_pixel, pix_pixel in enumerate(self.sub_to_pixelization):
            pixelization_to_sub_all[pix_pixel].append(regular_pixel)

        return pixelization_to_sub_all


class RectangularMapper(Mapper):

    def __init__(self, pixels, grid_stack, border, shape, geometry, hyper_image=None):
        """ Class representing a rectangular mapper, which maps unmasked pixels on a masked 2D array (in the form of \
        a grid, see the *hyper.array.grid_stack* module) to pixels discretized on a rectangular grid.

        The regular and uniform geometry of the rectangular grid is used to perform efficient pixel pairings.

        Parameters
        ----------
        pixels : int
            The number of pixels in the rectangular pixelization (y_pixels*x_pixels).
        grid_stack : grid_stack.GridStack
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        border : grid_stack.RegularGridBorder
            The border of the grid-stack's regular-grid.
        shape : (int, int)
            The dimensions of the rectangular grid of pixels (y_pixels, x_pixel)
        geometry : pixelization.Rectangular.Geometry
            The geometry (e.g. y / x edge locations, pixel-scales) of the rectangular pixelization.
        """
        self.shape = shape
        self.geometry = geometry
        super(RectangularMapper, self).__init__(pixels=pixels, grid_stack=grid_stack, border=border,
                                                hyper_image=hyper_image)

    @property
    def is_image_plane_pixelization(self):
        return False

    @property
    def regular_to_pixelization(self):
        """The 1D index mappings between the regular grid's pixels and rectangular pixelization's pixels."""
        return self.geometry.grid_arcsec_to_grid_pixel_indexes(grid_arcsec=self.grid_stack.regular)

    @property
    def sub_to_pixelization(self):
        """The 1D index mappings between the sub grid's pixels and rectangular pixelization's pixels"""
        return self.geometry.grid_arcsec_to_grid_pixel_indexes(grid_arcsec=self.grid_stack.sub)

    def reconstructed_pixelization_from_solution_vector(self, solution_vector):
        """Given the solution vector of an inversion (see *inversions.Inversion*), determine the reconstructed \
        pixelization of the rectangular pixelization by using the mapper."""
        recon = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=solution_vector,
                                                                                       shape=self.shape)
        return scaled_array.ScaledRectangularPixelArray(array=recon, pixel_scales=self.geometry.pixel_scales,
                                                        origin=self.geometry.origin)


class VoronoiMapper(Mapper):

    def __init__(self, pixels, grid_stack, border, voronoi, geometry, hyper_image=None):
        """Class representing a Voronoi mapper, which maps unmasked pixels on a masked 2D array (in the form of \
        a grid, see the *hyper.array.grid_stack* module) to pixels discretized on a Voronoi grid.

        The irregular and non-uniform geometry of the Voronoi grid means efficient pixel pairings requires knowledge \
        of how different grid_stack map to one another.

        Parameters
        ----------
        pixels : int
            The number of pixels in the Voronoi pixelization.
        grid_stack : grid_stack.GridStack
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        border : grid_stack.RegularGridBorder
            The border of the grid-stack's regular-grid.
        voronoi : scipy.spatial.Voronoi
            Class storing the Voronoi grid's geometry.
        geometry : pixelization.Voronoi.Geometry
            The geometry (e.g. y / x edge locations, pixel-scales) of the Vornoi pixelization.
        hyper_image : ndarray
            A pre-computed hyper-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """
        self.voronoi = voronoi
        self.geometry = geometry
        super(VoronoiMapper, self).__init__(pixels=pixels, grid_stack=grid_stack, border=border,
                                            hyper_image=hyper_image)

    @property
    def is_image_plane_pixelization(self):
        return True

    @property
    def regular_to_pixelization(self):
        """The 1D index mappings between the regular pixels and Voronoi pixelization pixels."""
        return mapper_util.voronoi_regular_to_pix_from_grids_and_geometry(regular_grid=self.grid_stack.regular,
               regular_to_nearest_pix=self.grid_stack.pixelization.regular_to_pixelization,
               pixel_centres=self.geometry.pixel_centres, pixel_neighbors=self.geometry.pixel_neighbors,
               pixel_neighbors_size=self.geometry.pixel_neighbors_size).astype('int')

    @property
    def sub_to_pixelization(self):
        """  The 1D index mappings between the sub pixels and Voronoi pixelization pixels. """
        return mapper_util.voronoi_sub_to_pix_from_grids_and_geometry(
            sub_grid=self.grid_stack.sub,
            regular_to_nearest_pix=self.grid_stack.pixelization.regular_to_pixelization,
            sub_to_regular=self.grid_stack.sub.sub_to_regular, pixel_centres=self.geometry.pixel_centres,
            pixel_neighbors=self.geometry.pixel_neighbors,
            pixel_neighbors_size=self.geometry.pixel_neighbors_size).astype('int')
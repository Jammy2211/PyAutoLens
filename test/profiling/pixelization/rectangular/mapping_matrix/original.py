import numba
import numpy as np
from analysis import galaxy
from analysis import ray_tracing
from profiling import profiling_data
from profiling import tools

from autolens import exc
from profiles import mass_profiles


class Pixelization(object):

    def __init__(self, pixels=100, regularization_coefficients=(1.0,)):
        """
        Abstract base class for a inversion, which discretizes a set of masked_image and sub grid grid into \
        pixels. These pixels incorrect_fit an masked_image using a linear inversion, where a regularization_matrix matrix
        enforces smoothness between pixel values.

        A number of 1D and 2D arrays are used to represent mappings betwen masked_image, sub, pix, and cluster pixels. The \
        nomenclature here follows grid_to_grid, such that it maps the index of a value on one grid to another. For \
        howtolens:

        - pix_to_image[2] = 5 tells us that the 3rd inversion-pixel maps to the 6th masked_image-pixel.
        - sub_to_pixelization[4,2] = 2 tells us that the 5th sub-pixel maps to the 3rd inversion-pixel.

        Parameters
        ----------
        pixels : int
            The number of pixels in the inversion.
        regularization_coefficients : (float,)
            The regularization_matrix coefficients used to smooth the pix reconstructed_image.
        """
        self.pixels = pixels
        self.regularization_coefficients = regularization_coefficients

    def mapping_matrix_from_sub_to_pix(self, sub_to_pix, grids):
        """
        Create a new mapping_matrix matrix, which describes the fractional unit surface brightness counts between each \
        masked_image-pixel and pixel. The mapping_matrix matrix is denoted 'f_ij' in Warren & Dye 2003,
        Nightingale & Dye 2015 and Nightingale, Dye & Massey 2018.

        The matrix has dimensions [image_pixels, pix_pixels] and non-zero entries represents an \
        masked_image-pixel to pixel mapping_matrix. For howtolens, if masked_image-pixel 0 maps to pixel 2, element \
        [0,2] of the mapping_matrix matrix will = 1.

        The mapping_matrix matrix is created using sub-gridding. Here, each observed masked_image-pixel is divided into a finer \
        sub_grid. For howtolens, if the sub-grid is sub_grid_size=4, each masked_image-pixel is split into a uniform 4 x 4 \
        sub grid and all 16 sub-pixels are individually paired with pixels.

        The entries in the mapping_matrix matrix therefore become fractional surface brightness values, representing the \
        number of sub-pixel to pixel mappings. For howtolens if 3 sub-pixels from masked_image-pixel 4 mappers to \
        pixel 2, then element [4,2] of the mapping_matrix matrix will = 3.0 * (1/sub_grid_size**2) = 3/16 = 0.1875.

        Parameters
        ----------
        grids
        sub_to_pix : [int, int]
            The pixel index each masked_image and sub-masked_image pixel is matched with. (e.g. if the fifth
            sub-pixel is matched with the 3rd pixel, sub_to_pixelization[4] = 2).

        """

        mapping_matrix = np.zeros((grids.image_plane_images_.shape[0], self.pixels))

        for sub_index in range(grids.sub.total_pixels):
            mapping_matrix[grids.sub.sub_to_regular[sub_index], sub_to_pix[sub_index]] += grids.sub.sub_grid_fraction

        return mapping_matrix


class Rectangular(Pixelization):

    def __init__(self, shape=(3, 3), regularization_coefficients=(1.0,)):
        """A rectangular inversion where pixels appear on a Cartesian, uniform and rectangular grid \
        of  shape (rows, columns).

        Like an masked_image grid, the indexing of the rectangular grid begins in the top-left corner and goes right and down.

        Parameters
        -----------
        shape : (int, int)
            The dimensions of the rectangular grid of pixels (x_pixels, y_pixel)
        regularization_coefficients : (float,)
            The regularization_matrix coefficients used to smooth the pix reconstructed_image.
        """

        if shape[0] <= 2 or shape[1] <= 2:
            raise exc.PixelizationException('The rectangular inversion must be at least dimensions 3x3')

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

    def grid_to_pix_from_grid(self, grid, geometry):
        """Compute the mappings between a set of masked_image pixels (or sub-pixels) and pixels, using the masked_image's
        traced pix-plane grid (or sub-grid) and the uniform rectangular inversion's geometry.

        Parameters
        ----------
        grid : [[float, float]]
            The x and y pix grid (or sub-coordinates) which are to be matched with their pixels.
        geometry : Geometry
            The rectangular pixel grid's geometry.
        """
        grid_to_pix = np.zeros(grid.shape[0], dtype='int')

        for index, pix_coordinate in enumerate(grid):
            x_pixel = geometry.x_arc_seconds_to_pixels(pix_coordinate[0])
            y_pixel = geometry.y_arc_seconds_to_pixels(pix_coordinate[1])

            grid_to_pix[index] = x_pixel * self.shape[1] + y_pixel

        return grid_to_pix

    def grid_to_pix_from_grid_jitted(self, grid, geometry):
        """Compute the mappings between a set of masked_image pixels (or sub-pixels) and pixels, using the masked_image's
        traced pix-plane grid (or sub-grid) and the uniform rectangular inversion's geometry.

        Parameters
        ----------
        grid : [[float, float]]
            The x and y pix grid (or sub-coordinates) which are to be matched with their pixels.
        geometry : Geometry
            The rectangular pixel grid's geometry.
        """
        return self.grid_to_pix_jit(grid, geometry.x_min, geometry.x_pixel_scale, geometry.y_min,
                                    geometry.y_pixel_scale, self.shape[1]).astype(dtype='int')

    @staticmethod
    @numba.jit(nopython=True)
    def grid_to_pix_jit(grid, x_min, x_pixel_scale, y_min, y_pixel_scale, y_shape):

        grid_to_pix = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            x_pixel = np.floor((grid[i, 0] - x_min) / x_pixel_scale)
            y_pixel = np.floor((grid[i, 1] - y_min) / y_pixel_scale)

            grid_to_pix[i] = x_pixel * y_shape + y_pixel

        return grid_to_pix

    def sub_to_pix_from_pix_grids(self, grids, borders):
        """
        Compute the inversion matrices of the rectangular inversion by following these steps:

        1) Setup the rectangular grid geometry, by making its corner appear at the higher / lowest x and y pix sub-
        grid.
        2) Pair masked_image and sub-masked_image pixels to the rectangular grid using their traced grid and its geometry.

        Parameters
        ----------

        """
        relocated_grids = borders.relocated_grids_from_grids(grids)
        geometry = self.geometry_from_pix_sub_grid(relocated_grids.sub)
        pix_neighbors = self.neighbors_from_pixelization()
        sub_to_pix = self.grid_to_pix_from_grid_jitted(relocated_grids.sub, geometry)
        return sub_to_pix


sub_grid_size = 4

sie = mass_profiles.EllipticalIsothermal(centre=(0.010, 0.032), einstein_radius=1.47, axis_ratio=0.849, phi=73.6)
shear = mass_profiles.ExternalShear(magnitude=0.0663, phi=160.5)

lens_galaxy = galaxy.Galaxy(mass_profile_0=sie, mass_profile_1=shear)

pix = Rectangular(shape=(20, 20))

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=sub_grid_size)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=sub_grid_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=sub_grid_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=sub_grid_size)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, sub_grid_size=sub_grid_size)

lsst_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[], image_plane_grids=lsst.grids)
euclid_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[], image_plane_grids=euclid.grids)
hst_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[], image_plane_grids=hst.grids)
hst_up_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[], image_plane_grids=hst_up.grids)
ao_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[], image_plane_grids=ao.grids)

lsst_sub_to_pix = pix.sub_to_pix_from_pix_grids(grids=lsst_tracer.source_plane.grids, borders=lsst.borders)
euclid_sub_to_pix = pix.sub_to_pix_from_pix_grids(grids=euclid_tracer.source_plane.grids, borders=euclid.borders)
hst_sub_to_pix = pix.sub_to_pix_from_pix_grids(grids=hst_tracer.source_plane.grids, borders=hst.borders)
hst_up_sub_to_pix = pix.sub_to_pix_from_pix_grids(grids=hst_up_tracer.source_plane.grids, borders=hst_up.borders)
ao_sub_to_pix = pix.sub_to_pix_from_pix_grids(grids=ao_tracer.source_plane.grids, borders=ao.borders)


@tools.tick_toc_x1
def lsst_solution():
    pix.mapping_matrix_from_sub_to_pix(sub_to_pix=lsst_sub_to_pix, grids=lsst.grids)


@tools.tick_toc_x1
def euclid_solution():
    pix.mapping_matrix_from_sub_to_pix(sub_to_pix=euclid_sub_to_pix, grids=euclid.grids)


@tools.tick_toc_x1
def hst_solution():
    pix.mapping_matrix_from_sub_to_pix(sub_to_pix=hst_sub_to_pix, grids=hst.grids)


@tools.tick_toc_x1
def hst_up_solution():
    pix.mapping_matrix_from_sub_to_pix(sub_to_pix=hst_up_sub_to_pix, grids=hst_up.grids)


@tools.tick_toc_x1
def ao_solution():
    pix.mapping_matrix_from_sub_to_pix(sub_to_pix=ao_sub_to_pix, grids=ao.grids)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()

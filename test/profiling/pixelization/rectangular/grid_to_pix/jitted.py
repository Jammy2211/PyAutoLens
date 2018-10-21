import numba
import numpy as np
import pytest
from profiling import profiling_data
from profiling import tools

from autolens import exc


class Pixelization(object):

    def __init__(self, pixels, regularization_coefficients=(1.0,), pix_signal_scale=1.0):
        """
        Abstract base class for a inversion, which discretizes a set of masked_image and sub grid grid into \
        pixels. These pixels then incorrect_fit a  data_vector-set using a linear inversion, where their regularization_matrix matrix
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
        pix_signal_scale : float
            A hyper-parameter which scales the signal attributed to each pixel, used for weighted regularization_matrix.
        """
        self.pixels = pixels
        self.regularization_coefficients = regularization_coefficients
        self.pix_signal_scale = pix_signal_scale


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


sub_grid_size = 4

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=sub_grid_size)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=sub_grid_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=sub_grid_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=sub_grid_size)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, sub_grid_size=sub_grid_size)

pix_shape = (50, 50)

pix = Rectangular(pix_shape)

lsst_geometry = pix.geometry_from_pix_sub_grid(pix_sub_grid=lsst.grids.sub)
euclid_geometry = pix.geometry_from_pix_sub_grid(pix_sub_grid=euclid.grids.sub)
hst_geometry = pix.geometry_from_pix_sub_grid(pix_sub_grid=hst.grids.sub)
hst_up_geometry = pix.geometry_from_pix_sub_grid(pix_sub_grid=hst_up.grids.sub)
ao_geometry = pix.geometry_from_pix_sub_grid(pix_sub_grid=ao.grids.sub)

assert pix.grid_to_pix_from_grid(grid=lsst.grids.sub, geometry=lsst_geometry) == \
       pytest.approx(pix.grid_to_pix_from_grid_jitted(grid=lsst.grids.sub, geometry=lsst_geometry))
pix.grid_to_pix_from_grid_jitted(grid=lsst.grids.sub, geometry=lsst_geometry)
pix.grid_to_pix_from_grid_jitted(grid=euclid.grids.sub, geometry=euclid_geometry)
pix.grid_to_pix_from_grid_jitted(grid=hst.grids.sub, geometry=hst_geometry)
pix.grid_to_pix_from_grid_jitted(grid=hst_up.grids.sub, geometry=hst_up_geometry)
pix.grid_to_pix_from_grid_jitted(grid=ao.grids.sub, geometry=ao_geometry)


@tools.tick_toc_x1
def lsst_solution():
    pix.grid_to_pix_from_grid_jitted(grid=lsst.grids.sub, geometry=lsst_geometry)


@tools.tick_toc_x1
def euclid_solution():
    pix.grid_to_pix_from_grid_jitted(grid=euclid.grids.sub, geometry=euclid_geometry)


@tools.tick_toc_x1
def hst_solution():
    pix.grid_to_pix_from_grid_jitted(grid=hst.grids.sub, geometry=hst_geometry)


@tools.tick_toc_x1
def hst_up_solution():
    pix.grid_to_pix_from_grid_jitted(grid=hst_up.grids.sub, geometry=hst_up_geometry)


@tools.tick_toc_x1
def ao_solution():
    pix.grid_to_pix_from_grid_jitted(grid=ao.grids.sub, geometry=ao_geometry)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()

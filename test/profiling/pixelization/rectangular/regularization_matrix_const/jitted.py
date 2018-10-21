import numba
import numpy as np
from profiling import tools

from autolens import exc


class RegularizationConstant(object):
    pixels = None
    regularization_coefficients = None

    def regularization_matrix_from_pix_neighbors(self, pix_neighbors):
        """
        Setup a inversion's constant regularization_matrix matrix (see test_pixelizations.py)

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

    def regularization_matrix_from_pix_neighbors_jitted(self, pix_neighbors):
        pix_neighbors = np.array([1, 1, 1, 1], dtype='int')
        return self.regularization_matrix_from_pix_neighbors_jit(pix_neighbors, self.pixels,
                                                                 self.regularization_coefficients[0])

    @staticmethod
    @numba.jit(nopython=True)
    def regularization_matrix_from_pix_neighbors_jit(pix_neighbors, pixels, regularization_coefficient):

        regularization_matrix = np.zeros(shape=(pixels, pixels))

        reg_coeff = regularization_coefficient ** 2.0

        for i in range(pixels):
            regularization_matrix[i, i] += 1e-8
            for j in range(pix_neighbors.shape[0]):
                regularization_matrix[i, i] += reg_coeff
                regularization_matrix[i, j] -= reg_coeff

        return regularization_matrix


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


class RectangularRegConst(Pixelization, RegularizationConstant):

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

        super(RectangularRegConst, self).__init__(shape[0] * shape[1], regularization_coefficients)

        self.shape = shape

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


sub_grid_size = 4

pix = RectangularRegConst(shape=(20, 20))

pix_neighbors = pix.neighbors_from_pixelization()

# assert pix.regularization_matrix_from_pix_neighbors(pix_neighbors) == \
#        pytest.approx(pix.regularization_matrix_from_pix_neighbors_jitted(pix_neighbors), 1e-2)

pix.regularization_matrix_from_pix_neighbors_jitted(pix_neighbors)


@tools.tick_toc_x1
def solution():
    pix.regularization_matrix_from_pix_neighbors_jitted(pix_neighbors)


if __name__ == "__main__":
    solution()

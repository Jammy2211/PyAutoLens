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


@tools.tick_toc_x10
def solution_10x10():
    pix = Rectangular(shape=(10, 10))
    pix.neighbors_from_pixelization()


@tools.tick_toc_x10
def solution_20x20():
    pix = Rectangular(shape=(20, 20))
    pix.neighbors_from_pixelization()


@tools.tick_toc_x10
def solution_30x30():
    pix = Rectangular(shape=(30, 30))
    pix.neighbors_from_pixelization()


@tools.tick_toc_x10
def solution_40x40():
    pix = Rectangular(shape=(40, 40))
    pix.neighbors_from_pixelization()


@tools.tick_toc_x10
def solution_50x50():
    pix = Rectangular(shape=(50, 50))
    pix.neighbors_from_pixelization()


@tools.tick_toc_x10
def solution_100x100():
    pix = Rectangular(shape=(100, 100))
    pix.neighbors_from_pixelization()


if __name__ == "__main__":
    solution_10x10()
    solution_20x20()
    solution_30x30()
    solution_40x40()
    solution_50x50()
    solution_100x100()

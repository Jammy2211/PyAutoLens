import numpy as np

class AbstractRegularization(object):
    """ Regularization is applied to a pixelization to enforce a smooth-source solution and prevent over-fitting \
    noise_map in the image. This is achieved by computing a 'regularization term' - which is the sum of differences in  \
    reconstructed flux between every set of neighboring source-pixels. This regularization term is added to the \
    solution's chi-squared as a penalty term, effecting a _reconstruction in the following ways:

    1) The regularization matrix (see below) is added to the curvature matrix used by the reconstructor to linearly \
    invert and fit the image data. Thus, it changes the source-_reconstruction in a fully linearly way, ensuring that \
    the minimum chi-squared solution is achieved accounting for the penalty term.

    2) The likelihood of the _reconstruction changes from L = -0.5 *(chi^2 + _noise_term) to L = -0.5 (chi^2 + \
    regularization_coefficient * regularization_term + _noise_term). The regularization coefficient is a \
    hyper-parameter which determines how strongly we smooth the _reconstruction.

    AutoLens sets the value of the regularization_coefficient(s) using the Bayesian framework of (Suyu 2006) and this \
    is detailed further in the (*_reconstruction.Reconstruction* class)

    The regularization matrix, H, is calculated by defining a set of B matrices which describe how the pixelization \
    pixels neighbor one another. For example, lets take a 3x3 square grid

    ______
    |0|1|2|
    |3|4|5|
    |6|7|8|
    ^^^^^^^

    and we want to regularize this grid such that each pixel is regularized with the pixel to its right and below it \
    (provided there are pixels in that direction). This means that:

    - pixel 0 is regularized with pixel 1 (to the right) and pixel 3 (below).
    - pixel 1 is regularized with pixel 2 (to the right) and pixel 4 (below),
    - However, pixel 2 is only regularized with pixel 5, as there is no pixel to its right.
    - and so on.

    We make two 9 x 9 B matrices, which describe regularization in each direction (i.e. rightwards and downwards). \
    We simply put a -1 and 1 in each row of a pixel index where it has a neighbor, where the value 1 goes in the \
    column of its neighbor's index. Thus, the B matrix describing neighboring pixels to their right looks like:

    B_x = [-1,  1,  0,  0,  0,  0,  0,  0,  0] # [0->1]
          [ 0, -1,  1,  0,  0,  0,  0,  0,  0] # [1->2]
          [ 0,  0, -1,  0,  0,  0,  0,  0,  0] # [] NOTE - no pixel neighbor.
          [ 0,  0,  0, -1,  1,  0,  0,  0,  0] # [3->4]
          [ 0,  0,  0,  0, -1,  1,  0,  0,  0] # [4->5]
          [ 0,  0,  0,  0,  0, -1,  0,  0,  0] # [] NOTE - no pixel neighbor.
          [ 0,  0,  0,  0,  0,  0, -1,  1,  0] # [6->7]
          [ 0,  0,  0,  0,  0,  0,  0, -1,  1] # [7->8]
          [ 0,  0,  0,  0,  0,  0,  0,  0, -1] # [] NOTE - no pixel neighbor.

    We now make another B matrix for the regularization downwards:

    B_y = [-1,  0,  0,  1,  0,  0,  0,  0,  0] # [0->3]
          [ 0, -1,  0,  0,  1,  0,  0,  0,  0] # [1->4]
          [ 0,  0, -1,  0,  0,  1,  0,  0,  0] # [2->5]
          [ 0,  0,  0, -1,  0,  0,  1,  0,  0] # [3->6]
          [ 0,  0,  0,  0, -1,  0,  0,  1,  0] # [4->7]
          [ 0,  0,  0,  0,  0, -1,  0,  0,  1] # [5->8]
          [ 0,  0,  0,  0,  0,  0, -1,  0,  0] # [] NOTE - no pixel neighbor.
          [ 0,  0,  0,  0,  0,  0,  0, -1,  0] # [] NOTE - no pixel neighbor.
          [ 0,  0,  0,  0,  0,  0,  0,  0, -1] # [] NOTE - no pixel neighbor.

    After making the B matrices that represent our pixel neighbors, we can compute the regularization matrix, H, \
    of each direction as H = B * B.T (matrix multiplication).

    E.g.

    H_x = B_x.T, * B_x
    H_y = B_y.T * B_y
    H = H_x + H_y

    Whilst the example above used a square-grid with regularization to the right and downwards, the matrix formalism \
    can be extended to describe regularization in more directions (e.g. upwards, to the left).

    It can also describe irregular grids, for example a Voronoi grid, where a B matrix is computed for every shared \
    Voronoi vertex of each Voronoi pixel. This would mean that number of B matrices is equal to the number of \
    Voronoi vertices in the pixel with the most Voronoi vertices (i.e. the most neighbours a pix-pixel has). However,
    we describe below a scheme to compute this solution more efficiently.

    ### COMBINING B MATRICES ###

    The B matrices above each had the -1's going down the diagonal. This is not necessary, and it is valid to put \
    each pixel pairing anywhere. So, if we had a 4x4 B matrix, where:

    - pixel 0 regularizes with pixel 1
    - pixel 2 regularizes with pixel 3
    - pixel 3 regularizes with pixel 0

    We can still set this up as one matrix (even though the pixel 0 comes up twice):

    B = [-1, 1, 0 ,0] # [0->1]
        [0, 0, 0 ,0] # We can skip rows by making them all zeros.
        [0, 0, -1 ,1] # [2->3]
        [1, 0, 0 ,-1] # [3->0] This is valid!

    So, for a Voronoi pixelzation, we don't have to make the same number of B matrices as Voronoi vertices,  \
    we can combine them into fewer B matrices as above.

    # SKIPPING THE B MATRIX CALCULATION #

    Infact, going through the rigmarole of computing and multiplying B matrices is completely uncessary, and it is \
    more computationally efficiently to directly compute H. This is possible, provided you know know all of the \
    neighboring pixel pairs (which, by definition, you need to know to set up the B matrices anyway). Thus, the \
   'regularization_matrix_from_pix_neighbors' functions in this module directly compute H from the pixel neighbors.

    # POSITIVE DEFINITE MATRIX #

    The regularization matrix must be positive-definite, as the Bayesian framework of Suyu 2006 requires that we \
    use its determinant in the calculation. To ensure this criteria is met, two regularization schemes are applied, \
    constant-regularization and weighted-regularization, which are discussed below.

    # CONSTANT REGULARIZATION #

    For the constant regularization_matrix scheme, there is only 1 regularization coefficient that is applied to all \
    neighboring pixels. This means that we when write B, we only need to regularize pixels in one direction \
    (e.g. pixel 0 regularizes pixel 1, but NOT visa versa). For example:

    B = [-1, 1]  [0->1]
        [0, -1]  1 does not regularization with 0

    # WEIGHTED REGULARIZATION #

    For the weighted regularization scheme, each pixel is given an 'effective regularization weight', which is applied \
    when each set of pixel neighbors are regularized with one another. The motivation of this is that different \
    regions of a pixelization want different levels of regularization (e.g., high smoothing where the source is not \
    located, less smoothing where it is, see (Nightingale, Dye and Massey 2018)).

    Unlike the constant regularization_matrix scheme, neighboring pixels must now be regularized with one another \
    in both directions (e.g. if pixel 0 regularizes pixel 1, pixel 1 must also regularize pixel 0). For example: \

    B = [-1, 1]  [0->1]
        [-1, -1]  1 now also regularizes 0

    For a constant regularization coefficient this would NOT produce a positive-definite matrix. However, for
    the weighted scheme, it does!

    The regularize weights change the B matrix as shown below - we simply multiply each pixel's effective \
    regularization weight by each row of B it has a -1 in, so:

    regularization_weights = [1, 2, 3, 4]

    B = [-1, 1, 0 ,0] # [0->1]
        [0, -2, 2 ,0] # [1->2]
        [0, 0, -3 ,3] # [2->3]
        [4, 0, 0 ,-4] # [3->0]

    If our -1's werent down the diagonal this would look like:

    B = [4, 0, 0 ,-4] # [3->0]
        [0, -2, 2 ,0] # [1->2]
        [-1, 1, 0 ,0] # [0->1]
        [0, 0, -3 ,3] # [2->3] This is valid!
    """


class RegularizationConstant(AbstractRegularization):

    pixels = None
    regularization_coefficients = None

    def regularization_matrix_from_pixel_neighbors(self, pixel_neighbors):
        """
        Setup a pixelization's constant regularization_matrix matrix (see test_pixelization.py)

        Parameters
        ----------
        pixel_neighbors : [[]]
            A list of the neighbors of each pixel.
        """

        regularization_matrix = np.zeros(shape=(self.pixels, self.pixels))

        regularization_coefficient = self.regularization_coefficients[0] ** 2.0

        for i in range(self.pixels):
            regularization_matrix[i, i] += 1e-8
            for j in pixel_neighbors[i]:
                regularization_matrix[i, i] += regularization_coefficient
                regularization_matrix[i, j] -= regularization_coefficient

        return regularization_matrix


class RegularizationWeighted(AbstractRegularization):

    pixels = None
    regularization_coefficients = None
    signal_scale = None

    def pix_signals_from_images(self, image_to_pix, galaxy_image):
        """Compute the (scaled) signal in each pixel, where the signal is the sum of its masked_image-pixel fluxes. \
        These pix-signals are then used to compute the effective regularization_matrix weight of each pixel.

        The pix signals are scaled in the following ways:

        1) Divided by the number of masked_image-pixels in the pixel, to ensure all pixels have the same \
        'relative' signal (i.e. a pixel with 10 images-pixels doesn't have x2 the signal of one with 5).

        2) Divided by the maximum pix-signal, so that all signals vary between 0 and 1. This ensures that the \
        regularizations weights they're used to compute are defined identically for all masked_image units / SNR's.

        3) Raised to the power of the hyper-parameter *signal_scale*, so the method can control the relative \
        contribution of the different regions of regularization_matrix.
        """

        pixel_signals = np.zeros((self.pixels,))
        pixel_sizes = np.zeros((self.pixels,))

        for image_index in range(galaxy_image.shape[0]):
            pixel_signals[image_to_pix[image_index]] += galaxy_image[image_index]
            pixel_sizes[image_to_pix[image_index]] += 1

        pixel_signals /= pixel_sizes
        pixel_signals /= max(pixel_signals)

        return pixel_signals ** self.signal_scale

    def regularization_weights_from_pixel_signals(self, pixel_signals):
        """Compute the regularization_matrix weights, which represent the effective regularization_matrix coefficient of every \
        pixel. These are computed using the (scaled) pix-signal in each pixel.

        Two regularization_matrix coefficients are used which map to:

        1) pixel_signals - This regularizes pix-plane pixels with a high pix-signal (i.e. where the pix is).
        2) 1.0 - pixel_signals - This regularizes pix-plane pixels with a low pix-signal (i.e. background sky)
        """
        return (self.regularization_coefficients[0] * pixel_signals +
                self.regularization_coefficients[1] * (1.0 - pixel_signals)) ** 2.0

    def regularization_matrix_from_pixel_neighbors(self, regularization_weights, pixel_neighbors):
        """
        Setup a weighted regularization_matrix matrix, where all pixels are regularized with one another in both \
        directions using a different effective regularization_matrix coefficient.

        Parameters
        ----------
        regularization_weights : list(float)
            The regularization_matrix weight of each pixel
        pixel_neighbors : [[]]
            A list of the neighbors of each pixel.
        """

        regularization_matrix = np.zeros(shape=(self.pixels, self.pixels))

        regularization_weight = regularization_weights ** 2.0

        for i in range(self.pixels):
            for j in pixel_neighbors[i]:
                regularization_matrix[i, i] += regularization_weight[j]
                regularization_matrix[j, j] += regularization_weight[j]
                regularization_matrix[i, j] -= regularization_weight[j]
                regularization_matrix[j, i] -= regularization_weight[j]

        return regularization_matrix

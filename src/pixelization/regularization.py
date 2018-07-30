import numpy as np

class RegularizationConstant(object):

    pixels = None
    regularization_coefficients = None

    def regularization_matrix_from_pix_neighbors(self, pix_neighbors):
        """
        Setup a pixelization's constant regularization matrix (see test_pixelization.py)

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


class RegularizationWeighted(object):

    pixels = None
    regularization_coefficients = None
    pix_signal_scale = None

    def pix_signals_from_images(self, image_to_pix, galaxy_image):
        """Compute the (scaled) signal in each pixel, where the signal is the sum of its image-pixel fluxes. \
        These pix-signals are then used to compute the effective regularization weight of each pixel.

        The pix signals are scaled in the following ways:

        1) Divided by the number of image-pixels in the pixel, to ensure all pixels have the same \
        'relative' signal (i.e. a pixel with 10 images-pixels doesn't have x2 the signal of one with 5).

        2) Divided by the maximum pix-signal, so that all signals vary between 0 and 1. This ensures that the \
        regularizations weights they're used to compute are defined identically for all image units / SNR's.

        3) Raised to the power of the hyper-parameter *pix_signal_scale*, so the method can control the relative \
        contribution of the different regions of regularization.
        """

        pix_signals = np.zeros((self.pixels,))
        pix_sizes = np.zeros((self.pixels,))

        for image_index in range(galaxy_image.shape[0]):
            pix_signals[image_to_pix[image_index]] += galaxy_image[image_index]
            pix_sizes[image_to_pix[image_index]] += 1

        pix_signals /= pix_sizes
        pix_signals /= max(pix_signals)

        return pix_signals ** self.pix_signal_scale

    def regularization_weights_from_pix_signals(self, pix_signals):
        """Compute the regularization weights, which represent the effective regularization coefficient of every \
        pixel. These are computed using the (scaled) pix-signal in each pixel.

        Two regularization coefficients are used which map to:

        1) pix_signals - This regularizes pix-plane pixels with a high pix-signal (i.e. where the pix is).
        2) 1.0 - pix_signals - This regularizes pix-plane pixels with a low pix-signal (i.e. background sky)
        """
        return (self.regularization_coefficients[0] * pix_signals +
                self.regularization_coefficients[1] * (1.0 - pix_signals)) ** 2.0

    def regularization_matrix_from_pix_neighbors(self, regularization_weights, pix_neighbors):
        """
        Setup a weighted regularization matrix, where all pixels are regularized with one another in both \
        directions using a different effective regularization coefficient.

        Parameters
        ----------
        regularization_weights : list(float)
            The regularization weight of each pixel
        pix_neighbors : [[]]
            A list of the neighbors of each pixel.
        """

        regularization_matrix = np.zeros(shape=(self.pixels, self.pixels))

        reg_weight = regularization_weights ** 2.0

        for i in range(self.pixels):
            for j in pix_neighbors[i]:
                regularization_matrix[i, i] += reg_weight[j]
                regularization_matrix[j, j] += reg_weight[j]
                regularization_matrix[i, j] -= reg_weight[j]
                regularization_matrix[j, i] -= reg_weight[j]

        return regularization_matrix
import numpy as np
import numba

@numba.jit(nopython=True, parallel=True)
def constant_regularization_matrix_from_pixel_neighbors(coefficients, pixel_neighbors, pixel_neighbors_size):
    """From the pixel-neighbors, setup the regularization matrix using the constant regularization scheme.

    Parameters
    ----------
    pixel_neighbors : [[]]
        A list of the neighbors of each pixel.
    """

    pixels = len(pixel_neighbors)

    regularization_matrix = np.zeros(shape=(pixels, pixels))

    regularization_coefficient = coefficients[0] ** 2.0

    for i in range(pixels):
        regularization_matrix[i, i] += 1e-8
        for j in range(pixel_neighbors_size[i]):
            neighbor_index = pixel_neighbors[i, j]
            regularization_matrix[i, i] += regularization_coefficient
            regularization_matrix[i, neighbor_index] -= regularization_coefficient

    return regularization_matrix

@numba.jit(nopython=True, parallel=True)
def weighted_pixel_signals_from_images(pixels, signal_scale, regular_to_pix, galaxy_image):
    """Compute the (scaled) signal in each pixel, where the signal is the sum of its datas_-pixel fluxes. \
    These pixel-signals are used to compute the effective regularization weight of each pixel.

    The pixel signals are scaled in the following ways:

    1) Divided by the number of datas_-pixels in the pixel, to ensure all pixels have the same \
    'relative' signal (i.e. a pixel with 10 regular-pixels doesn't have x2 the signal of one with 5).

    2) Divided by the maximum pixel-signal, so that all signals vary between 0 and 1. This ensures that the \
    regularizations weights are defined identically for any datas_ units or signal-to-noise ratio.

    3) Raised to the power of the hyper-parameter *signal_scale*, so the method can control the relative \
    contribution regularization in different regions of pixelization.
    """

    pixel_signals = np.zeros((pixels,))
    pixel_sizes = np.zeros((pixels,))

    for regular_index in range(galaxy_image.shape[0]):
        pixel_signals[regular_to_pix[regular_index]] += galaxy_image[regular_index]
        pixel_sizes[regular_to_pix[regular_index]] += 1

    pixel_signals /= pixel_sizes
    pixel_signals /= np.max(pixel_signals)

    return pixel_signals ** signal_scale

def weighted_regularization_weights_from_pixel_signals(coefficients, pixel_signals):
    """Compute the regularization weights, which are the effective regularization coefficient of every \
    pixel. They are computed using the (scaled) pixel-signal of each pixel.

    Two regularization coefficients are used, corresponding to the:

    1) (pixel_signals) - pixels with a high pixel-signal (i.e. where the signal is located in the pixelization).
    2) (1.0 - pixel_signals) - pixels with a low pixel-signal (i.e. where the signal is not located in the \
     pixelization).
    """
    return (coefficients[0] * pixel_signals + coefficients[1] * (1.0 - pixel_signals)) ** 2.0

@numba.jit(nopython=True, parallel=True)
def weighted_regularization_matrix_from_pixel_neighbors(regularization_weights, pixel_neighbors,
                                                        pixel_neighbors_size):
    """ From the pixel-neighbors, setup the regularization matrix using the weighted regularization scheme.

    Parameters
    ----------
    regularization_weights : list(float)
        The regularization_matrix weight of each pixel
    pixel_neighbors : [[]]
        A list of the neighbors of each pixel.
    """

    pixels = len(regularization_weights)

    regularization_matrix = np.zeros(shape=(pixels, pixels))

    regularization_weight = regularization_weights ** 2.0

    for i in range(pixels):
        for j in range(pixel_neighbors_size[i]):
            neighbor_index = pixel_neighbors[i, j]
            regularization_matrix[i, i] += regularization_weight[neighbor_index]
            regularization_matrix[neighbor_index, neighbor_index] += regularization_weight[neighbor_index]
            regularization_matrix[i, neighbor_index] -= regularization_weight[neighbor_index]
            regularization_matrix[neighbor_index, i] -= regularization_weight[neighbor_index]

    return regularization_matrix
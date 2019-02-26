from autolens import decorator_util
import numpy as np

from autolens.data import convolution


class ConvolverMappingMatrix(convolution.Convolver):

    def __init__(self, mask, psf):
        """
        Class to create number array and frames used to convolve a psf with a 1D vector of non-masked values.

        Parameters
        ----------
        mask : Mask
            An datas_ masks, where True eliminates datas.
        psf : datas_.PSF or ndarray
            An array representing a PSF.
        """

        super(ConvolverMappingMatrix, self).__init__(mask, psf)

    def convolve_mapping_matrix(self, mapping_matrix):
        """For a given inversion mapping matrix, convolve every pixel's mapped regular with the PSF kernel.

        A mapping matrix provides non-zero entries in all elements which map two pixels to one another
        (see *inversions.mappers*).

        For example, lets take an regular which is masked using a 'cross' of 5 pixels:

        [[ True, False,  True]],
        [[False, False, False]],
        [[ True, False,  True]]

        As example mapping matrix of this cross is as follows (5 regular pixels x 3 source pixels):

        [1, 0, 0] [0->0]
        [1, 0, 0] [1->0]
        [0, 1, 0] [2->1]
        [0, 1, 0] [3->1]
        [0, 0, 1] [4->2]

        For each source-pixel, we can create an regular of its unit-surface brightnesses by mapping the non-zero
        entries back to masks. For example, doing this for source pixel 1 gives:

        [[0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0]]
        [[0.0, 0.0, 0.0]]

        And source pixel 2:

        [[0.0, 0.0, 0.0]],
        [[0.0, 1.0, 1.0]]
        [[0.0, 0.0, 0.0]]

        We then convolve each of these regular with our PSF kernel, in 2 dimensions, like we would a normal regular. For
        example, using the kernel below:

        kernel:

        [[0.0, 0.1, 0.0]]
        [[0.1, 0.6, 0.1]]
        [[0.0, 0.1, 0.0]]

        Blurred Source Pixel 1 (we don't need to perform the convolution into masked pixels):

        [[0.0, 0.6, 0.0]],
        [[0.6, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]]

        Blurred Source pixel 2:

        [[0.0, 0.0, 0.0]],
        [[0.0, 0.7, 0.7]],
        [[0.0, 0.0, 0.0]]

        Finally, we map each of these blurred regular back to a blurred mapping matrix, which is analogous to the
        mapping matrix.

        [0.6, 0.0, 0.0] [0->0]
        [0.6, 0.0, 0.0] [1->0]
        [0.0, 0.7, 0.0] [2->1]
        [0.0, 0.7, 0.0] [3->1]
        [0.0, 0.0, 0.6] [4->2]

        If the mapping matrix is sub-gridded, we perform the convolution on the fractional surface brightnesses in an
        identical fashion to above.

        Parameters
        -----------
        mapping_matrix : ndarray
            The 2D mapping matix describing how every inversion pixel maps to an datas_ pixel.
        """
        return self.convolve_matrix_jit(mapping_matrix, self.image_frame_indexes,
                                        self.image_frame_psfs, self.image_frame_lengths)

    @staticmethod
    @decorator_util.jit()
    def convolve_matrix_jit(mapping_matrix, image_frame_indexes, image_frame_kernels, image_frame_lengths):

        blurred_mapping_matrix = np.zeros(mapping_matrix.shape)

        for pixel_index in range(mapping_matrix.shape[1]):
            for image_index in range(mapping_matrix.shape[0]):

                value = mapping_matrix[image_index, pixel_index]

                if value > 0:

                    frame_indexes = image_frame_indexes[image_index]
                    frame_kernels = image_frame_kernels[image_index]
                    frame_length = image_frame_lengths[image_index]

                    for kernel_index in range(frame_length):
                        vector_index = frame_indexes[kernel_index]
                        kernel = frame_kernels[kernel_index]
                        blurred_mapping_matrix[vector_index, pixel_index] += value * kernel

        return blurred_mapping_matrix

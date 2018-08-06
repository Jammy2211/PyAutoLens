import numpy as np
from autolens import exc
import numba

"""
This module is for the application of convolution to sparse_grid vectors.

Take a simple mask:

[[True, False, True],
 [False, False, False],
 [True, False, True]]
 
Here True means that the value is masked.

A set of values in a corresponding image_grid might be represented in a 1D array:

[2, 8, 2, 5, 7, 5, 3, 1, 4]

This module allows us to find the relationships between data_to_image in a mask for a psf of a given sub_grid_size so that
convolutions can be efficiently applied to reduced arrays such as the one above.

A Convolver can be created for a given mask:

frame_maker = Convolver(self.mask)

This can then produce a convolver_image for any given psf shape and corresponding blurring region mask:

convolver_image = frame_maker.convolver_for_kernel_shape((3, 3), blurring_region_mask)

Here the blurring region mask describes the region under the mask from which a given PSF psf may blur pixels. If the
regular mask specifies True for a givena pixel and the blurring region mask False then that pixel will be blurred in
using the blurring array.

A convolver_image can then be made for any given psf:

kernel_convolver = convolver_image.convolver_for_kernel(psf)

Which is applied to a reduced array and blurring array:

convolved_array = convolver_image.convolve_array(array, blurring_array)

The array is pixels within the non-masked region, whilst the blurring array is pixels outside of the non-masked region
but inside of the blurring mask region.

The convolver_image can also be applied for some sub_grid-shape of the psf:

convolved_vector = convolver_image.convolve_vector(vector, sub_shape=(3, 3))

Or applied to a whole mapping matrix:

convolved_mapping_matrix = convolver_image.convolve_mapping_matrix(mapping_matrix)

Where the mapping matrix is an array of dictionaries with each index of the array corresponding to a source pixel.

It is also possible to specify a blurring region mask:

Convolver(self.mask, blurring_region_mask)

PSF will be calculated from values that are masked by mask but not by blurring region mask. That is, any entry with a 
True value for mask and a False value for blurring region mask may contribute to the PSF convolved value of a nearby
entry with a False value for mask.

"""


class Convolver(object):
    """Class to setup the 1D convolution of an image / mapping matrix.

    IMAGE FRAMES:
    ------------

    For a masked image, in 2D, one can compute for every unmasked pixel all other unmasked pixels it will blur light \
    into for a given PSF psf size, e.g.:

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an example image.Mask, where:
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from analysis)
    |x|x|x|o|o|o|x|x|x|x|     o = False (Pixel is not masked and included in analysis)
    |x|x|x|o|o|o|x|x|x|x|
    |x|x|x|o|o|o|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|

    Here, there are 9 unmasked pixels. Indexing of each unmasked pixel goes from the top-left corner right,
    and downwards, therefore:

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|0|1|2|x|x|x|x|
    |x|x|x|3|4|5|x|x|x|x|
    |x|x|x|6|7|8|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|

    For every unmasked pixel, the Convolver over-lays the PSF psf and computes three quantities;

    image_frame_indexes - The indexes of all image pixels it will blur light into.
    image_frame_psfs - The psf values that overlap each image pixel it will blur light into.
    image_frame_length - The number of image-pixels it will blur light into (because unmasked pixels are excluded)

    For example, if we had the following 3x3 psf:

    |0.1|0.2|0.3|
    |0.4|0.5|0.6|
    |0.7|0.8|0.9|

    For pixel 0 above, when we overlap the psf 4 unmasked pixels overlap this psf, such that:

    image_frame_indexes = [0, 1, 3, 4]
    image_frame_psfs = [0.5, 0.6, 0.8, 0.9]
    image_frame_length = 4

    Noting that the other 5 psf values (0.1, 0.2, 0.3, 0.4, 0.7) overlap masked pixels and are thus discard.

    For pixel 1, we get the following results:

    image_frame_indexes = [0, 1, 2, 3, 4, 5]
    image_frame_psfs = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    image_frame_lengths = 6

    In the majority of cases, there will be no unmasked pixels when the psf overlaps. This is the case above for \
    central pixel 4, where:

    image_frame_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    image_frame_psfs = [0,1, 0.2, 0,3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    image_frame_lengths = 6

    Once we have set up all these quantities, the convolution routine simply uses them to convolve a 1D array of image
    data / a mapping matrix image.

    BLURRING FRAMES:
    --------------

    Whilst the scheme above accounts for all blurred light within the mask, it does not account for the fact that
    pixels outside of the mask will also blur light into it. For galaxy light profiles, this effect is accounted for \
    using blurring frames, however it is omitted for mapping matrix images.

    First, a blurring mask is computed from a mask, which describes all pixels which are close enough to the mask \
    to blur light into it for a given psf size. Following the example above, the following blurring mask is \
    computed:

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an example image.Mask, where:
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|o|o|o|o|o|x|x|x|     x = True (Pixel is masked and excluded from analysis)
    |x|x|o|x|x|x|o|x|x|x|     o = False (Pixel is not masked and included in analysis)
    |x|x|o|x|x|x|o|x|x|x|
    |x|x|o|x|x|x|o|x|x|x|
    |x|x|o|o|o|o|o|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|

    Indexing again goes from the top-left corner right and downwards:

    |x|x| x| x| x| x| x|x|x|x|
    |x|x| x| x| x| x| x|x|x|x|
    |x|x| x| x| x| x| x|x|x|x|
    |x|x| 0| 1| 2| 3| 4|x|x|x|
    |x|x| 5| x| x| x| 6|x|x|x|
    |x|x| 7| x| x| x| 8|x|x|x|
    |x|x| 9| x| x| x|10|x|x|x|
    |x|x|11|12|13|14|15|x|x|x|
    |x|x| x| x| x| x| x|x|x|x|
    |x|x| x| x| x| x| x|x|x|x|

    For every unmasked blurring-pixel, the Convolver over-lays the PSF psf and computes three quantities;

    blurring_frame_indexes - The indexes of all unmasked image pixels it will blur light into.
    bluring_frame_kernels - The psf values that overlap each unmasked image pixel it will blur light into.
    blurring_frame_length - The number of image-pixels it will blur light into (because unmasked pixels are excluded)

    Note, therefore, that the blurring frame does not perform any blurring which blurs light into other blurring pixels.
    It only performs computations which add light inside of the mask, which is the most computationally efficient method.

    For pixel 0 above, when we overlap the 3x3 psf above only 1 unmasked image pixels overlap the psf, such that:

    blurring_frame_indexes = [0]
    blurring_frame_psfs = [0.9]
    blurring_frame_length = 1

    For pixel 1 above, when we overlap the 3x3 psf above 2 unmasked image pixels overlap the psf, such that:

    blurring_frame_indexes = [0, 1]
    blurring_frame_psfs = [0.8, 0.9]
    blurring_frame_length = 2

    For pixel 3 above, when we overlap the 3x3 psf above 3 unmasked image pixels overlap the psf, such that:

    blurring_frame_indexes = [0, 1, 2]
    blurring_frame_psfs = [0.7, 0.8, 0.9]
    blurring_frame_length = 3
    """

    def __init__(self, mask, psf):
        """
        Class to create image frames and blurring frames used to convolve a psf with a 1D image of non-masked \
        values.

        Parameters
        ----------
        mask : Mask
            A mask where True eliminates data.
        burring_mask : Mask
            A mask of pixels outside the mask but whose light blurs into it after convolution.
        psf : image.PSF or ndarray
            An array representing a PSF psf.
        """

        if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
            raise exc.KernelException("PSF kernel must be odd")

        self.mask_index_array = np.full(mask.shape, -1)
        self.pixels_in_mask = int(np.size(mask) - np.sum(mask))

        count = 0
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if not mask[x, y]:
                    self.mask_index_array[x, y] = count
                    count += 1

        self.psf = psf
        self.psf_shape = psf.shape
        self.psf_max_size = self.psf_shape[0] * self.psf_shape[1]

        image_index = 0
        self.image_frame_indexes = np.zeros((self.pixels_in_mask, self.psf_max_size), dtype='int')
        self.image_frame_psfs = np.zeros((self.pixels_in_mask, self.psf_max_size))
        self.image_frame_lengths = np.zeros((self.pixels_in_mask), dtype='int')
        for x in range(self.mask_index_array.shape[0]):
            for y in range(self.mask_index_array.shape[1]):
                if not mask[x][y]:
                    image_frame_indexes, image_frame_psfs = self.frame_at_coords_jit((x, y), mask,
                                                                                        self.mask_index_array, self.psf[:, :])
                    self.image_frame_indexes[image_index, :] = image_frame_indexes
                    self.image_frame_psfs[image_index, :] = image_frame_psfs
                    self.image_frame_lengths[image_index] = image_frame_indexes[image_frame_indexes >= 0].shape[0]
                    image_index += 1

    @staticmethod
    @numba.jit(nopython=True)
    def frame_at_coords_jit(coords, mask, mask_index_array, psf):
        """
        Parameters
        ----------
        coords: (int, int)
            The image_grid of mask_index_array on which the frame should be centred
        psf_shape: (int, int)
            The shape of the psf for which this frame will be used
        Returns
        -------
        frame: ndarray
            A subset of mask_index_array of shape psf_shape where elements with image_grid outside of image_frame_indexes have
            value -1
        """

        psf_shape = psf.shape
        psf_max_size = psf_shape[0] * psf_shape[1]

        half_x = int(psf_shape[0] / 2)
        half_y = int(psf_shape[1] / 2)

        frame = -1*np.ones((psf_max_size))
        psf_frame = -1.0*np.ones((psf_max_size))

        count = 0
        for i in range(psf_shape[0]):
            for j in range(psf_shape[1]):
                x = coords[0] - half_x + i
                y = coords[1] - half_y + j
                if 0 <= x < mask_index_array.shape[0] and 0 <= y < mask_index_array.shape[1]:
                    value = mask_index_array[x, y]
                    if value >= 0 and not mask[x, y]:
                        frame[count] = value
                        psf_frame[count] = psf[i, j]
                        count += 1

        return frame, psf_frame


class ConvolverImage(Convolver):

    def __init__(self, mask, blurring_mask, psf):
        """
        Class to create image frames and blurring frames used to convolve a psf with a 1D image of non-masked \
        values.

        Parameters
        ----------
        mask : Mask
            A mask where True eliminates data.
        burring_mask : Mask
            A mask of pixels outside the mask but whose light blurs into it after convolution.
        psf : image.PSF or ndarray
            An array representing a PSF psf.
        """

        if mask.shape != blurring_mask.shape:
            raise exc.KernelException("Mask and Blurring mask must be same shape to generate Convolver")

        super(ConvolverImage, self).__init__(mask, psf)

        blurring_mask = blurring_mask
        self.pixels_in_blurring_mask = int(np.size(blurring_mask) - np.sum(blurring_mask))

        image_index = 0
        self.blurring_frame_indexes = np.zeros((self.pixels_in_blurring_mask, self.psf_max_size), dtype='int')
        self.blurring_frame_psfs = np.zeros((self.pixels_in_blurring_mask, self.psf_max_size))
        self.blurring_frame_lengths = np.zeros((self.pixels_in_blurring_mask), dtype='int')
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x][y] and not blurring_mask[x, y]:
                    image_frame_indexes, image_frame_psfs = self.frame_at_coords_jit((x, y), mask,
                                                                                        self.mask_index_array, self.psf)
                    self.blurring_frame_indexes[image_index, :] = image_frame_indexes
                    self.blurring_frame_psfs[image_index, :] = image_frame_psfs
                    self.blurring_frame_lengths[image_index] = image_frame_indexes[image_frame_indexes >= 0].shape[0]
                    image_index += 1

    def convolve_image_jit(self, image_array, blurring_array):
        return self.convolve_image_jitted(image_array, self.image_frame_indexes,
                                          self.image_frame_psfs, self.image_frame_lengths,
                                          blurring_array, self.blurring_frame_indexes,
                                          self.blurring_frame_psfs, self.blurring_frame_lengths)

    @staticmethod
    @numba.jit(nopython=True)
    def convolve_image_jitted(image_array, image_frame_indexes, image_frame_kernels, image_frame_lengths,
                              blurring_array, blurring_frame_indexes, blurring_frame_kernels, blurring_frame_lengths):

        new_array = np.zeros(image_array.shape)

        for image_index in range(len(image_array)):

            frame_indexes = image_frame_indexes[image_index]
            frame_kernels = image_frame_kernels[image_index]
            frame_length = image_frame_lengths[image_index]
            value = image_array[image_index]

            for kernel_index in range(frame_length):

                vector_index = frame_indexes[kernel_index]
                kernel = frame_kernels[kernel_index]
                new_array[vector_index] += value * kernel

        for image_index in range(len(blurring_array)):

            frame_indexes = blurring_frame_indexes[image_index]
            frame_kernels = blurring_frame_kernels[image_index]
            frame_length = blurring_frame_lengths[image_index]
            value = blurring_array[image_index]

            for kernel_index in range(frame_length):

                vector_index = frame_indexes[kernel_index]
                kernel = frame_kernels[kernel_index]
                new_array[vector_index] += value * kernel

        return new_array


class ConvolverMappingMatrix(Convolver):

    def __init__(self, mask, psf):
        """
        Class to create number array and frames used to convolve a psf with a 1D vector of non-masked values.

        Parameters
        ----------
        mask : Mask
            A mask where True eliminates data.
        mask : Mask
            A mask of pixels outside the mask but whose light blurs into it after convolution.
        psf : image.PSF or ndarray
            An array representing a PSF psf.

        Attributes
        ----------
        blurring_frame_indexes: [ndarray]
            An array of frames created by the frame maker. Maps positions in the psf to values in the 1D vector for
            masked pixels.
        image_frame_indexes: [ndarray]
            An array of frames created by the frame maker. A frame maps positions in the psf to values in the 1D
            vector.
        """

        super(ConvolverMappingMatrix, self).__init__(mask, psf)

    def convolve_mapping_matrix_jit(self, mapping):
        """
        Simple version of function that applies this convolver_image to a whole mapping matrix.

        Parameters
        ----------
        blurring_array: [Float]
            An array representing the mapping of a source pixel to a set of image pixels within the blurring region.
        array: [float]
            An array representing the mapping of a source pixel to a set of image pixels.

        Returns
        -------
        convolved_array: [float]
            A matrix representing the mapping of source data_to_image to image_grid data_to_image accounting for
            convolution
        """
        return self.convolve_matrix_jitted(mapping, self.image_frame_indexes,
                                           self.image_frame_psfs, self.image_frame_lengths)

    @staticmethod
    @numba.jit(nopython=True)
    def convolve_matrix_jitted(mapping_matrix, image_frame_indexes, image_frame_kernels, image_frame_lengths):

        blurred_mapping = np.zeros(mapping_matrix.shape)

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
                        blurred_mapping[vector_index, pixel_index] += value * kernel

        return blurred_mapping
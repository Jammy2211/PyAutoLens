import numba
import numpy as np

from autolens import exc

class Convolver(object):
    """Class to setup the 1D convolution of an regular / mapping matrix.

    Take a simple 3x3 regular and mask:

    [[2, 8, 2],
    [5, 7, 5],
    [3, 1, 4]]

    [[True, False, True],   (True means that the value is masked)
    [False, False, False],
    [True, False, True]]

    A set of values in a corresponding 1d array of this regular might be represented as:

    [2, 8, 2, 5, 7, 5, 3, 1, 4]

    and after masking as:

    [8, 5, 7, 5, 1]

    Setup is required to perform 2D real-space convolution on the masked regular array. This module finds the \
    relationship between the unmasked 2D regular data, masked regular data and psf, so that 2D real-space convolutions \
    can be efficiently applied to reduced 1D masked arrays.

    This calculation also accounts for the blurring of light outside of the masked regions which blurs into \
    the masked region.

    IMAGE FRAMES:
    ------------

    For a masked regular in 2D, one can compute for every pixel all of the unmasked pixels it will blur light into for \
    a given PSF psf size, e.g.:

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an imaging.Mask, where:
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from lensing)
    |x|x|x|o|o|o|x|x|x|x|     o = False (Pixel is not masked and included in lensing)
    |x|x|x|o|o|o|x|x|x|x|
    |x|x|x|o|o|o|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|

    Here, there are 9 unmasked pixels. Indexing of each unmasked pixel goes from the top-left corner right and \
    downwards, therefore:

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

    For every unmasked pixel, the Convolver over-lays the PSF and computes three quantities;

    image_frame_indexes - The indexes of all masked regular pixels it will blur light into.
    image_frame_psfs - The psf values that overlap each masked regular pixel it will blur light into.
    image_frame_length - The number of masked regular-pixels it will blur light into (unmasked pixels are excluded)

    For example, if we had the following 3x3 psf:

    |0.1|0.2|0.3|
    |0.4|0.5|0.6|
    |0.7|0.8|0.9|

    For pixel 0 above, when we overlap the psf 4 unmasked pixels overlap this psf, such that:

    image_frame_indexes = [0, 1, 3, 4]
    image_frame_psfs = [0.5, 0.6, 0.8, 0.9]
    image_frame_length = 4

    Noting that the other 5 psf values (0.1, 0.2, 0.3, 0.4, 0.7) overlap masked pixels and are thus discarded.

    For pixel 1, we get the following results:

    image_frame_indexes = [0, 1, 2, 3, 4, 5]
    image_frame_psfs = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    image_frame_lengths = 6

    In the majority of cases, the psf will overlap only unmasked pixels. This is the case above for \
    central pixel 4, where:

    image_frame_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    image_frame_psfs = [0,1, 0.2, 0,3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    image_frame_lengths = 9

    Once we have set up all these quantities, the convolution routine simply uses them to convolve a 1D array of a
    masked regular or the masked regular of a mapping in the inversion module.

    BLURRING FRAMES:
    --------------

    Whilst the scheme above accounts for all blurred light within the mask, it does not account for the fact that \
    pixels outside of the mask will also blur light into it. This effect is accounted for using blurring frames.

    It is omitted for mapping matrix blurring, as an inversion does not fit data outside of the mask.

    First, a blurring mask is computed from a mask, which describes all pixels which are close enough to the mask \
    to blur light into it for a given psf size. Following the example above, the following blurring mask is \
    computed:

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an example regular.Mask, where:
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|o|o|o|o|o|x|x|x|     x = True (Pixel is masked and excluded from lensing)
    |x|x|o|x|x|x|o|x|x|x|     o = False (Pixel is not masked and included in lensing)
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

    blurring_frame_indexes - The indexes of all unmasked regular pixels (not unmasked blurring regular pixels) it will \
    blur light into.
    bluring_frame_kernels - The psf values that overlap each regular pixel it will blur light into.
    blurring_frame_length - The number of regular-pixels it will blur light into.

    The blurring frame therefore does not perform any blurring which blurs light into other blurring pixels. \
    It only performs computations which add light inside of the mask.

    For pixel 0 above, when we overlap the 3x3 psf above only 1 unmasked regular pixels overlaps the psf, such that:

    blurring_frame_indexes = [0] (This 0 refers to regular pixel 0 within the mask, not blurring_frame_pixel 0)
    blurring_frame_psfs = [0.9]
    blurring_frame_length = 1

    For pixel 1 above, when we overlap the 3x3 psf above 2 unmasked regular pixels overlap the psf, such that:

    blurring_frame_indexes = [0, 1]  (This 0 and 1 refer to regular pixels 0 and 1 within the mask)
    blurring_frame_psfs = [0.8, 0.9]
    blurring_frame_length = 2

    For pixel 3 above, when we overlap the 3x3 psf above 3 unmasked regular pixels overlap the psf, such that:

    blurring_frame_indexes = [0, 1, 2]  (Again, these are regular pixels 0, 1 and 2)
    blurring_frame_psfs = [0.7, 0.8, 0.9]
    blurring_frame_length = 3
    """

    def __init__(self, mask, psf):
        """
        Class to create regular frames and blurring frames used to convolve a psf with a 1D regular of non-masked \
        values.

        Parameters
        ----------
        mask : Mask
            A mask where True eliminates data.
        psf : regular.PSF or ndarray
            An array representing a PSF.
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
                    image_frame_indexes, image_frame_psfs = self.frame_at_coordinates_jit((x, y), mask,
                                                                                          self.mask_index_array,
                                                                                     self.psf[:, :])
                    self.image_frame_indexes[image_index, :] = image_frame_indexes
                    self.image_frame_psfs[image_index, :] = image_frame_psfs
                    self.image_frame_lengths[image_index] = image_frame_indexes[image_frame_indexes >= 0].shape[0]
                    image_index += 1

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def frame_at_coordinates_jit(coordinates, mask, mask_index_array, psf):
        """ Compute the frame (indexes of pixels light is blurred into) and psf_frame (psf kernel values of those \
        pixels) for a given coordinate in a mask and its PSF.

        Parameters
        ----------
        coordinates: (int, int)
            The coordinates of mask_index_array on which the frame should be centred
        psf_shape: (int, int)
            The shape of the psf for which this frame will be used
        """

        psf_shape = psf.shape
        psf_max_size = psf_shape[0] * psf_shape[1]

        half_x = int(psf_shape[0] / 2)
        half_y = int(psf_shape[1] / 2)

        frame = -1 * np.ones((psf_max_size))
        psf_frame = -1.0 * np.ones((psf_max_size))

        count = 0
        for i in range(psf_shape[0]):
            for j in range(psf_shape[1]):
                x = coordinates[0] - half_x + i
                y = coordinates[1] - half_y + j
                if 0 <= x < mask_index_array.shape[0] and 0 <= y < mask_index_array.shape[1]:
                    value = mask_index_array[x, y]
                    if value >= 0 and not mask[x, y]:
                        frame[count] = value
                        psf_frame[count] = psf[i, j]
                        count += 1

        return frame, psf_frame


class ConvolverImage(Convolver):

    def __init__(self, mask, blurring_mask, psf):
        """ Class to create regular frames and blurring frames used to convolve a psf with a 1D regular of non-masked \
        values.

        Parameters
        ----------
        mask : Mask
            The regular mask, where True eliminates data.
        blurring_mask : Mask
            A mask of pixels outside the mask but whose light blurs into it after PSF convolution.
        psf : regular.PSF or ndarray
            An array representing a PSF.
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
                    image_frame_indexes, image_frame_psfs = self.frame_at_coordinates_jit((x, y), mask,
                                                                                          self.mask_index_array, self.psf)
                    self.blurring_frame_indexes[image_index, :] = image_frame_indexes
                    self.blurring_frame_psfs[image_index, :] = image_frame_psfs
                    self.blurring_frame_lengths[image_index] = image_frame_indexes[image_frame_indexes >= 0].shape[0]
                    image_index += 1

    def convolve_image(self, image_array, blurring_array):
        """For a given 1D regular array and blurring array, convolve the two using this convolver.

        Parameters
        -----------
        image_array : ndarray
            1D array of the regular values which are to be blurred with the convolver's PSF.
        blurring_array : ndarray
            1D array of the blurring regular values which blur into the regular-array after PSF convolution.
        """
        return self.convolve_jit(image_array, self.image_frame_indexes, self.image_frame_psfs, self.image_frame_lengths,
                                 blurring_array, self.blurring_frame_indexes, self.blurring_frame_psfs,
                                 self.blurring_frame_lengths)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def convolve_jit(image_array, image_frame_indexes, image_frame_kernels, image_frame_lengths,
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

        for blurring_index in range(len(blurring_array)):

            frame_indexes = blurring_frame_indexes[blurring_index]
            frame_kernels = blurring_frame_kernels[blurring_index]
            frame_length = blurring_frame_lengths[blurring_index]
            value = blurring_array[blurring_index]

            for kernel_index in range(frame_length):
                vector_index = frame_indexes[kernel_index]
                kernel = frame_kernels[kernel_index]
                new_array[vector_index] += value * kernel

        return new_array
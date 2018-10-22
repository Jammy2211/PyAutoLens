import numba
import numpy as np

from autolens import exc

"""
This module is for the application of convolution to _data vectors.



A Convolver can be created for a given mask and psf:

convolver = Convolver(mask, psf)

This can then produce a convolved _data for any convolver_image for any given psf shape and corresponding blurring region mask:

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

Or applied to a whole mapping_matrix matrix:

convolved_mapping_matrix = convolver_image.convolve_mapping_matrix(mapping_matrix)

Where the mapping_matrix matrix is an array of dictionaries with each index of the array corresponding to a source pixel.

It is also possible to specify a blurring region mask:

Convolver(self.mask, blurring_region_mask)

PSF will be calculated from values that are masked by mask but not by blurring region mask. That is, any entry with a 
True value for mask and a False value for blurring region mask may contribute to the PSF convolved value of a nearby
entry with a False value for mask.

"""


class Convolver(object):
    """Class to setup the 1D convolution of an _data / mapping_matrix matrix.

    Take a simple 3x3 _data and mask:

    [[2, 8, 2],
    [5, 7, 5],
    [3, 1, 4]]

    [[True, False, True],   (True means that the value is masked)
    [False, False, False],
    [True, False, True]]

    A set of values in a corresponding 1d array of this _data might be represented as:

    [2, 8, 2, 5, 7, 5, 3, 1, 4]

    and after masking as:

    [8, 5, 7, 5, 1]

    Setup is required to perform 2D real-space convolution on the masked _data array. This module finds the \
    relationship between the unmasked 2D _data data, masked _data data and psf, so that 2D real-space convolutions \
    can be efficiently applied to reduced 1D masked arrays.

    This calculation also accounts for the blurring of light outside of the masked regions which blurs into \
    the masked region.

    IMAGE FRAMES:
    ------------

    For a masked _data in 2D, one can compute for every pixel all of the unmasked pixels it will blur light into for \
    a given PSF psf size, e.g.:

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an howtolens imaging.Mask, where:
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

    image_frame_indexes - The indexes of all masked _data pixels it will blur light into.
    image_frame_psfs - The psf values that overlap each masked _data pixel it will blur light into.
    image_frame_length - The number of masked _data-pixels it will blur light into (unmasked pixels are excluded)

    For howtolens, if we had the following 3x3 psf:

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
    masked _data or the masked _data of a mapping_matrix in the inversion module.

    BLURRING FRAMES:
    --------------

    Whilst the scheme above accounts for all blurred light within the mask, it does not account for the fact that \
    pixels outside of the mask will also blur light into it. This effect is accounted for using blurring frames.

    It is omitted for mapping_matrix matrix blurring, as a inversion does not incorrect_fit data outside of the mask.

    First, a blurring mask is computed from a mask, which describes all pixels which are close enough to the mask \
    to blur light into it for a given psf size. Following the howtolens above, the following blurring mask is \
    computed:

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an howtolens _data.Mask, where:
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

    blurring_frame_indexes - The indexes of all unmasked _data pixels (not unmasked blurring _data pixells) it will \
    blur light into.
    bluring_frame_kernels - The psf values that overlap each _data pixel it will blur light into.
    blurring_frame_length - The number of _data-pixels it will blur light into.

    The blurring frame therefore does not perform any blurring which blurs light into other blurring pixels. \
    It only performs computations which add light inside of the mask.

    For pixel 0 above, when we overlap the 3x3 psf above only 1 unmasked _data pixels overlaps the psf, such that:

    blurring_frame_indexes = [0] (This 0 refers to _data pixel 0 within the mask, not blurring_frame_pixel 0)
    blurring_frame_psfs = [0.9]
    blurring_frame_length = 1

    For pixel 1 above, when we overlap the 3x3 psf above 2 unmasked _data pixels overlap the psf, such that:

    blurring_frame_indexes = [0, 1]  (This 0 and 1 refer to _data pixels 0 and 1 within the mask)
    blurring_frame_psfs = [0.8, 0.9]
    blurring_frame_length = 2

    For pixel 3 above, when we overlap the 3x3 psf above 3 unmasked _data pixels overlap the psf, such that:

    blurring_frame_indexes = [0, 1, 2]  (Again, these are _data pixels 0, 1 and 2)
    blurring_frame_psfs = [0.7, 0.8, 0.9]
    blurring_frame_length = 3
    """

    def __init__(self, mask, psf):
        """
        Class to create _data frames and blurring frames used to convolve a psf with a 1D _data of non-masked \
        values.

        Parameters
        ----------
        mask : Mask
            A mask where True eliminates data.
        burring_mask : Mask
            A mask of pixels outside the mask but whose light blurs into it after convolution.
        psf : _data.PSF or ndarray
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
                                                                                     self.mask_index_array,
                                                                                     self.psf[:, :])
                    self.image_frame_indexes[image_index, :] = image_frame_indexes
                    self.image_frame_psfs[image_index, :] = image_frame_psfs
                    self.image_frame_lengths[image_index] = image_frame_indexes[image_frame_indexes >= 0].shape[0]
                    image_index += 1

    @staticmethod
    @numba.jit(nopython=True, cache=True)
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

        frame = -1 * np.ones((psf_max_size))
        psf_frame = -1.0 * np.ones((psf_max_size))

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
        Class to create _data frames and blurring frames used to convolve a psf with a 1D _data of non-masked \
        values.

        Parameters
        ----------
        mask : Mask
            The _data mask, where True eliminates data.
        blurring_mask : Mask
            A mask of pixels outside the mask but whose light blurs into it after PSF convolution.
        psf : _data.PSF or ndarray
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
                    image_frame_indexes, image_frame_psfs = self.frame_at_coords_jit((x, y), mask,
                                                                                     self.mask_index_array, self.psf)
                    self.blurring_frame_indexes[image_index, :] = image_frame_indexes
                    self.blurring_frame_psfs[image_index, :] = image_frame_psfs
                    self.blurring_frame_lengths[image_index] = image_frame_indexes[image_frame_indexes >= 0].shape[0]
                    image_index += 1

    def convolve_image(self, image_array, blurring_array):
        """For a given 1D _data array and blurring array, convolve the two using this convolver.

        Parameters
        -----------
        image_array : ndarray
            1D array of the _data values which are to be blurred with the convolver's PSF.
        blurring_array : ndarray
            1D array of the blurring _data values which blur into the _data-array after PSF convolution.
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
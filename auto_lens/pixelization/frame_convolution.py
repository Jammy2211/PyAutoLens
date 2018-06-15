import numpy as np
from auto_lens import exc

"""
This module is for the application of convolution to sparse_grid vectors.

Take a simple mask:

[[True, False, True],
 [False, False, False],
 [True, False, True]]
 
Here True means that the value is masked.

A set of values in a corresponding image_grid might be represented in a 1D array:

[2, 8, 2, 5, 7, 5, 3, 1, 4]

This module allows us to find the relationships between data_to_pixel in a mask for a kernel of a given size so that
convolutions can be efficiently applied to reduced arrays such as the one above.

A FrameMaker can be created for a given mask:

frame_maker = FrameMaker(mask)

This can then produce a convolver for any given kernel shape and corresponding blurring region mask:

convolver = frame_maker.convolver_for_kernel_shape((3, 3), blurring_region_mask)

Here the blurring region mask describes the region under the mask from which a given PSF kernel may blur pixels. If the
regular mask specifies True for a given pixel and the blurring region mask False then that pixel will be blurred in
using the blurring array.

A convolver can then be made for any given kernel:

kernel_convolver = convolver.convolver_for_kernel(kernel)

Which is applied to a reduced array and blurring array:

convolved_array = convolver.convolve_array(array, blurring_array)

The array is pixels within the non-masked region, whilst the blurring array is pixels outside of the non-masked region
but inside of the blurring mask region.

The convolver can also be applied for some sub_grid-shape of the kernel:

convolved_vector = convolver.convolve_vector(vector, sub_shape=(3, 3))

Or applied to a whole mapping matrix:

convolved_mapping_matrix = convolver.convolve_mapping_matrix(mapping_matrix)

Where the mapping matrix is an array of dictionaries with each index of the array corresponding to a source pixel.

It is also possible to specify a blurring region mask:

FrameMaker(mask, blurring_region_mask)

PSF will be calculated from values that are masked by mask but not by blurring region mask. That is, any entry with a 
True value for mask and a False value for blurring region mask may contribute to the PSF convolved value of a nearby
entry with a False value for mask.

"""


class FrameMaker(object):
    def __init__(self, mask):
        """
        Class to create number array and frames used in 1D convolution
        Parameters
        ----------
        mask: ndarray
                A mask where 0 eliminates data
        """
        self.mask = mask

        self.number_array = np.full(self.mask.shape, -1)

        count = 0

        for x in range(self.mask.shape[0]):
            for y in range(self.mask.shape[1]):
                if not self.mask[x, y]:
                    self.number_array[x, y] = count
                    count += 1

    def make_frame_array(self, kernel_shape):
        """
        Parameters
        ----------
            An array in which non-masked elements have been numbered 0, 1, 2,...N
        kernel_shape: (int, int)
            The shape of the kernel for which frames will be used
        Returns
        -------
        frame_array: [ndarray]
            A list of frames where the position of a frame corresponds to the number at the centre of that frame
        """
        if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0:
            raise exc.KernelException("Kernel must be odd")
        frame_array = []

        # TODO: How would I avoid using for-loops here?
        for x in range(self.number_array.shape[0]):
            for y in range(self.number_array.shape[1]):
                if not self.mask[x][y]:
                    frame_array.append(self.frame_at_coords((x, y), kernel_shape))

        return frame_array

    def make_blurring_frame_array(self, kernel_shape, blurring_region_mask=None):
        """
        Parameters
        ----------
        kernel_shape: (int, int)
            The shape of the kernel
        blurring_region_mask: Mask
            A mask describing the boundary of the region from which values may be convoluted

        Returns
        -------
        blurring_frame_array [ndarray]
            A list of frames where the position corresponds to a position in the blurring region data grid and the
            entries correspond to positions in the primary data grid
        """
        if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0:
            raise exc.KernelException("Kernel must be odd")
        if blurring_region_mask is not None and self.mask.shape != blurring_region_mask.shape:
            raise AssertionError("mask and blurring_region_mask must have the same shape")

        frame_array = []
        for x in range(self.mask.shape[0]):
            for y in range(self.mask.shape[1]):
                if self.mask[x][y] and (blurring_region_mask is None or not blurring_region_mask[x, y]):
                    frame = self.frame_at_coords((x, y), kernel_shape)
                    frame_array.append(frame)
        return frame_array

    def frame_at_coords(self, coords, kernel_shape):
        """
        Parameters
        ----------
        coords: (int, int)
            The image_grid of number_array on which the frame should be centred
        kernel_shape: (int, int)
            The shape of the kernel for which this frame will be used
        Returns
        -------
        frame: ndarray
            A subset of number_array of shape kernel_shape where elements with image_grid outside of frame_array have
            value -1
        """
        half_x = int(kernel_shape[0] / 2)
        half_y = int(kernel_shape[1] / 2)

        frame = np.full((kernel_shape[0] * kernel_shape[1],), -1)

        for i in range(kernel_shape[0]):
            for j in range(kernel_shape[1]):
                x = coords[0] - half_x + i
                y = coords[1] - half_y + j
                if 0 <= x < self.number_array.shape[0] and 0 <= y < self.number_array.shape[1]:
                    value = self.number_array[x, y]
                    if value >= 0 and not self.mask[x, y]:
                        frame[j + kernel_shape[1] * i] = value

        return frame

    def convolver_for_kernel_shape(self, kernel_shape, blurring_region_mask=None):
        """
        Create a convolver that can be used to apply a kernel of any shape to a 1D vector of non-masked values
        Parameters
        ----------
        blurring_region_mask: Mask
            A mask describing the blurring region. If False then that pixel is included int he blurring region.
        kernel_shape: (int, int)
            The shape of the kernel
        Returns
        -------
            convolver: Convolver
        """
        return Convolver(self.make_frame_array(kernel_shape),
                         self.make_blurring_frame_array(
                             kernel_shape,
                             blurring_region_mask) if blurring_region_mask is not None else None)


class Convolver(object):
    def __init__(self, frame_array, blurring_frame_array):
        """
        Class to convolve a kernel with a 1D vector of non-masked values
        Parameters
        ----------
        blurring_frame_array: [ndarray]
            An array of frames created by the frame maker. Maps positions in the kernel to values in the 1D vector for
            masked pixels.
        frame_array: [ndarray]
            An array of frames created by the frame maker. A frame maps positions in the kernel to values in the 1D
            vector.
        """
        self.frame_array = frame_array
        self.blurring_frame_array = blurring_frame_array

    def convolver_for_kernel(self, kernel):
        """
        Parameters
        ----------
        kernel: ndarray
            An array representing a kernel

        Returns
        -------
        convolver: KernelConvolver
            An object used to convolve images
        """
        return KernelConvolver(kernel, self.frame_array, self.blurring_frame_array)


class KernelConvolver(object):
    def __init__(self, kernel, frame_array, blurring_frame_array=None):
        self.shape = kernel.shape

        self.length = self.shape[0] * self.shape[1]
        self.kernel = kernel.flatten()
        self.frame_array = frame_array
        self.blurring_frame_array = blurring_frame_array

    def convolve_1d_array(self, array, blurring_array):
        """
        Simple version of function that applies this convolver to a whole mapping matrix.

        Parameters
        ----------
        blurring_array: [Float]
            An array representing the mapping of a source pixel to a set of image pixels within the blurring region.
        array: [float]
            An array representing the mapping of a source pixel to a set of image pixels.

        Returns
        -------
        convolved_array: [float]
            A matrix representing the mapping of source data_to_pixel to image_grid data_to_pixel accounting for
            convolution
        """
        return map(self.convolve_array, array, blurring_array)

    def convolve_array(self, pixel_array, blurring_array=None, sub_shape=None):
        """
        Parameters
        ----------
        blurring_array: [Float]
            An array representing the mapping of a source pixel to a set of image pixels within the blurring region.
        sub_shape: (int, int)
            Defines a sub_grid-region of the kernel for which the result should be calculated
        pixel_array: [float]
            A 1D array
        Returns
        -------
        convolved_vector: [float]
            A vector convolved with the kernel
        """

        new_array = np.zeros(pixel_array.shape)

        for pixel_index in range(len(pixel_array)):
            frame = self.frame_array[pixel_index]
            value = pixel_array[pixel_index]

            if value > 0:
                new_array = self.convolution_for_value_frame_and_new_array(value, frame, new_array, sub_shape)

        if blurring_array is not None:
            for pixel_index in range(len(blurring_array)):
                frame = self.blurring_frame_array[pixel_index]
                value = blurring_array[pixel_index]

                if value > 0:
                    new_array = self.convolution_for_value_frame_and_new_array(value, frame, new_array, sub_shape)

        return new_array

    def convolution_for_value_frame_and_new_array(self, value, frame, new_array, sub_shape=None):
        """
        Convolves a value with the kernel and populates a new array according to the entries in the frame

        Parameters
        ----------
        value: float
            Some value
        frame: ndarray
            An array describing which entries in the new array convolved values should be inserted into
        new_array: ndarray
            An array into which convolved values are inserted
        sub_shape: (int, int)
            The shape of a reduced size kernel

        Returns
        -------
        new_array: ndarray
            The array with values convolved into it
        """

        limits = None
        if sub_shape is not None:
            limits = calculate_limits(self.shape, sub_shape)

        for kernel_index in range(self.length):
            if sub_shape is not None and not is_in_sub_shape(kernel_index, limits, self.shape):
                continue

            vector_index = frame[kernel_index]

            if vector_index == -1:
                continue
            result = value * self.kernel[kernel_index]
            if result > 0:
                new_array[vector_index] += result

        return new_array


def calculate_limits(shape, sub_shape):
    """
    Finds limits from a shape and subshape for calculation of subsize kernel convolutions
    Parameters
    ----------
    shape: (int, int)
        The shape of the kernel
    sub_shape: (int, int)
        The shape of the subkernel to be considered

    Returns
    -------

    """
    lower_x = (shape[0] - sub_shape[0]) / 2
    lower_y = (shape[1] - sub_shape[1]) / 2
    upper_x = shape[0] - lower_x
    upper_y = shape[1] - lower_y
    return lower_x, lower_y, upper_x, upper_y


def is_in_sub_shape(kernel_index_1d, limits, shape):
    # """
    # Determines if a particular index is within given limits inside of a given shape
    # Parameters
    # ----------
    # kernel_index_1d: int
    #     The index in a flattened kernel
    # limits: Tuple[int, int, int, int]
    #     x_min, y_min, x_max, y_max limits
    # shape: (int, int)
    #     The shape of the kernel
    #
    # Returns
    # -------
    #
    # """
    return limits[1] <= kernel_index_1d / \
           shape[0] < limits[3] and limits[0] <= kernel_index_1d % shape[0] < shape[0] - limits[1]

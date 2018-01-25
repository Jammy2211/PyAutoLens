import numpy as np

"""
This module is for the application of convolution to sparse vectors.

Take a simple mask:

[[0, 1, 0],
 [1, 1, 1],
 [0, 1, 0]]

A set of values in a corresponding image might be represented in a 1D array:

[2, 8, 2, 5, 7, 5, 3, 1, 4]

However, values that are masked out need not be considered. Dropping masked values from this array gives:

[8, 5, 7, 5, 1]

This module allows us to find the relationships between pixels in a mask for a kernel of a given size so that
convolutions can be efficiently applied to reduced arrays such as the one above.

A FrameMaker can be created for a given mask:

frame_maker = FrameMaker(mask)

This can then produce a convolver for any given kernel shape:

convolver = frame_maker.convolver_for_kernel_shape((3, 3))

A convolver can then be made for any given kernel:

kernel_convolver = convolver.convolver_for_kernel(kernel)

Which is applied to a reduced vector:

convolved_vector = convolver.convolve_vector(vector)

"""


class KernelException(Exception):
    pass


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
        self.__number_array = None

    @property
    def number_array(self):
        """
        Creates an array where points inside the mask are numbered
        Parameters

        Returns
        -------
        number_array: ndarray
            An array where non-masked elements are numbered 0, 1, 2,...N with masked elements designated -1
        """
        if self.__number_array is None:
            self.__number_array = -1 * np.ones(self.mask.shape, dtype=np.int64)
            n = 0
            for x in range(self.mask.shape[0]):
                for y in range(self.mask.shape[1]):
                    if self.mask[x, y] == 1:
                        self.__number_array[x, y] = n
                        n += 1
        return self.__number_array

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
            raise KernelException("Kernel must be odd")
        frame_array = []
        for x in range(self.number_array.shape[0]):
            for y in range(self.number_array.shape[1]):
                if self.number_array[x][y] > -1:
                    frame_array.append(self.frame_at_coords((x, y), kernel_shape))

        return frame_array

    def frame_at_coords(self, coords, kernel_shape):
        """
        Parameters
        ----------
        coords: (int, int)
            The coordinates of number_array on which the frame should be centred
        kernel_shape: (int, int)
            The shape of the kernel for which this frame will be used
        Returns
        -------
        frame: ndarray
            A subset of number_array of shape kernel_shape where elements with coordinates outside of frame_array have
            value -1
        """
        half_x = int(kernel_shape[0] / 2)
        half_y = int(kernel_shape[1] / 2)

        frame = {}

        for i in range(kernel_shape[0]):
            for j in range(kernel_shape[1]):
                x = coords[0] - half_x + i
                y = coords[1] - half_y + j
                if 0 <= x < self.number_array.shape[0] and 0 <= y < self.number_array.shape[1]:
                    value = self.number_array[x, y]
                    if value >= 0:
                        frame[j + kernel_shape[1] * i] = value

        return frame

    def convolver_for_kernel_shape(self, kernel_shape):
        """
        Create a convolver that can be used to apply a kernel of any shape to a 1D vector of non-masked values
        Parameters
        ----------
        kernel_shape: (int, int)
            The shape of the kernel
        Returns
        -------
            convolver: Convolver
        """
        return Convolver(self.make_frame_array(kernel_shape))


class Convolver(object):
    def __init__(self, frame_array):
        """
        Class to convolve a kernel with a 1D vector of non-masked values
        Parameters
        ----------
        frame_array: [ndarray]
            An array of frames created by the frame maker. A frame maps positions in the kernel to values in the 1D
            vector.
        """
        self.frame_array = frame_array

    def convolver_for_kernel(self, kernel):
        return KernelConvolver(self.frame_array, kernel)


class KernelConvolver(object):
    def __init__(self, frame_array, kernel):
        self.shape = kernel.shape
        self.kernel = kernel.flatten()
        self.frame_array = frame_array
        self.__result_dict = {}

    def convolve_vector(self, pixel_dict, sub_shape=None):
        """
        Convolves a kernel with a 1D vector of non-masked values
        Parameters
        ----------
        sub_shape: (int, int)
            Defines a subregion of the kernel for which the result should be calculated
        pixel_dict: [int: float]
            A dictionary that maps image pixel indices to values
        Returns
        -------
        convolved_vector: [float]
            A vector convolved with the kernel
        """

        # noinspection PyUnresolvedReferences
        result = {}
        for key in pixel_dict.keys():
            new_dict = self.convolution_for_pixel_index_vector(key, pixel_dict, sub_shape)
            for new_key in new_dict.keys():
                if new_key in result:
                    result[new_key] += new_dict[new_key]
                else:
                    result[new_key] = new_dict[new_key]

        return result

    def result_for_value_and_index(self, value, index):
        if value not in self.__result_dict:
            self.__result_dict[value] = {}
        if index not in self.__result_dict[value]:
            self.__result_dict[value][index] = value * self.kernel[index]
        return self.__result_dict[value][index]

    def convolution_for_pixel_index_vector(self, pixel_index, pixel_dict, sub_shape=None):
        """
        Creates a vector of values describing the convolution of the kernel with a value in the vector
        Parameters
        ----------
        sub_shape: (int, int)
            Defines a subregion of the kernel for which the result should be calculated
        pixel_index: int
            The index in the vector to be convolved
        pixel_dict: [int: float]
            A dictionary that maps image pixel indices to values
        Returns
        -------
        convolution_dict: [int: float]
            A dictionary with values populated according to the convolution of the kernel
            with one particular value
        """

        # noinspection PyUnresolvedReferences
        new_dict = {}

        value = pixel_dict[pixel_index]

        frame = self.frame_array[pixel_index]

        keys = frame.keys()

        if sub_shape is not None:
            limits = calculate_limits(self.shape, sub_shape)

            keys = filter(lambda index: is_in_sub_shape(index, limits, self.shape), keys)

        for kernel_index in keys:
            vector_index = frame[kernel_index]
            result = self.result_for_value_and_index(value, kernel_index)
            if result > 0:
                new_dict[vector_index] = result

        return new_dict


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
    """
    Determines if a particular index is within given limits inside of a given shape
    Parameters
    ----------
    kernel_index_1d: int
        The index in a flattened kernel
    limits: Tuple[int, int, int, int]
        x_min, y_min, x_max, y_max limits
    shape: (int, int)
        The shape of the kernel

    Returns
    -------

    """
    return limits[1] <= kernel_index_1d / shape[0] < limits[3] and limits[0] <= kernel_index_1d % shape[0] < shape[0] - \
                                                                                                             limits[1]

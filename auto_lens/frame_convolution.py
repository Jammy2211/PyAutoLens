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

Which is then applied to a reduced array and kernel to find the convolution efficiently:

convolved_vector = convolver.convolve_vector_with_kernel(vector, kernel)

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
            self.__number_array = -1 * np.ones(self.mask.shape)
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

        frame = -1 * np.ones(kernel_shape)

        for i in range(kernel_shape[0]):
            for j in range(kernel_shape[1]):
                x = coords[0] - half_x + i
                y = coords[1] - half_y + j
                if 0 <= x < self.number_array.shape[0] and 0 <= y < self.number_array.shape[1]:
                    frame[i, j] = self.number_array[x, y]

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


# TODO: KernelConvolver with value to convolution result map


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
        if frame_array[0].shape != kernel.shape:
            raise AssertionError(
                "Frame {} and kernel {} shapes do not match".format(frame_array[0].shape, kernel.shape))
        self.frame_array = frame_array
        self.kernel = kernel

    def convolve_vector(self, vector):
        """
        Convolves a kernel with a 1D vector of non-masked values
        Parameters
        ----------
        vector: [float]
            A vector of numbers excluding those that are masked
        Returns
        -------
        convolved_vector: [float]
            A vector convolved with the kernel
        """

        # noinspection PyUnresolvedReferences
        result = np.zeros(len(vector))
        for index in range(len(vector)):
            # noinspection PyUnresolvedReferences
            result = np.add(result, self.convolution_for_pixel_index_vector(index, vector))
        return result

    def convolution_for_pixel_index_vector(self, pixel_index, vector):
        """
        Creates a vector of values describing the convolution of the kernel with a value in the vector
        Parameters
        ----------
        pixel_index: int
            The index in the vector to be convolved
        vector: [float]
            A vector of numbers excluding those that are masked
        Returns
        -------
        convolution_array: [float]
            An array with the same length of the vector with values populated according to the convolution of the kernel
            with one particular value
        """

        # noinspection PyUnresolvedReferences
        new_vector = np.zeros(len(vector))

        value = vector[pixel_index]

        if value == 0:
            return new_vector

        frame = self.frame_array[pixel_index]
        result = value * self.kernel
        for x in range(frame.shape[0]):
            for y in range(frame.shape[1]):
                vector_index = frame[x, y]
                if vector_index > -1:
                    new_vector[int(vector_index)] = result[x, y]
        return new_vector

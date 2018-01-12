import numpy as np


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


class Convolver(object):
    def __init__(self, frame_array, number_array):
        self.frame_array = frame_array
        self.number_array = number_array.flatten()

    def convolve_vector_with_kernel(self, vector, kernel):
        if self.frame_array[0].shape != kernel.shape:
            raise AssertionError(
                "Frame {} and kernel {} shapes do not match".format(self.frame_array[0].shape, kernel.shape))
        # noinspection PyUnresolvedReferences
        result = np.zeros(len(vector))
        for index in range(len(vector)):
            # noinspection PyUnresolvedReferences
            result = np.add(result, self.convolution_for_pixel_index_vector_and_kernel(index, vector, kernel))
        return result

    def is_frame_for_index(self, pixel_index):
        return self.number_array[pixel_index] > -1

    def frame_for_index(self, pixel_index):
        return self.frame_array[self.number_array[pixel_index]]

    def convolution_for_pixel_index_vector_and_kernel(self, pixel_index, vector, kernel):
        # noinspection PyUnresolvedReferences
        new_vector = np.zeros(len(vector))

        if self.is_frame_for_index(pixel_index):
            value = vector[pixel_index]
            frame = self.frame_for_index(pixel_index)
            result = value * kernel
            for x in range(frame.shape[0]):
                for y in range(frame.shape[1]):
                    vector_index = frame[x, y]
                    if vector_index > -1:
                        new_vector[int(vector_index)] = result[x, y]
        return new_vector

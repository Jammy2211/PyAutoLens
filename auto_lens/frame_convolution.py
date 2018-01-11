import numpy as np


def number_array_for_mask(mask):
    array = -1 * np.ones(mask.shape)
    n = 0
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x, y] == 1:
                array[x, y] = n
                n += 1
    return array


def make_frame_dict(number_array, kernel_shape):
    frame_dict = {}
    for x in range(number_array.shape[0]):
        for y in range(number_array.shape[1]):
            number = number_array[x][y]
            if number > -1:
                frame_dict[number] = []

    return frame_dict

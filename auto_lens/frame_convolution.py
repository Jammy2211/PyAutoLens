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


def make_frame_array(number_array, kernel_shape):
    frame_array = []
    for x in range(number_array.shape[0]):
        for y in range(number_array.shape[1]):
            if number_array[x][y] > -1:
                frame_array.append(frame_at_coords(number_array, (x, y), kernel_shape))

    return frame_array


def frame_at_coords(number_array, coords, kernel_shape):
    half_x = int(kernel_shape[0] / 2)
    half_y = int(kernel_shape[1] / 2)

    frame = -1 * np.ones(kernel_shape)

    for i in range(kernel_shape[0]):
        for j in range(kernel_shape[1]):
            x = coords[0] - half_x + i
            y = coords[1] - half_y + j
            if 0 <= x < number_array.shape[0] and 0 <= y < number_array.shape[1]:
                frame[i, j] = number_array[x, y]

    return frame

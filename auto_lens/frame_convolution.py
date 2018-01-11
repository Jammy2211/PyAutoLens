import numpy as np


def number_array_for_mask(mask):
    """
    Creates an array where points inside the mask are numbered
    Parameters
    ----------
    mask: ndarray
        A mask where 0 eliminates data

    Returns
    -------
    number_array: ndarray
        An array where non-masked elements are numbered 0, 1, 2,...N with masked elements designated -1
    """
    array = -1 * np.ones(mask.shape)
    n = 0
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x, y] == 1:
                array[x, y] = n
                n += 1
    return array


def make_frame_array(number_array, kernel_shape):
    """
    Parameters
    ----------
    number_array: ndarray
        An array in which non-masked elements have been numbered 0, 1, 2,...N
    kernel_shape: (int, int)
        The shape of the kernel for which frames will be used
    Returns
    -------
    frame_array: [ndarray]
        A list of frames where the position of a frame corresponds to the number at the centre of that frame
    """
    frame_array = []
    for x in range(number_array.shape[0]):
        for y in range(number_array.shape[1]):
            if number_array[x][y] > -1:
                frame_array.append(frame_at_coords(number_array, (x, y), kernel_shape))

    return frame_array


def frame_at_coords(number_array, coords, kernel_shape):
    """
    Parameters
    ----------
    number_array: ndarray
        An array in which non-masked elements have been numbered 0, 1, 2,...N
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
            if 0 <= x < number_array.shape[0] and 0 <= y < number_array.shape[1]:
                frame[i, j] = number_array[x, y]

    return frame

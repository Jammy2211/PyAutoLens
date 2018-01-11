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

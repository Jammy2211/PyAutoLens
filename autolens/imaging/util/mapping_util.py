import logging

import numba
import numpy as np

from autolens.imaging.util import mask_util

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG

@numba.jit(nopython=True, cache=True)
def map_1d_indexes_to_2d_indexes_for_shape(indexes_1d, shape):

    indexes_2d = np.zeros((indexes_1d.shape[0], 2))

    for i, index_1d in enumerate(indexes_1d):

        indexes_2d[i,0] = int(index_1d / shape[1])
        indexes_2d[i,1] = int(index_1d % shape[1])

    return indexes_2d

@numba.jit(nopython=True, cache=True)
def map_2d_indexes_to_1d_indexes_for_shape(indexes_2d, shape):

    indexes_1d = np.zeros(indexes_2d.shape[0])

    for i in range(indexes_2d.shape[0]):

        indexes_1d[i] = int((indexes_2d[i,0])*shape[1] + indexes_2d[i,1])

    return indexes_1d

@numba.jit(nopython=True, cache=True)
def sub_to_image_from_mask(mask, sub_grid_size):
    """Compute a 1D array that maps every unmasked pixel's sub-pixel to its corresponding 1d datas_-pixel.

    For howtolens, if sub-pixel 8 is in datas_-pixel 1, sub_to_image[7] = 1."""

    total_sub_pixels = mask_util.total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size)

    sub_to_image = np.zeros(shape=total_sub_pixels)
    image_index = 0
    sub_index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                for y1 in range(sub_grid_size):
                    for x1 in range(sub_grid_size):
                        sub_to_image[sub_index] = image_index
                        sub_index += 1

                image_index += 1

    return sub_to_image

@numba.jit(nopython=True, cache=True)
def map_2d_array_to_masked_1d_array_from_array_2d_and_mask(mask, array_2d):
    """For a given 2D array and masks, mappers all unmasked pixels to a 1D array."""

    total_image_pixels = mask_util.total_image_pixels_from_mask(mask)

    array_1d = np.zeros(shape=total_image_pixels)
    index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                array_1d[index] = array_2d[y, x]
                index += 1

    return array_1d

@numba.jit(nopython=True, cache=True)
def map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(array_1d, shape, one_to_two):
    """For a masked 1D array, mappers it to a 2D array using the mappings from 1D to 2D (one_to_two)."""

    array_2d = np.zeros(shape)

    for index in range(len(one_to_two)):
        array_2d[one_to_two[index, 0], one_to_two[index, 1]] = array_1d[index]

    return array_2d

@numba.jit(nopython=True, cache=True)
def map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d, shape):
    """Map a 1D array where every value in its original 2D array was unmasked, to this original 2D array."""

    array_2d = np.zeros(shape)

    index = 0
    for y in range(shape[0]):
        for x in range(shape[1]):
            array_2d[y, x] = array_1d[index]
            index += 1

    return array_2d

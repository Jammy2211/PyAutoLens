import logging

import numpy as np

from autolens import decorator_util
from autolens.array.util import mask_util
from autolens.array.mapping_util import mask_mapping_util

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


@decorator_util.jit()
def map_1d_indexes_to_2d_indexes_for_shape(indexes_1d, shape):
    """For pixels on a 2D array of shape (rows, columns), map an array of 1D pixel indexes to 2D pixel indexes.

    Indexing is defined from the top-left corner rightwards and downwards, whereby the top-left pixel on the 2D array
    corresponds to index 0, the pixel to its right pixel 1, and so on.

    For a 2D array of shape (3,3), 1D pixel indexes are converted as follows:

    - 1D Pixel index 0 maps -> 2D pixel index [0,0].
    - 1D Pixel index 1 maps -> 2D pixel index [0,1].
    - 1D Pixel index 4 maps -> 2D pixel index [1,0].
    - 1D Pixel index 8 maps -> 2D pixel index [2,2].

    Parameters
     ----------
    indexes_1d : ndarray
        The 1D pixel indexes which are mapped to 2D indexes.
    shape : (int, int)
        The shape of the 2D array which the pixels are defined on.

    Returns
    --------
    ndarray
        An array of 2d pixel indexes with dimensions (total_indexes, 2).

    Examples
    --------
    indexes_1d = np.array([0, 1, 2, 5])
    indexes_2d = map_1d_indexes_to_2d_indexes_for_shape(indexes_1d=indexes_1d, shape=(3,3))
    """
    indexes_2d = np.zeros((indexes_1d.shape[0], 2))

    for i, index_1d in enumerate(indexes_1d):
        indexes_2d[i, 0] = int(index_1d / shape[1])
        indexes_2d[i, 1] = int(index_1d % shape[1])

    return indexes_2d


@decorator_util.jit()
def map_2d_indexes_to_1d_indexes_for_shape(indexes_2d, shape):
    """For pixels on a 2D array of shape (rows, colums), map an array of 2D pixel indexes to 1D pixel indexes.

    Indexing is defined from the top-left corner rightwards and downwards, whereby the top-left pixel on the 2D array
    corresponds to index 0, the pixel to its right pixel 1, and so on.

    For a 2D array of shape (3,3), 2D pixel indexes are converted as follows:

    - 2D Pixel index [0,0] maps -> 1D pixel index 0.
    - 2D Pixel index [0,1] maps -> 2D pixel index 1.
    - 2D Pixel index [1,0] maps -> 2D pixel index 4.
    - 2D Pixel index [2,2] maps -> 2D pixel index 8.

    Parameters
     ----------
    indexes_2d : ndarray
        The 2D pixel indexes which are mapped to 1D indexes.
    shape : (int, int)
        The shape of the 2D array which the pixels are defined on.

    Returns
    --------
    ndarray
        An array of 1d pixel indexes with dimensions (total_indexes).

    Examples
    --------
    indexes_2d = np.array([[0,0], [1,0], [2,0], [2,2]])
    indexes_1d = map_1d_indexes_to_1d_indexes_for_shape(indexes_2d=indexes_2d, shape=(3,3))
    """
    indexes_1d = np.zeros(indexes_2d.shape[0])

    for i in range(indexes_2d.shape[0]):
        indexes_1d[i] = int((indexes_2d[i, 0]) * shape[1] + indexes_2d[i, 1])

    return indexes_1d


@decorator_util.jit()
def sub_array_1d_from_sub_array_2d_mask_and_sub_size(sub_array_2d, mask, sub_size):
    """For a 2D sub array and mask, map the values of all unmasked pixels to a 1D sub-array.

    A sub-array is an array whose dimensions correspond to the hyper array (e.g. used to make the grid) \
    multiplid by the sub_size. E.g., it is an array that would be generated using the sub-grid and not binning \
    up values in sub-pixels back to the grid.

    The pixel coordinate origin is at the top left corner of the 2D array and goes right-wards and downwards,
    with sub-pixels then going right and downwards in each pixel. For example, for an array of shape (3,3) and a \
    sub-grid size of 2 where all pixels are unmasked:

    - pixel [0,0] of the 2D array will correspond to index 0 of the 1D array.
    - pixel [0,1] of the 2D array will correspond to index 1 of the 1D array.
    - pixel [1,0] of the 2D array will correspond to index 2 of the 1D array.
    - pixel [2,0] of the 2D array will correspond to index 4 of the 1D array.
    - pixel [1,0] of the 2D array will correspond to index 12 of the 1D array.

    Parameters
    ----------
    sub_array_2d : ndarray
        A 2D array of values on the dimensions of the sub-grid.
    mask : ndarray
        A 2D array of bools, where *False* values mean unmasked and are included in the mapping_util.
    array_2d : ndarray
        The 2D array of values which are mapped to a 1D array.

    Returns
    --------
    ndarray
        A 1D array of values mapped from the 2D array with dimensions (total_unmasked_pixels).

    Examples
    --------

    sub_array_2d = np.array([[ 1.0,  2.0,  5.0,  6.0],
                             [ 3.0,  4.0,  7.0,  8.0],
                             [ 9.0, 10.0, 13.0, 14.0],
                             [11.0, 12.0, 15.0, 16.0])

    mask = np.array([[True, False],
                     [False, False]])

    sub_array_1d = map_sub_array_2d_to_masked_sub_array_1d_from_sub_array_2d_mask_and_sub_size( \
        mask=mask, array_2d=array_2d)
    """

    total_sub_pixels = mask_util.total_sub_pixels_from_mask_and_sub_size(
        mask=mask, sub_size=sub_size
    )

    sub_array_1d = np.zeros(shape=total_sub_pixels)
    index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                for y1 in range(sub_size):
                    for x1 in range(sub_size):
                        sub_array_1d[index] = sub_array_2d[
                            y * sub_size + y1, x * sub_size + x1
                        ]
                        index += 1

    return sub_array_1d


def sub_array_2d_from_sub_array_1d_mask_and_sub_size(sub_array_1d, mask, sub_size):
    """For a 1D array that was computed by mapping_util unmasked values from a 2D array of shape (rows, columns), map its \
    values back to the original 2D array where masked values are set to zero.

    This uses a 1D array 'one_to_two' where each index gives the 2D pixel indexes of the 1D array's unmasked pixels, \
    for example:

    - If one_to_two[0] = [0,0], the first value of the 1D array maps to the pixel [0,0] of the 2D array.
    - If one_to_two[1] = [0,1], the second value of the 1D array maps to the pixel [0,1] of the 2D array.
    - If one_to_two[4] = [1,1], the fifth value of the 1D array maps to the pixel [1,1] of the 2D array.

    Parameters
     ----------
    sub_array_1d : ndarray
        The 1D array of values which are mapped to a 2D array.
    shape : (int, int)
        The shape of the 2D array which the pixels are defined on.
    sub_one_to_two : ndarray
        An array describing the 2D array index that every 1D array index maps too.

    Returns
    --------
    ndarray
        A 2D array of values mapped from the 1D array with dimensions shape.

    Examples
    --------
    one_to_two = np.array([[0,1], [1,0], [1,1], [1,2], [2,1]])

    array_1d = np.array([[2.0, 4.0, 5.0, 6.0, 8.0])

    array_2d = map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two( \
        array_1d=array_1d, shape=(3,3), one_to_two=one_to_two)
    """

    sub_shape = (mask.shape[0] * sub_size, mask.shape[1] * sub_size)

    sub_one_to_two = mask_mapping_util.sub_mask_1d_index_to_submask_index_from_mask_and_sub_size(
        mask=mask, sub_size=sub_size
    ).astype(
        "int"
    )

    return sub_array_2d_from_sub_array_1d_sub_shape_and_sub_mask_1d_index_to_submask_index(
        sub_array_1d=sub_array_1d,
        sub_shape=sub_shape,
        sub_mask_1d_index_to_submask_index=sub_one_to_two,
    )


@decorator_util.jit()
def sub_array_2d_from_sub_array_1d_sub_shape_and_sub_mask_1d_index_to_submask_index(
    sub_array_1d, sub_shape, sub_mask_1d_index_to_submask_index
):

    array_2d = np.zeros(sub_shape)

    for index in range(len(sub_mask_1d_index_to_submask_index)):
        array_2d[
            sub_mask_1d_index_to_submask_index[index, 0],
            sub_mask_1d_index_to_submask_index[index, 1],
        ] = sub_array_1d[index]

    return array_2d

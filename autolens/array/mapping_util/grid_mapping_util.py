import logging

import numpy as np

from autolens.array.mapping_util import array_mapping_util

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


def sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(sub_grid_2d, mask, sub_size):
    """For a 2D grid and mask, map the values of all unmasked pixels to a 1D grid.

    The pixel coordinate origin is at the top left corner of the 2D grid and goes right-wards and downwards, such \
    that for an grid of shape (3,3) where all pixels are unmasked:

    - pixel [0,0] of the 2D grid will correspond to index 0 of the 1D grid.
    - pixel [0,1] of the 2D grid will correspond to index 1 of the 1D grid.
    - pixel [1,0] of the 2D grid will correspond to index 4 of the 1D grid.

    Parameters
     ----------
    mask : ndgrid
        A 2D grid of bools, where *False* values are unmasked and included in the mapping_util.
    sub_grid_2d : ndgrid
        The 2D grid of values which are mapped to a 1D grid.

    Returns
    --------
    ndgrid
        A 1D grid of values mapped from the 2D grid with dimensions (total_unmasked_pixels).

    Examples
    --------
    mask = np.grid([[True, False, True],
                     [False, False, False]
                     [True, False, True]])

    grid_2d = np.grid([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                        [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
                        [[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]]])

    grid_1d = map_grid_2d_to_masked_grid_1d_from_grid_2d_and_mask(mask=mask, grid_2d=grid_2d)
    """

    sub_grid_1d_y = array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
        sub_array_2d=sub_grid_2d[:, :, 0], mask=mask, sub_size=sub_size
    )

    sub_grid_1d_x = array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
        sub_array_2d=sub_grid_2d[:, :, 1], mask=mask, sub_size=sub_size
    )

    return np.stack((sub_grid_1d_y, sub_grid_1d_x), axis=-1)


def sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(sub_grid_1d, mask, sub_size):
    """For a 1D array that was computed by mapping_util unmasked values from a 2D array of shape (rows, columns), map its \
    values back to the original 2D array where masked values are set to zero.

    This uses a 1D array 'one_to_two' where each index gives the 2D pixel indexes of the 1D array's unmasked pixels, \
    for example:

    - If one_to_two[0] = [0,0], the first value of the 1D array maps to the pixel [0,0] of the 2D array.
    - If one_to_two[1] = [0,1], the second value of the 1D array maps to the pixel [0,1] of the 2D array.
    - If one_to_two[4] = [1,1], the fifth value of the 1D array maps to the pixel [1,1] of the 2D array.

    Parameters
     ----------
    sub_grid_1d : ndarray
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

    array_2d = map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(array_1d=array_1d, shape=(3,3),
                                                                                  one_to_two=one_to_two)
    """

    sub_grid_2d_y = array_mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
        sub_array_1d=sub_grid_1d[:, 0], mask=mask, sub_size=sub_size
    )

    sub_grid_2d_x = array_mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
        sub_array_1d=sub_grid_1d[:, 1], mask=mask, sub_size=sub_size
    )

    return np.stack((sub_grid_2d_y, sub_grid_2d_x), axis=-1)

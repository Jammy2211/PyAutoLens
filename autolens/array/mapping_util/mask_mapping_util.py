import logging

import numpy as np

from autolens import decorator_util
from autolens.array.util import mask_util

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


@decorator_util.jit()
def sub_mask_1d_index_to_mask_1d_index_from_mask(mask, sub_size):
    """"For pixels on a 2D array of shape (rows, colums), compute a 1D array which, for every unmasked pixel on \
    this 2D array, maps the 1D sub-pixel indexes to their 1D pixel indexes.

    For example, for a sub-grid size of 2, the following mappings from sub-pixels to 2D array pixels are:

    - sub_mask_1d_index_to_mask_1d_index[0] = 0 -> The first sub-pixel maps to the first unmasked pixel on the 2D array.
    - sub_mask_1d_index_to_mask_1d_index[3] = 0 -> The fourth sub-pixel maps to the first unmasked pixel on the 2D array.
    - sub_mask_1d_index_to_mask_1d_index[7] = 1 -> The eighth sub-pixel maps to the second unmasked pixel on the 2D array.

    The term 'grid' is used because the grid is defined as the grid of coordinates on the centre of every \
    pixel on the 2D array. Thus, this array maps sub-pixels on a sub-grid to pixels on a grid.


                     [True, False, True]])
    sub_mask_1d_index_to_mask_1d_index = sub_mask_1d_index_to_mask_1d_index_from_mask(mask=mask, sub_size=2)
    """

    total_sub_pixels = mask_util.total_sub_pixels_from_mask_and_sub_size(
        mask=mask, sub_size=sub_size
    )

    sub_mask_1d_index_to_mask_1d_index = np.zeros(shape=total_sub_pixels)
    mask_1d_index = 0
    sub_mask_1d_index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                for y1 in range(sub_size):
                    for x1 in range(sub_size):
                        sub_mask_1d_index_to_mask_1d_index[
                            sub_mask_1d_index
                        ] = mask_1d_index
                        sub_mask_1d_index += 1

                mask_1d_index += 1

    return sub_mask_1d_index_to_mask_1d_index


def mask_1d_index_to_sub_mask_1d_indexes_from_mask(mask, sub_size):
    """"For pixels on a 2D array of shape (rows, colums), compute a 1D array which, for every unmasked pixel on \
    this 2D array, maps the 1D sub-pixel indexes to their 1D pixel indexes.

    For example, for a sub-grid size of 2, the following mappings from sub-pixels to 2D array pixels are:

    - sub_mask_1d_index_to_mask_1d_index[0] = 0 -> The first sub-pixel maps to the first unmasked pixel on the 2D array.
    - sub_mask_1d_index_to_mask_1d_index[3] = 0 -> The fourth sub-pixel maps to the first unmasked pixel on the 2D array.
    - sub_mask_1d_index_to_mask_1d_index[7] = 1 -> The eighth sub-pixel maps to the second unmasked pixel on the 2D array.

    The term 'grid' is used because the grid is defined as the grid of coordinates on the centre of every \
    pixel on the 2D array. Thus, this array maps sub-pixels on a sub-grid to pixels on a grid.


                     [True, False, True]])
    sub_mask_1d_index_to_mask_1d_index = sub_mask_1d_index_to_mask_1d_index_from_mask(mask=mask, sub_size=2)
    """

    total_pixels = mask_util.total_pixels_from_mask(mask=mask)

    mask_1d_index_to_sub_mask_indexes = [[] for _ in range(total_pixels)]

    sub_mask_1d_index_to_mask_1d_index = sub_mask_1d_index_to_mask_1d_index_from_mask(
        mask=mask, sub_size=sub_size
    ).astype("int")

    for sub_mask_1d_index, mask_1d_index in enumerate(
        sub_mask_1d_index_to_mask_1d_index
    ):
        mask_1d_index_to_sub_mask_indexes[mask_1d_index].append(sub_mask_1d_index)

    return mask_1d_index_to_sub_mask_indexes


@decorator_util.jit()
def submask_index_to_sub_mask_1d_index_from_sub_mask(sub_mask):
    """Create a 2D array which maps every False entry of a 2D mask to its 1D mask array index 2D binned mask. Every \
    True entry is given a value -1.

    This is used as a convenience tool for creating arrays mapping_util between different grids and arrays.

    For example, if we had a 3x4:

    [[False, True, False, False],
     [False, True, False, False],
     [False, False, False, True]]]

    The mask_2d_to_mask_1d array would be:

    [[0, -1, 2, 3],
     [4, -1, 5, 6],
     [7, 8, 9, -1]]

    Parameters
    ----------
    sub_mask : ndarray
        The 2D mask that the mapping_util array is created for.

    Returns
    -------
    ndarray
        The 2D array mapping_util 2D mask entries to their 1D masked array indexes.

    Examples
    --------
    mask_2d = np.full(fill_value=False, shape=(9,9))
    sub_two_to_one = mask_2d_to_mask_1d_index_frommask(mask_2d=mask_2d)
    """

    submask_index_to_1d_index = np.full(fill_value=-1, shape=sub_mask.shape)

    sub_mask_1d_index = 0

    for sub_mask_y in range(sub_mask.shape[0]):
        for sub_mask_x in range(sub_mask.shape[1]):
            if sub_mask[sub_mask_y, sub_mask_x] == False:
                submask_index_to_1d_index[sub_mask_y, sub_mask_x] = sub_mask_1d_index
                sub_mask_1d_index += 1

    return submask_index_to_1d_index


@decorator_util.jit()
def sub_mask_1d_index_to_submask_index_from_mask_and_sub_size(mask, sub_size):
    """Compute a 1D array that maps every unmasked sub-pixel to its corresponding 2d pixel using its (y,x) pixel indexes.

    For example, for a sub-grid size of 2, f pixel [2,5] corresponds to the first pixel in the masked 1D array:

    - The first sub-pixel in this pixel on the 1D array is grid_to_pixel[4] = [2,5]
    - The second sub-pixel in this pixel on the 1D array is grid_to_pixel[5] = [2,6]
    - The third sub-pixel in this pixel on the 1D array is grid_to_pixel[5] = [3,5]

    Parameters
    -----------
    mask : ndarray
        A 2D array of bools, where *False* values are unmasked.
    sub_size : int
        The size of the sub-grid in each mask pixel.

    Returns
    --------
    ndarray
        The 2D blurring mask array whose unmasked values (*False*) correspond to where the mask will have PSF light \
        blurred into them.

    Examples
    --------
    mask = np.array([[True, True, True],
                     [True, False, True]
                     [True, True, True]])

    blurring_mask = blurring_mask_from_mask_and_psf_shape(mask=mask, psf_shape=(3,3))

    """

    total_sub_pixels = mask_util.total_sub_pixels_from_mask_and_sub_size(
        mask=mask, sub_size=sub_size
    )
    sub_mask_1d_index_to_submask_index = np.zeros(shape=(total_sub_pixels, 2))
    sub_mask_1d_index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                for y1 in range(sub_size):
                    for x1 in range(sub_size):
                        sub_mask_1d_index_to_submask_index[sub_mask_1d_index, :] = (
                            (y * sub_size) + y1,
                            (x * sub_size) + x1,
                        )
                        sub_mask_1d_index += 1

    return sub_mask_1d_index_to_submask_index

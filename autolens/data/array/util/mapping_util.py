import logging

import numpy as np

from autolens import decorator_util
from autolens.data.array.util import mask_util

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
def sub_to_regular_from_mask(mask, sub_grid_size):
    """"For pixels on a 2D array of shape (rows, colums), compute a 1D array which, for every unmasked pixel on \
    this 2D array, maps the 1D sub-pixel indexes to their 1D pixel indexes.

    For example, for a sub-grid size of 2, the following mappings from sub-pixels to 2D array pixels are:

    - sub_to_regular[0] = 0 -> The first sub-pixel maps to the first unmasked pixel on the regular 2D array.
    - sub_to_regular[3] = 0 -> The fourth sub-pixel maps to the first unmasked pixel on the regular 2D array.
    - sub_to_regular[7] = 1 -> The eighth sub-pixel maps to the second unmasked pixel on the regular 2D array.

    The term 'regular' is used because the regular-grid is defined as the grid of coordinates on the centre of every \
    pixel on the 2D array. Thus, this array maps sub-pixels on a sub-grid to regular-pixels on a regular-grid.


                     [True, False, True]])
    sub_to_regular = sub_to_regular_from_mask(mask=mask, sub_grid_size=2)
    """

    total_sub_pixels = mask_util.total_sub_pixels_from_mask_and_sub_grid_size(mask=mask,
                                                                              sub_grid_size=sub_grid_size)

    sub_to_regular = np.zeros(shape=total_sub_pixels)
    regular_index = 0
    sub_index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                for y1 in range(sub_grid_size):
                    for x1 in range(sub_grid_size):
                        sub_to_regular[sub_index] = regular_index
                        sub_index += 1

                regular_index += 1

    return sub_to_regular


@decorator_util.jit()
def map_array_2d_to_masked_array_1d_from_array_2d_and_mask(mask, array_2d):
    """For a 2D array and mask, map the values of all unmasked pixels to a 1D array.

    The pixel coordinate origin is at the top left corner of the 2D array and goes right-wards and downwards, such
    that for an array of shape (3,3) where all pixels are unmasked:

    - pixel [0,0] of the 2D array will correspond to index 0 of the 1D array.
    - pixel [0,1] of the 2D array will correspond to index 1 of the 1D array.
    - pixel [1,0] of the 2D array will correspond to index 4 of the 1D array.

    Parameters
     ----------
    mask : ndarray
        A 2D array of bools, where *False* values are unmasked and included in the mapping.
    array_2d : ndarray
        The 2D array of values which are mapped to a 1D array.

    Returns
    --------
    ndarray
        A 1D array of values mapped from the 2D array with dimensions (total_unmasked_pixels).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])

    array_2d = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])

    array_1d = map_2d_array_to_masked_1d_array_from_array_2d_and_mask(mask=mask, array_2d=array_2d)
    """

    total_image_pixels = mask_util.total_regular_pixels_from_mask(mask)

    array_1d = np.zeros(shape=total_image_pixels)
    index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                array_1d[index] = array_2d[y, x]
                index += 1

    return array_1d


@decorator_util.jit()
def map_sub_array_2d_to_masked_sub_array_1d_from_sub_array_2d_mask_and_sub_grid_size(
        sub_array_2d, mask, sub_grid_size):
    """For a 2D sub array and mask, map the values of all unmasked pixels to a 1D sub-array.

    A sub-array is an array whose dimensions correspond to the normal array (e.g. used to make the regular grid) \
    multiplid by the sub_grid_size. E.g., it is an array that would be generated using the sub-grid and not binning \
    up values in sub-pixels back to the regular-grid.

    The regular pixel coordinate origin is at the top left corner of the 2D array and goes right-wards and downwards,
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
        A 2D array of bools, where *False* values mean unmasked and are included in the mapping.
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

    sub_array_1d = map_sub_array_2d_to_masked_sub_array_1d_from_sub_array_2d_mask_and_sub_grid_size( \
        mask=mask, array_2d=array_2d)
    """

    total_sub_pixels = mask_util.total_sub_pixels_from_mask_and_sub_grid_size(mask=mask,
                                                                              sub_grid_size=sub_grid_size)

    sub_array_1d = np.zeros(shape=total_sub_pixels)
    index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                for y1 in range(sub_grid_size):
                    for x1 in range(sub_grid_size):
                        sub_array_1d[index] = sub_array_2d[
                            y * sub_grid_size + y1, x * sub_grid_size + x1]
                        index += 1

    return sub_array_1d


@decorator_util.jit()
def map_grid_2d_to_masked_grid_1d_from_grid_2d_and_mask(mask, grid_2d):
    """For a 2D grid and mask, map the values of all unmasked pixels to a 1D grid.

    The pixel coordinate origin is at the top left corner of the 2D grid and goes right-wards and downwards, such \
    that for an grid of shape (3,3) where all pixels are unmasked:

    - pixel [0,0] of the 2D grid will correspond to index 0 of the 1D grid.
    - pixel [0,1] of the 2D grid will correspond to index 1 of the 1D grid.
    - pixel [1,0] of the 2D grid will correspond to index 4 of the 1D grid.

    Parameters
     ----------
    mask : ndgrid
        A 2D grid of bools, where *False* values are unmasked and included in the mapping.
    grid_2d : ndgrid
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

    total_image_pixels = mask_util.total_regular_pixels_from_mask(mask=mask)

    grid_1d = np.zeros(shape=(total_image_pixels, 2))
    index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                grid_1d[index, 0] = grid_2d[y, x, 0]
                grid_1d[index, 1] = grid_2d[y, x, 1]
                index += 1

    return grid_1d


@decorator_util.jit()
def map_masked_array_1d_to_array_2d_from_array_1d_shape_and_one_to_two(array_1d, shape,
                                                                       one_to_two):
    """For a 1D array that was computed by mapping unmasked values from a 2D array of shape (rows, columns), map its \
    values back to the original 2D array where masked values are set to zero.

    This uses a 1D array 'one_to_two' where each index gives the 2D pixel indexes of the 1D array's unmasked pixels, \
    for example:

    - If one_to_two[0] = [0,0], the first value of the 1D array maps to the pixel [0,0] of the 2D array.
    - If one_to_two[1] = [0,1], the second value of the 1D array maps to the pixel [0,1] of the 2D array.
    - If one_to_two[4] = [1,1], the fifth value of the 1D array maps to the pixel [1,1] of the 2D array.

    Parameters
     ----------
    array_1d : ndarray
        The 1D array of values which are mapped to a 2D array.
    shape : (int, int)
        The shape of the 2D array which the pixels are defined on.
    one_to_two : ndarray
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

    array_2d = np.zeros(shape)

    for index in range(len(one_to_two)):
        array_2d[one_to_two[index, 0], one_to_two[index, 1]] = array_1d[index]

    return array_2d


@decorator_util.jit()
def map_masked_1d_grid_to_2d_grid_from_grid_1d_shape_and_one_to_two(grid_1d, shape,
                                                                    one_to_two):
    """For a 1D array that was computed by mapping unmasked values from a 2D array of shape (rows, columns), map its \
    values back to the original 2D array where masked values are set to zero.

    This uses a 1D array 'one_to_two' where each index gives the 2D pixel indexes of the 1D array's unmasked pixels, \
    for example:

    - If one_to_two[0] = [0,0], the first value of the 1D array maps to the pixel [0,0] of the 2D array.
    - If one_to_two[1] = [0,1], the second value of the 1D array maps to the pixel [0,1] of the 2D array.
    - If one_to_two[4] = [1,1], the fifth value of the 1D array maps to the pixel [1,1] of the 2D array.

    Parameters
     ----------
    grid_1d : ndarray
        The 1D array of values which are mapped to a 2D array.
    shape : (int, int)
        The shape of the 2D array which the pixels are defined on.
    one_to_two : ndarray
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

    grid_2d = np.zeros((shape[0], shape[1], 2))

    for index in range(len(one_to_two)):
        grid_2d[one_to_two[index, 0], one_to_two[index, 1], 0] = grid_1d[index, 0]
        grid_2d[one_to_two[index, 0], one_to_two[index, 1], 1] = grid_1d[index, 1]

    return grid_2d


@decorator_util.jit()
def map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d, shape):
    """For a 1D array that was flattened from a 2D array of shape (rows, columns), map its values back to the \
    original 2D array.

    The pixel coordinate origin is at the top left corner of the 2D array and goes right-wards and downwards, such
    that for an array of shape (3,3):

    - pixel 0 of the 1D array will correspond to index [0,0] of the 2D array.
    - pixel 1 of the 1D array will correspond to index [0,1] of the 2D array.
    - pixel 4 of the 1D array will correspond to index [1,0] of the 2D array.

    Parameters
     ----------
    array_1d : ndarray
        The 1D array of values which are mapped to a 2D array.
    shape : (int, int)
        The shape of the 2D array which the pixels are defined on.

    Returns
    --------
    ndarray
        A 2D array of values mapped from the 1D array with dimensions (shape).

    Examples
    --------
    one_to_two = np.array([[0,1], [1,0], [1,1], [1,2], [2,1]])

    array_1d = np.array([[2.0, 4.0, 5.0, 6.0, 8.0])

    array_2d = map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(array_1d=array_1d, shape=(3,3),
                                                                                  one_to_two=one_to_two)
    """

    array_2d = np.zeros(shape)

    index = 0
    for y in range(shape[0]):
        for x in range(shape[1]):
            array_2d[y, x] = array_1d[index]
            index += 1

    return array_2d


@decorator_util.jit()
def sparse_to_unmasked_sparse_from_mask_and_pixel_centres(total_sparse_pixels, mask,
                                                          unmasked_sparse_grid_pixel_centres):
    """Determine the mapping between every masked pixelization-grid pixel and pixelization-grid pixel. This is \
    performed by checking whether each pixelization-grid pixel is within the regular-masks, and mapping the indexes.

    Parameters
    -----------
    total_sparse_pixels : int
        The total number of pixels in the pixelization grid which fall within the regular-masks.
    mask : ccd.masks.Mask
        The regular-masks within which pixelization pixels must be inside
    unmasked_sparse_grid_pixel_centres : ndarray
        The centres of the unmasked pixelization grid pixels.
    """

    pix_to_full_pix = np.zeros(total_sparse_pixels)

    pixel_index = 0

    for full_pixel_index in range(unmasked_sparse_grid_pixel_centres.shape[0]):

        y = unmasked_sparse_grid_pixel_centres[full_pixel_index, 0]
        x = unmasked_sparse_grid_pixel_centres[full_pixel_index, 1]

        if not mask[y, x]:
            pix_to_full_pix[pixel_index] = full_pixel_index
            pixel_index += 1

    return pix_to_full_pix


@decorator_util.jit()
def unmasked_sparse_to_sparse_from_mask_and_pixel_centres(mask,
                                                          unmasked_sparse_grid_pixel_centres,
                                                          total_sparse_pixels):
    """Determine the mapping between every pixelization-grid pixel and masked pixelization-grid pixel. This is \
    performed by checking whether each pixelization-grid pixel is within the regular-masks, and mapping the indexes.

    Pixelization pixels are paired with the next masked pixel index. This may mean that a pixel is not paired with a \
    pixel near it, if the next pixel is on the next row of the grid. This is not a problem, as it is only \
    unmasked pixels that are referenced when computing image_to_pix, which is what this array is used for.

    Parameters
    -----------
    total_sparse_pixels : int
        The total number of pixels in the pixelization grid which fall within the regular-masks.
    mask : ccd.masks.Mask
        The regular-masks within which pixelization pixels must be inside
    unmasked_sparse_grid_pixel_centres : ndarray
        The centres of the unmasked pixelization grid pixels.
    """

    total_unmasked_sparse_pixels = unmasked_sparse_grid_pixel_centres.shape[0]

    unmasked_sparse_to_sparse = np.zeros(total_unmasked_sparse_pixels)
    pixel_index = 0

    for unmasked_sparse_pixel_index in range(total_unmasked_sparse_pixels):

        y = unmasked_sparse_grid_pixel_centres[unmasked_sparse_pixel_index, 0]
        x = unmasked_sparse_grid_pixel_centres[unmasked_sparse_pixel_index, 1]

        unmasked_sparse_to_sparse[unmasked_sparse_pixel_index] = pixel_index

        if not mask[y, x]:
            if pixel_index < total_sparse_pixels - 1:
                pixel_index += 1

    return unmasked_sparse_to_sparse


@decorator_util.jit()
def regular_to_sparse_from_sparse_mappings(regular_to_unmasked_sparse,
                                           unmasked_sparse_to_sparse):
    """Using the mapping between the regular-grid and unmasked pixelization grid, compute the mapping between each \
    regular pixel and the masked pixelization grid.

    Parameters
    -----------
    regular_to_unmasked_sparse : ndarray
        The index mapping between every regular-pixel and masked pixelization pixel.
    unmasked_sparse_to_sparse : ndarray
        The index mapping between every masked pixelization pixel and unmasked pixelization pixel.
    """
    total_regular_pixels = regular_to_unmasked_sparse.shape[0]

    regular_to_sparse = np.zeros(total_regular_pixels)

    for regular_index in range(total_regular_pixels):
        regular_to_sparse[regular_index] = unmasked_sparse_to_sparse[
            regular_to_unmasked_sparse[regular_index]]

    return regular_to_sparse


@decorator_util.jit()
def sparse_grid_from_unmasked_sparse_grid(unmasked_sparse_grid,
                                          sparse_to_unmasked_sparse):
    """Use the central arc-second coordinate of every unmasked pixelization grid's pixels and mapping between each \
    pixelization pixel and unmasked pixelization pixel to compute the central arc-second coordinate of every masked \
    pixelization grid pixel.

    Parameters
    -----------
    unmasked_sparse_grid : ndarray
        The (y,x) arc-second centre of every unmasked pixelization grid pixel.
    sparse_to_unmasked_sparse : ndarray
        The index mapping between every pixelization pixel and masked pixelization pixel.
    """
    total_pix_pixels = sparse_to_unmasked_sparse.shape[0]

    pix_grid = np.zeros((total_pix_pixels, 2))

    for pixel_index in range(total_pix_pixels):
        pix_grid[pixel_index, 0] = unmasked_sparse_grid[
            sparse_to_unmasked_sparse[pixel_index], 0]
        pix_grid[pixel_index, 1] = unmasked_sparse_grid[
            sparse_to_unmasked_sparse[pixel_index], 1]

    return pix_grid


@decorator_util.jit()
def regular_to_sparse_from_cluster_grid(cluster_labels, cluster_to_regular_all,
                                        cluster_to_regular_sizes,
                                        total_regular_pixels):
    regular_to_sparse = np.zeros(total_regular_pixels)

    for cluster_index in range(cluster_to_regular_all.shape[0]):
        for cluster_count in range(cluster_to_regular_sizes[cluster_index]):
            regular_index = cluster_to_regular_all[cluster_index, cluster_count]
            regular_to_sparse[regular_index] = cluster_labels[cluster_index]

    return regular_to_sparse

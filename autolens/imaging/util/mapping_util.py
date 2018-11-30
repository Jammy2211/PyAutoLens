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

@numba.jit(nopython=True, cache=True)
def sparse_to_unmasked_sparse_from_mask_and_pixel_centres(total_sparse_pixels, mask, unmasked_sparse_grid_pixel_centres):
    """Determine the mapping between every masked pixelization-grid pixel and pixelization-grid pixel. This is
    performed by checking whether each pixelization-grid pixel is within the image-mask, and mapping the indexes.

    Parameters
    -----------
    total_sparse_pixels : int
        The total number of pixels in the pixelization grid which fall within the image-mask.
    mask : imaging.mask.Mask
        The image-mask within which pixelization pixels must be inside
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

@numba.jit(nopython=True, cache=True)
def unmasked_sparse_to_sparse_from_mask_and_pixel_centres(mask, unmasked_sparse_grid_pixel_centres):
    """Determine the mapping between every pixelization-grid pixel and masked pixelization-grid pixel. This is
    performed by checking whether each pixelization-grid pixel is within the image-mask, and mapping the indexes.

    Pixelization pixels are paired with the next masked pixel index. This may mean that a pixel is not paired with a
    pixel near it, if the next pixel is on the next row of the grid. This is not a problem, as it is only
    unmasked pixels that are referened when computing image_to_pix, which is what this array is used for.

    Parameters
    -----------
    total_pix_pixels : int
        The total number of pixels in the pixelization grid which fall within the image-mask.
    mask : imaging.mask.Mask
        The image-mask within which pixelization pixels must be inside
    unmasked_sparse_grid_pixel_centres : ndarray
        The centres of the unmasked pixelization grid pixels.
    """

    total_all_pixels = unmasked_sparse_grid_pixel_centres.shape[0]

    full_pix_to_pix = np.zeros(total_all_pixels)
    pixel_index = 0

    for full_pixel_index in range(total_all_pixels):

        y = unmasked_sparse_grid_pixel_centres[full_pixel_index, 0]
        x = unmasked_sparse_grid_pixel_centres[full_pixel_index, 1]

        full_pix_to_pix[full_pixel_index] = pixel_index

        if not mask[y, x]:

            pixel_index += 1

    return full_pix_to_pix

@numba.jit(nopython=True, cache=True)
def image_to_sparse_from_sparse_mappings(image_to_unmasked_sparse, unmasked_sparse_to_sparse):
    """Using the mapping between the image-grid and unmasked pixelization grid, compute the mapping between each image
    pixel and the masked pixelization grid.

    Parameters
    -----------
    image_to_unmasked_sparse : ndarray
        The index mapping between every image-pixel and masked pixelization pixel.
    unmasked_sparse_to_sparse : ndarray
        The index mapping between every masked pixelization pixel and unmasked pixelization pixel.
    """
    total_image_pixels = image_to_unmasked_sparse.shape[0]

    image_to_pix = np.zeros(total_image_pixels)

    for image_index in range(total_image_pixels):

        image_to_pix[image_index] = unmasked_sparse_to_sparse[image_to_unmasked_sparse[image_index]]

    return image_to_pix

@numba.jit(nopython=True, cache=True)
def sparse_grid_from_unmasked_sparse_grid(unmasked_sparse_grid, sparse_to_unmasked_sparse):
    """Use the central arc-second coordinate of every unmasked pixelization grid's pixels and mapping between each
    pixelization pixel and unmasked pixelization pixel to compute the central arc-second coordinate of every masked
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

        pix_grid[pixel_index, :] = unmasked_sparse_grid[sparse_to_unmasked_sparse[pixel_index], :]

    return pix_grid
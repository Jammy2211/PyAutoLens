import numba
import numpy as np

@numba.jit(nopython=True, cache=True)
def total_masked_pixels(mask, full_pix_grid_pixel_centres):
    """Given the full (i.e. without removing pixels which are outside the image-mask) pixelization grid's pixel centers
    and the image-mask, compute the total number of pixels which are within the image-mask and thus used by the
    pixelization grid.

    Parameters
    -----------
    mask : imaging.mask.Mask
        The image-mask within which pixelization pixels must be inside
    full_pix_grid_pixel_centres : ndarray
        The centres of the unmasked pixelization grid pixels.
    """

    total_masked_pixels = 0

    for all_pixel_index in range(full_pix_grid_pixel_centres.shape[0]):

        y = full_pix_grid_pixel_centres[all_pixel_index, 0]
        x = full_pix_grid_pixel_centres[all_pixel_index, 1]

        if not mask[y,x]:
            total_masked_pixels += 1

    return total_masked_pixels

@numba.jit(nopython=True, cache=True)
def pix_to_full_pix(total_masked_pixels, mask, full_pix_grid_pixel_centres):
    """Determine the mapping between every masked pixelization-grid pixel and pixelization-grid pixel. This is
    performed by checking whether each pixelization-grid pixel is within the image-mask, and mapping the indexes.

    Parameters
    -----------
    total_masked_pixels : int
        The total number of pixels in the pixelization grid which fall within the image-mask.
    mask : imaging.mask.Mask
        The image-mask within which pixelization pixels must be inside
    full_pix_grid_pixel_centres : ndarray
        The centres of the unmasked pixelization grid pixels.
    """

    pix_to_full_pix = np.zeros(total_masked_pixels)

    pixel_index = 0

    for full_pixel_index in range(full_pix_grid_pixel_centres.shape[0]):

        y = full_pix_grid_pixel_centres[full_pixel_index, 0]
        x = full_pix_grid_pixel_centres[full_pixel_index, 1]

        if not mask[y, x]:

            pix_to_full_pix[pixel_index] = full_pixel_index
            pixel_index += 1

    return pix_to_full_pix

@numba.jit(nopython=True, cache=True)
def full_pix_to_pix(mask, full_pix_grid_pixel_centres):
    """Determine the mapping between every pixelization-grid pixel and masked pixelization-grid pixel. This is
    performed by checking whether each pixelization-grid pixel is within the image-mask, and mapping the indexes.

    Pixelization pixels are paired with the next masked pixel index. This may mean that a pixel is not paired with a
    pixel near it, if the next pixel is on the next row of the grid. This is not a problem, as it is only
    unmasked pixels that are referened when computing image_to_pix, which is what this array is used for.

    Parameters
    -----------
    total_masked_pixels : int
        The total number of pixels in the pixelization grid which fall within the image-mask.
    mask : imaging.mask.Mask
        The image-mask within which pixelization pixels must be inside
    full_pix_grid_pixel_centres : ndarray
        The centres of the unmasked pixelization grid pixels.
    """

    total_all_pixels = full_pix_grid_pixel_centres.shape[0]

    full_pix_to_pix = np.zeros(total_all_pixels)
    pixel_index = 0

    for full_pixel_index in range(total_all_pixels):

        y = full_pix_grid_pixel_centres[full_pixel_index, 0]
        x = full_pix_grid_pixel_centres[full_pixel_index, 1]

        full_pix_to_pix[full_pixel_index] = pixel_index

        if not mask[y, x]:

            pixel_index += 1

    return full_pix_to_pix

@numba.jit(nopython=True, cache=True)
def pix_grid_from_(total_masked_pixels, pixelization_grid, pix_to_unmasked_pix):

    pix_grid = np.zeros((total_masked_pixels, 2))

    masked_pixel_index = 0
    for pixel_index in pix_to_unmasked_pix:
        pix_grid[masked_pixel_index, :] = pixelization_grid[pixel_index, :]
        masked_pixel_index += 1

    return pix_grid

# def image_to_pix(mask, ):
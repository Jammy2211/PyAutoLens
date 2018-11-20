import numba
import numpy as np

# @numba.jit(nopython=True, cache=True)
def total_pix_pixels_from_mask(mask, full_pix_grid_pixel_centres):
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

    total_pix_pixels = 0

    for full_pixel_index in range(full_pix_grid_pixel_centres.shape[0]):

        y = full_pix_grid_pixel_centres[full_pixel_index, 0]
        x = full_pix_grid_pixel_centres[full_pixel_index, 1]

        if not mask[y,x]:
            total_pix_pixels += 1

    return total_pix_pixels

# @numba.jit(nopython=True, cache=True)
def pix_to_full_pix_from_mask_and_pixel_centres(total_pix_pixels, mask, full_pix_grid_pixel_centres):
    """Determine the mapping between every masked pixelization-grid pixel and pixelization-grid pixel. This is
    performed by checking whether each pixelization-grid pixel is within the image-mask, and mapping the indexes.

    Parameters
    -----------
    total_pix_pixels : int
        The total number of pixels in the pixelization grid which fall within the image-mask.
    mask : imaging.mask.Mask
        The image-mask within which pixelization pixels must be inside
    full_pix_grid_pixel_centres : ndarray
        The centres of the unmasked pixelization grid pixels.
    """

    pix_to_full_pix = np.zeros(total_pix_pixels)

    pixel_index = 0

    for full_pixel_index in range(full_pix_grid_pixel_centres.shape[0]):

        y = full_pix_grid_pixel_centres[full_pixel_index, 0]
        x = full_pix_grid_pixel_centres[full_pixel_index, 1]

        if not mask[y, x]:

            pix_to_full_pix[pixel_index] = full_pixel_index
            pixel_index += 1

    return pix_to_full_pix

# @numba.jit(nopython=True, cache=True)
def full_pix_to_pix_from_mask_and_pixel_centres(mask, full_pix_grid_pixel_centres):
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

# @numba.jit(nopython=True, cache=True)
def image_to_pix_from_pix_mappings(image_to_full_pix, full_pix_to_pix):
    """Using the mapping between the image-grid and unmasked pixelization grid, compute the mapping between each image
    pixel and the masked pixelization grid.

    Parameters
    -----------
    image_to_full_pix : ndarray
        The index mapping between every image-pixel and masked pixelization pixel.
    full_pix_to_pix : ndarray
        The index mapping between every masked pixelization pixel and unmasked pixelization pixel.
    """
    total_image_pixels = image_to_full_pix.shape[0]

    image_to_pix = np.zeros(total_image_pixels)

    for image_index in range(total_image_pixels):

        image_to_pix[image_index] = full_pix_to_pix[image_to_full_pix[image_index]]

    return image_to_pix

# @numba.jit(nopython=True, cache=True)
def pix_grid_from_full_pix_grid(full_pix_grid, pix_to_full_pix):
    """Use the central arc-second coordinate of every unmasked pixelization grid's pixels and mapping between each
    pixelization pixel and unmasked pixelization pixel to compute the central arc-second coordinate of every masked
    pixelization grid pixel.

    Parameters
    -----------
    full_pix_grid : ndarray
        The (y,x) arc-second centre of every unmasked pixelization grid pixel.
    pix_to_full_pix : ndarray
        The index mapping between every pixelization pixel and masked pixelization pixel.
    """
    total_pix_pixels = pix_to_full_pix.shape[0]

    pix_grid = np.zeros((total_pix_pixels, 2))

    for pixel_index in range(total_pix_pixels):

        pix_grid[pixel_index, :] = full_pix_grid[pix_to_full_pix[pixel_index], :]

    return pix_grid
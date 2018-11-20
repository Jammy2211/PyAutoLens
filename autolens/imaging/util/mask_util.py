import os
import numba
import numpy as np
from astropy.io import fits

from autolens import exc
from autolens.imaging.util import mapping_util

@numba.jit(nopython=True, cache=True)
def mask_centres_from_shape_pixel_scale_and_centre(shape, pixel_scale, centre):

    y_cen = (float(shape[0] - 1) / 2) - (centre[0] / pixel_scale)
    x_cen = (float(shape[1] - 1) / 2) + (centre[1] / pixel_scale)

    return y_cen, x_cen

@numba.jit(nopython=True, cache=True)
def total_image_pixels_from_mask(mask):
    """Compute the total number of unmasked image pixels in a masks."""

    total_image_pixels = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                total_image_pixels += 1

    return total_image_pixels

@numba.jit(nopython=True, cache=True)
def total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size):
    """Compute the total number of sub-pixels in unmasked image pixels in a masks."""
    return total_image_pixels_from_mask(mask) * sub_grid_size ** 2

@numba.jit(nopython=True, cache=True)
def mask_circular_from_shape_pixel_scale_and_radius(shape, pixel_scale, radius_arcsec, centre=(0.0, 0.0)):
    """Compute a circular masks from an input masks radius and image shape."""

    mask = np.full(shape, True)

    y_cen, x_cen = mask_centres_from_shape_pixel_scale_and_centre(shape=mask.shape, pixel_scale=pixel_scale, centre=centre)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_arcsec = (y - y_cen) * pixel_scale
            x_arcsec = (x - x_cen) * pixel_scale

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if r_arcsec <= radius_arcsec:
                mask[y, x] = False

    return mask

@numba.jit(nopython=True, cache=True)
def mask_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec,
                                                  centre=(0.0, 0.0)):
    """Compute an annular masks from an input inner and outer masks radius and image shape."""

    mask = np.full(shape, True)

    y_cen, x_cen = mask_centres_from_shape_pixel_scale_and_centre(shape=mask.shape, pixel_scale=pixel_scale, centre=centre)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_arcsec = (y - y_cen) * pixel_scale
            x_arcsec = (x - x_cen) * pixel_scale

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if outer_radius_arcsec >= r_arcsec >= inner_radius_arcsec:
                mask[y, x] = False

    return mask

@numba.jit(nopython=True, cache=True)
def mask_anti_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec,
                                                       outer_radius_2_arcsec, centre=(0.0, 0.0)):
    """Compute an annular masks from an input inner and outer masks radius and image shape."""

    mask = np.full(shape, True)

    y_cen, x_cen = mask_centres_from_shape_pixel_scale_and_centre(shape=mask.shape, pixel_scale=pixel_scale,
                                                                  centre=centre)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_arcsec = (y - y_cen) * pixel_scale
            x_arcsec = (x - x_cen) * pixel_scale

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if  inner_radius_arcsec >= r_arcsec or outer_radius_2_arcsec >= r_arcsec >= outer_radius_arcsec:
                mask[y, x] = False

    return mask

@numba.jit(nopython=True, cache=True)
def mask_blurring_from_mask_and_psf_shape(mask, psf_shape):
    """Compute a blurring masks from an input masks and psf shape.

    The blurring masks corresponds to all pixels which are outside of the masks but will have a fraction of their \
    light blur into the masked region due to PSF convolution."""

    blurring_mask = np.full(mask.shape, True)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                for y1 in range((-psf_shape[0] + 1) // 2, (psf_shape[0] + 1) // 2):
                    for x1 in range((-psf_shape[1] + 1) // 2, (psf_shape[1] + 1) // 2):
                        if 0 <= x + x1 <= mask.shape[1] - 1 and 0 <= y + y1 <= mask.shape[0] - 1:
                            if mask[y + y1, x + x1]:
                                blurring_mask[y + y1, x + x1] = False
                        else:
                            raise exc.MaskException(
                                "setup_blurring_mask extends beyond the sub_grid_size of the masks - pad the "
                                "masked_image before masking")

    return blurring_mask

@numba.jit(nopython=True, cache=True)
def masked_grid_1d_index_to_2d_pixel_index_from_mask(mask):
    """Compute a 1D array that maps every unmasked pixel to its corresponding 2d pixel using its (y,x) pixel indexes.

    For howtolens if pixel [2,5] corresponds to the second pixel on the 1D array, grid_to_pixel[1] = [2,5]"""

    total_image_pixels = total_image_pixels_from_mask(mask)
    grid_to_pixel = np.zeros(shape=(total_image_pixels, 2))
    pixel_count = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                grid_to_pixel[pixel_count, :] = y, x
                pixel_count += 1

    return grid_to_pixel

@numba.jit(nopython=True, cache=True)
def total_edge_pixels_from_mask(mask):
    """Compute the total number of border-pixels in a masks."""

    border_pixel_total = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                if mask[y + 1, x] or mask[y - 1, x] or mask[y, x + 1] or mask[y, x - 1] or \
                        mask[y + 1, x + 1] or mask[y + 1, x - 1] or mask[y - 1, x + 1] or mask[y - 1, x - 1]:
                    border_pixel_total += 1

    return border_pixel_total

@numba.jit(nopython=True, cache=True)
def edge_pixels_from_mask(mask):
    """Compute a 1D array listing all edge pixel indexes in the masks. An edge pixel is a pixel which is not fully \
    surrounding by False masks values i.e. it is on an edge."""

    edge_pixel_total = total_edge_pixels_from_mask(mask)

    edge_pixels = np.zeros(edge_pixel_total)
    edge_index = 0
    image_index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                if mask[y + 1, x] or mask[y - 1, x] or mask[y, x + 1] or mask[y, x - 1] or \
                        mask[y + 1, x + 1] or mask[y + 1, x - 1] or mask[y - 1, x + 1] or mask[y - 1, x - 1]:
                    edge_pixels[edge_index] = image_index
                    edge_index += 1

                image_index += 1

    return edge_pixels

@numba.jit(nopython=True, cache=True)
def check_if_border_pixel(mask, edge_pixel_1d, masked_grid_index_to_pixel):

    edge_pixel_index = int(edge_pixel_1d)

    y = int(masked_grid_index_to_pixel[edge_pixel_index, 0])
    x = int(masked_grid_index_to_pixel[edge_pixel_index, 1])

    if np.sum(mask[0:y, x]) == y or \
            np.sum(mask[y, x:mask.shape[1]]) == mask.shape[1] - x - 1 or \
            np.sum(mask[y:mask.shape[0], x]) == mask.shape[0] - y - 1 or \
            np.sum(mask[y, 0:x]) == x:
        return True
    else:
        return False

@numba.jit(nopython=True, cache=True)
def total_border_pixels_from_mask_and_edge_pixels(mask, edge_pixels, masked_grid_index_to_pixel):
    """Compute the total number of border-pixels in a masks."""

    border_pixel_total = 0

    for i in range(edge_pixels.shape[0]):

        if check_if_border_pixel(mask, edge_pixels[i], masked_grid_index_to_pixel):
            border_pixel_total += 1

    return border_pixel_total

@numba.jit(nopython=True, cache=True)
def border_pixels_from_mask(mask):
    """Compute a 1D array listing all border pixel indexes in the masks. A border pixel is a pixel which:

     1) is not fully surrounding by False masks values.
     2) Can reach the edge of the array without hitting a masked pixel in one of four directions (upwards, downwards,
     left, right).

     The border pixels are thus pixels which are on the exterior edge of the mask. For example, the inner ring of edge \
     pixels in an annular mask are edge pixels but not border pixels."""

    edge_pixels = edge_pixels_from_mask(mask)
    masked_grid_index_to_pixel = masked_grid_1d_index_to_2d_pixel_index_from_mask(mask)

    border_pixel_total = total_border_pixels_from_mask_and_edge_pixels(mask, edge_pixels, masked_grid_index_to_pixel)

    border_pixels = np.zeros(border_pixel_total)

    border_pixel_index = 0

    for edge_pixel_index in range(edge_pixels.shape[0]):

        if check_if_border_pixel(mask, edge_pixels[edge_pixel_index], masked_grid_index_to_pixel):
            border_pixels[border_pixel_index] = edge_pixels[edge_pixel_index]
            border_pixel_index += 1

    return border_pixels
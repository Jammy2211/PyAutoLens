import os
import numba
import numpy as np
from astropy.io import fits

from autolens import exc

@numba.jit(nopython=True, cache=True)
def total_image_pixels_from_mask(mask):
    """Compute the total number of unmasked datas_ pixels in a masks."""

    total_image_pixels = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                total_image_pixels += 1

    return total_image_pixels

@numba.jit(nopython=True, cache=True)
def total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size):
    """Compute the total number of sub-pixels in unmasked datas_ pixels in a masks."""
    return total_image_pixels_from_mask(mask) * sub_grid_size ** 2

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
    """Compute a 1D array listing all border pixel indexes in the masks. A border pixel is a pixel which is not fully \
    surrounding by False masks values i.e. it is on an edge."""

    border_pixel_total = total_edge_pixels_from_mask(mask)

    border_pixels = np.zeros(border_pixel_total)
    border_index = 0
    image_index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                if mask[y + 1, x] or mask[y - 1, x] or mask[y, x + 1] or mask[y, x - 1] or \
                        mask[y + 1, x + 1] or mask[y + 1, x - 1] or mask[y - 1, x + 1] or mask[y - 1, x - 1]:
                    border_pixels[border_index] = image_index
                    border_index += 1

                image_index += 1

    return border_pixels

@numba.jit(nopython=True, cache=True)
def edge_sub_pixels_from_mask_pixel_scales_and_sub_grid_size(mask, pixel_scales, sub_grid_size):
    """Compute a 1D array listing all sub-pixel border pixel indexes in the masks. A border sub-pixel is a sub-pixel \
    whose datas_ pixel is not fully surrounded by False masks values and it is closest to the edge."""
    border_pixel_total = total_edge_pixels_from_mask(mask)
    border_sub_pixels = np.zeros(border_pixel_total)

    image_index = 0

    y_cen = float(mask.shape[0] - 1) / 2
    x_cen = float(mask.shape[1] - 1) / 2

    y_sub_half = pixel_scales[0] / 2
    y_sub_step = pixel_scales[0] / (sub_grid_size + 1)

    x_sub_half = pixel_scales[1] / 2
    x_sub_step = pixel_scales[1] / (sub_grid_size + 1)

    border_index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                if mask[y + 1, x] or mask[y - 1, x] or mask[y, x + 1] or mask[y, x - 1] or \
                        mask[y + 1, x + 1] or mask[y + 1, x - 1] or mask[y - 1, x + 1] or mask[y - 1, x - 1]:

                    y_arcsec = (y - y_cen) * pixel_scales[0]
                    x_arcsec = (x - x_cen) * pixel_scales[1]

                    sub_grid = np.zeros((sub_grid_size ** 2, 2))
                    sub_index = 0

                    for y1 in range(sub_grid_size):
                        for x1 in range(sub_grid_size):
                            sub_grid[sub_index, 0] = y_arcsec - y_sub_half + (y1 + 1) * y_sub_step
                            sub_grid[sub_index, 1] = x_arcsec - x_sub_half + (x1 + 1) * x_sub_step
                            sub_index += 1

                    sub_grid_radii = np.add(np.square(sub_grid[:, 0]), np.square(sub_grid[:, 1]))
                    border_sub_index = image_index * (sub_grid_size ** 2) + np.argmax(sub_grid_radii)
                    border_sub_pixels[border_index] = border_sub_index
                    border_index += 1

                image_index += 1

    return border_sub_pixels

@numba.jit(nopython=True, cache=True)
def mask_circular_from_shape_pixel_scale_and_radius(shape, pixel_scale, radius_arcsec, centre=(0.0, 0.0)):
    """Compute a circular masks from an input masks radius and datas_ shape."""

    mask = np.full(shape, True)

    y_cen = (float(mask.shape[0] - 1) / 2) - (centre[0] / pixel_scale)
    x_cen = (float(mask.shape[1] - 1) / 2) + (centre[1] / pixel_scale)

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
    """Compute an annular masks from an input inner and outer masks radius and datas_ shape."""

    mask = np.full(shape, True)

    y_cen = (float(mask.shape[0] - 1) / 2) - (centre[0] / pixel_scale)
    x_cen = (float(mask.shape[1] - 1) / 2) + (centre[1] / pixel_scale)

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
    """Compute an annular masks from an input inner and outer masks radius and datas_ shape."""

    mask = np.full(shape, True)

    y_cen = (float(mask.shape[1] - 1) / 2) - (centre[1] / pixel_scale)
    x_cen = (float(mask.shape[0] - 1) / 2) + (centre[0] / pixel_scale)

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
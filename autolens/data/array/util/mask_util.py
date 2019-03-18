from autolens import decorator_util
from autolens.data.array.util import array_util, grid_util
import numpy as np
from skimage.transform import rescale

from autolens import exc


@decorator_util.jit()
def mask_centres_from_shape_pixel_scale_and_centre(shape, pixel_scale, centre):

    y_centre_arcsec = (float(shape[0] - 1) / 2) - (centre[0] / pixel_scale)
    x_centre_arcsec = (float(shape[1] - 1) / 2) + (centre[1] / pixel_scale)

    return (y_centre_arcsec, x_centre_arcsec)

@decorator_util.jit()
def total_regular_pixels_from_mask(mask):
    """Compute the total number of unmasked regular pixels in a masks."""

    total_regular_pixels = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                total_regular_pixels += 1

    return total_regular_pixels

@decorator_util.jit()
def total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size):
    """Compute the total number of sub-pixels in unmasked regular pixels in a masks."""
    return total_regular_pixels_from_mask(mask) * sub_grid_size ** 2

@decorator_util.jit()
def total_sparse_pixels_from_mask(mask, unmasked_sparse_grid_pixel_centres):
    """Given the full (i.e. without removing pixels which are outside the regular-masks) pixelization grid's pixel centers
    and the regular-masks, compute the total number of pixels which are within the regular-masks and thus used by the
    pixelization grid.

    Parameters
    -----------
    mask : ccd.masks.Mask
        The regular-masks within which pixelization pixels must be inside
    unmasked_sparse_grid_pixel_centres : ndarray
        The centres of the unmasked pixelization grid pixels.
    """

    total_sparse_pixels = 0

    for unmasked_sparse_pixel_index in range(unmasked_sparse_grid_pixel_centres.shape[0]):

        y = unmasked_sparse_grid_pixel_centres[unmasked_sparse_pixel_index, 0]
        x = unmasked_sparse_grid_pixel_centres[unmasked_sparse_pixel_index, 1]

        if not mask[y,x]:
            total_sparse_pixels += 1

    return total_sparse_pixels

@decorator_util.jit()
def mask_circular_from_shape_pixel_scale_and_radius(shape, pixel_scale, radius_arcsec, centre=(0.0, 0.0)):
    """Compute a circular masks from an input masks radius and regular shape."""

    mask = np.full(shape, True)

    centres_arcsec = mask_centres_from_shape_pixel_scale_and_centre(shape=mask.shape, pixel_scale=pixel_scale, centre=centre)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_arcsec = (y - centres_arcsec[0]) * pixel_scale
            x_arcsec = (x - centres_arcsec[1]) * pixel_scale

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if r_arcsec <= radius_arcsec:
                mask[y, x] = False

    return mask

@decorator_util.jit()
def mask_circular_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec,
                                                           centre=(0.0, 0.0)):
    """Compute an annular masks from an input inner and outer masks radius and regular shape."""

    mask = np.full(shape, True)

    centres_arcsec = mask_centres_from_shape_pixel_scale_and_centre(shape=mask.shape, pixel_scale=pixel_scale, centre=centre)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_arcsec = (y - centres_arcsec[0]) * pixel_scale
            x_arcsec = (x - centres_arcsec[1]) * pixel_scale

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if outer_radius_arcsec >= r_arcsec >= inner_radius_arcsec:
                mask[y, x] = False

    return mask

@decorator_util.jit()
def mask_circular_anti_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec,
                                                                outer_radius_2_arcsec, centre=(0.0, 0.0)):
    """Compute an annular masks from an input inner and outer masks radius and regular shape."""

    mask = np.full(shape, True)

    centres_arcsec = mask_centres_from_shape_pixel_scale_and_centre(shape=mask.shape, pixel_scale=pixel_scale,
                                                                  centre=centre)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_arcsec = (y - centres_arcsec[0]) * pixel_scale
            x_arcsec = (x - centres_arcsec[1]) * pixel_scale

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if  inner_radius_arcsec >= r_arcsec or outer_radius_2_arcsec >= r_arcsec >= outer_radius_arcsec:
                mask[y, x] = False

    return mask

@decorator_util.jit()
def elliptical_radius_from_y_x_phi_and_axis_ratio(y_arcsec, x_arcsec, phi, axis_ratio):
    r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

    theta_rotated = np.arctan2(y_arcsec, x_arcsec) + np.radians(phi)

    y_arcsec_elliptical = r_arcsec * np.sin(theta_rotated)
    x_arcsec_elliptical = r_arcsec * np.cos(theta_rotated)

    return np.sqrt(x_arcsec_elliptical ** 2.0 + (y_arcsec_elliptical / axis_ratio) ** 2.0)

@decorator_util.jit()
def mask_elliptical_from_shape_pixel_scale_and_radius(shape, pixel_scale, major_axis_radius_arcsec, axis_ratio, phi,
                                                      centre=(0.0, 0.0)):
    """Compute a circular masks from an input masks radius and regular shape."""

    mask = np.full(shape, True)

    centres_arcsec = mask_centres_from_shape_pixel_scale_and_centre(shape=mask.shape, pixel_scale=pixel_scale,
                                                                  centre=centre)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_arcsec = (y - centres_arcsec[0]) * pixel_scale
            x_arcsec = (x - centres_arcsec[1]) * pixel_scale

            r_arcsec_elliptical = elliptical_radius_from_y_x_phi_and_axis_ratio(y_arcsec, x_arcsec, phi, axis_ratio)

            if r_arcsec_elliptical <= major_axis_radius_arcsec:
                mask[y, x] = False

    return mask

@decorator_util.jit()
def mask_elliptical_annular_from_shape_pixel_scale_and_radius(shape, pixel_scale,
                                                              inner_major_axis_radius_arcsec, inner_axis_ratio, inner_phi,
                                                              outer_major_axis_radius_arcsec, outer_axis_ratio, outer_phi,
                                                      centre=(0.0, 0.0)):
    """Compute a circular masks from an input masks radius and regular shape."""

    mask = np.full(shape, True)

    centres_arcsec = mask_centres_from_shape_pixel_scale_and_centre(shape=mask.shape, pixel_scale=pixel_scale,
                                                                  centre=centre)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_arcsec = (y - centres_arcsec[0]) * pixel_scale
            x_arcsec = (x - centres_arcsec[1]) * pixel_scale

            inner_r_arcsec_elliptical = elliptical_radius_from_y_x_phi_and_axis_ratio(y_arcsec, x_arcsec,
                                                                                      inner_phi, inner_axis_ratio)

            outer_r_arcsec_elliptical = elliptical_radius_from_y_x_phi_and_axis_ratio(y_arcsec, x_arcsec,
                                                                                      outer_phi, outer_axis_ratio)

            if inner_r_arcsec_elliptical >= inner_major_axis_radius_arcsec and \
                outer_r_arcsec_elliptical <= outer_major_axis_radius_arcsec:
                mask[y, x] = False

    return mask

@decorator_util.jit()
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
                                "datas array before masking")

    return blurring_mask

@decorator_util.jit()
def masked_grid_1d_index_to_2d_pixel_index_from_mask(mask):
    """Compute a 1D array that maps every unmasked pixel to its corresponding 2d pixel using its (y,x) pixel indexes.

    For howtolens if pixel [2,5] corresponds to the second pixel on the 1D array, grid_to_pixel[1] = [2,5]"""

    total_regular_pixels = total_regular_pixels_from_mask(mask)
    grid_to_pixel = np.zeros(shape=(total_regular_pixels, 2))
    pixel_count = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                grid_to_pixel[pixel_count, :] = y, x
                pixel_count += 1

    return grid_to_pixel

@decorator_util.jit()
def masked_sub_grid_1d_index_to_2d_sub_pixel_index_from_mask(mask, sub_grid_size):
    """Compute a 1D array that maps every unmasked pixel to its corresponding 2d pixel using its (y,x) pixel indexes.

    For howtolens if pixel [2,5] corresponds to the second pixel on the 1D array, grid_to_pixel[1] = [2,5]"""

    total_sub_pixels = total_sub_pixels_from_mask_and_sub_grid_size(mask=mask, sub_grid_size=sub_grid_size)
    sub_grid_to_sub_pixel = np.zeros(shape=(total_sub_pixels, 2))
    sub_pixel_count = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                for y1 in range(sub_grid_size):
                    for x1 in range(sub_grid_size):
                        sub_grid_to_sub_pixel[sub_pixel_count, :] = (y*sub_grid_size)+y1, (x*sub_grid_size)+x1
                        sub_pixel_count += 1

    return sub_grid_to_sub_pixel

@decorator_util.jit()
def total_edge_pixels_from_mask(mask):
    """Compute the total number of borders-pixels in a masks."""

    border_pixel_total = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                if mask[y + 1, x] or mask[y - 1, x] or mask[y, x + 1] or mask[y, x - 1] or \
                        mask[y + 1, x + 1] or mask[y + 1, x - 1] or mask[y - 1, x + 1] or mask[y - 1, x - 1]:
                    border_pixel_total += 1

    return border_pixel_total

@decorator_util.jit()
def edge_pixels_from_mask(mask):
    """Compute a 1D array listing all edge pixel indexes in the masks. An edge pixel is a pixel which is not fully \
    surrounding by False masks values i.e. it is on an edge."""

    edge_pixel_total = total_edge_pixels_from_mask(mask)

    edge_pixels = np.zeros(edge_pixel_total)
    edge_index = 0
    regular_index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                if mask[y + 1, x] or mask[y - 1, x] or mask[y, x + 1] or mask[y, x - 1] or \
                        mask[y + 1, x + 1] or mask[y + 1, x - 1] or mask[y - 1, x + 1] or mask[y - 1, x - 1]:
                    edge_pixels[edge_index] = regular_index
                    edge_index += 1

                regular_index += 1

    return edge_pixels

@decorator_util.jit()
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

@decorator_util.jit()
def total_border_pixels_from_mask_and_edge_pixels(mask, edge_pixels, masked_grid_index_to_pixel):
    """Compute the total number of borders-pixels in a masks."""

    border_pixel_total = 0

    for i in range(edge_pixels.shape[0]):

        if check_if_border_pixel(mask, edge_pixels[i], masked_grid_index_to_pixel):
            border_pixel_total += 1

    return border_pixel_total

@decorator_util.jit()
def border_pixels_from_mask(mask):
    """Compute a 1D array listing all borders pixel indexes in the masks. A borders pixel is a pixel which:

     1) is not fully surrounding by False masks values.
     2) Can reach the edge of the array without hitting a masked pixel in one of four directions (upwards, downwards,
     left, right).

     The borders pixels are thus pixels which are on the exterior edge of the masks. For example, the inner ring of edge \
     pixels in an annular masks are edge pixels but not borders pixels."""

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

@decorator_util.jit()
def edge_buffed_mask_from_mask(mask):

    edge_buffed_mask = mask.copy()

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                edge_buffed_mask[y + 1, x] = False
                edge_buffed_mask[y - 1, x] = False
                edge_buffed_mask[y, x + 1] = False
                edge_buffed_mask[y, x - 1] = False
                edge_buffed_mask[y + 1, x + 1] = False
                edge_buffed_mask[y + 1, x - 1] = False
                edge_buffed_mask[y - 1, x + 1] = False
                edge_buffed_mask[y - 1, x - 1] = False

    return edge_buffed_mask

@decorator_util.jit()
def bin_up_mask_2d(mask_2d, bin_up_factor):
    """Bin up an array to coarser resolution, by binning up groups of pixels and using their sum value to determine \
     the value of the new pixel.

    If an array of shape (8,8) is input and the bin up size is 2, this would return a new array of size (4,4) where \
    every pixel was the sum of each collection of 2x2 pixels on the (8,8) array.

    If binning up the array leads to an edge being cut (e.g. a (9,9) array binned up by 2), an array is first \
    extracted around the centre of that array.


    Parameters
    ----------
    mask_2d : ndarray
        The 2D array that is resized.
    new_shape : (int, int)
        The (y,x) new pixel dimension of the trimmed array.
    origin : (int, int)
        The oigin of the resized array, e.g. the central pixel around which the array is extracted.

    Returns
    -------
    ndarray
        The resized 2D array from the input 2D array.

    Examples
    --------
    array_2d = np.ones((5,5))
    resize_array = resize_array_2d(array_2d=array_2d, new_shape=(2,2), origin=(2, 2))
    """

    padded_array_2d = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(
        array_2d=mask_2d, bin_up_factor=bin_up_factor, pad_value=True)

    binned_array_2d = np.zeros(shape=(padded_array_2d.shape[0] // bin_up_factor,
                                      padded_array_2d.shape[1] // bin_up_factor))

    for y in range(binned_array_2d.shape[0]):
        for x in range(binned_array_2d.shape[1]):
            value = True
            for y1 in range(bin_up_factor):
                for x1 in range(bin_up_factor):
                    padded_y = y*bin_up_factor + y1
                    padded_x = x*bin_up_factor + x1
                    if padded_array_2d[padded_y, padded_x] == False:
                        value = False

            binned_array_2d[y,x] = value

    return binned_array_2d

def rescaled_mask_from_mask_and_rescale_factor(mask, rescale_factor):

    rescaled_mask = rescale(image=mask, scale=rescale_factor, mode='edge', anti_aliasing=False, multichannel=False)
    rescaled_mask[0, :] = True
    rescaled_mask[rescaled_mask.shape[0]-1, :] = True
    rescaled_mask[:, 0] = True
    rescaled_mask[:, rescaled_mask.shape[1]-1] = True
    return rescaled_mask == 1

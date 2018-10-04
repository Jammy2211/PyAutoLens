import inspect
import logging
from functools import wraps

import os
import numba
import numpy as np
from astropy.io import fits

from autolens import exc

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


class Memoizer(object):

    def __init__(self):
        """
        Class to store the results of a function given a set of inputs.
        """
        self.results = {}
        self.calls = 0
        self.arg_names = None

    def __call__(self, func):
        """
        Memoize decorator. Any time a function is called that a memoizer has been attached to its results are stored in
        the results dictionary or retrieved from the dictionary if the function has already been called with those
        arguments.

        Note that the same memoizer persists over all instances of a class. Any state for a given instance that is not
        given in the representation of that instance will be ignored. That is, it is possible that the memoizer will
        give incorrect results if instance state does not affect __str__ but does affect the value returned by the
        memoized method.

        Parameters
        ----------
        func: function
            A function for which results should be memoized

        Returns
        -------
        decorated: function
            A function that memoizes results
        """
        if self.arg_names is not None:
            raise AssertionError("Instantiate a new Memoizer for each function")
        self.arg_names = inspect.getfullargspec(func).args

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = ", ".join(
                ["('{}', {})".format(arg_name, arg) for arg_name, arg in
                 list(zip(self.arg_names, args)) + [(k, v) for k, v in kwargs.items()]])
            if key not in self.results:
                self.calls += 1
            self.results[key] = func(*args, **kwargs)
            return self.results[key]

        return wrapper


@numba.jit(nopython=True, cache=True)
def total_image_pixels_from_mask(mask):
    """Compute the total number of unmasked _image pixels in a mask."""

    total_image_pixels = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                total_image_pixels += 1

    return total_image_pixels


@numba.jit(nopython=True, cache=True)
def total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size):
    """Compute the total number of sub-pixels in unmasked _image pixels in a mask."""
    return total_image_pixels_from_mask(mask) * sub_grid_size ** 2


@numba.jit(nopython=True, cache=True)
def total_border_pixels_from_mask(mask):
    """Compute the total number of border-pixels in a mask."""

    border_pixel_total = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                if mask[x + 1, y] or mask[x - 1, y] or mask[x, y + 1] or mask[x, y - 1] or \
                        mask[x + 1, y + 1] or mask[x + 1, y - 1] or mask[x - 1, y + 1] or mask[x - 1, y - 1]:
                    border_pixel_total += 1

    return border_pixel_total


@numba.jit(nopython=True, cache=True)
def image_grid_2d_from_shape_and_pixel_scale(shape, pixel_scale):
    """
    Computes the (x,y) arc second coordinates of every pixel in an _image of shape (rows, columns).

    Coordinates are defined from the top-left corner, such that the first pixel at location [0, 0] has negative x \
    and y values in arc seconds.
    """

    grid_2d = np.zeros((shape[0], shape[1], 2))

    x_cen = float(shape[0] - 1) / 2
    y_cen = float(shape[1] - 1) / 2

    for x in range(shape[0]):
        for y in range(shape[1]):
            grid_2d[x, y, 0] = (x - x_cen) * pixel_scale
            grid_2d[x, y, 1] = (y - y_cen) * pixel_scale

    return grid_2d


@numba.jit(nopython=True, cache=True)
def image_grid_1d_masked_from_mask_and_pixel_scale(mask, pixel_scale):
    """Compute a 1D grid of (x,y) coordinates, using the center of every unmasked pixel."""

    grid_2d = image_grid_2d_from_shape_and_pixel_scale(mask.shape, pixel_scale)

    total_image_pixels = total_image_pixels_from_mask(mask)
    image_grid = np.zeros(shape=(total_image_pixels, 2))
    pixel_count = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                image_grid[pixel_count, :] = grid_2d[x, y]
                pixel_count += 1

    return image_grid


@numba.jit(nopython=True, cache=True)
def sub_grid_1d_masked_from_mask_pixel_scale_and_sub_grid_size(mask, pixel_scale, sub_grid_size):
    """Compute a 1D grid of (x,y) sub-pixel coordinates, using the sub-pixel centers of every unmasked pixel."""

    total_sub_pixels = total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size)

    sub_grid = np.zeros(shape=(total_sub_pixels, 2))

    x_cen = float(mask.shape[0] - 1) / 2
    y_cen = float(mask.shape[1] - 1) / 2

    sub_index = 0

    sub_half = pixel_scale / 2
    sub_step = pixel_scale / (sub_grid_size + 1)

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):

            if not mask[x, y]:

                x_arcsec = (x - x_cen) * pixel_scale
                y_arcsec = (y - y_cen) * pixel_scale

                for x1 in range(sub_grid_size):
                    for y1 in range(sub_grid_size):
                        sub_grid[sub_index, 0] = x_arcsec - sub_half + (x1 + 1) * sub_step
                        sub_grid[sub_index, 1] = y_arcsec - sub_half + (y1 + 1) * sub_step
                        sub_index += 1

    return sub_grid


@numba.jit(nopython=True, cache=True)
def grid_to_pixel_from_mask(mask):
    """Compute a 1D array that maps every unmasked pixel to its corresponding 2d pixel using its (x,y) pixel indexes.

    For howtolens if pixel [2,5] corresponds to the second pixel on the 1D array, grid_to_pixel[1] = [2,5]"""

    total_image_pixels = total_image_pixels_from_mask(mask)
    grid_to_pixel = np.zeros(shape=(total_image_pixels, 2))
    pixel_count = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                grid_to_pixel[pixel_count, :] = x, y
                pixel_count += 1

    return grid_to_pixel


@numba.jit(nopython=True, cache=True)
def sub_to_image_from_mask(mask, sub_grid_size):
    """Compute a 1D array that maps every unmasked pixel's sub-pixel to its corresponding 1d _image-pixel.

    For howtolens, if sub-pixel 8 is in _image-pixel 1, sub_to_image[7] = 1."""

    total_sub_pixels = total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size)

    sub_to_image = np.zeros(shape=total_sub_pixels)
    image_index = 0
    sub_index = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                for x1 in range(sub_grid_size):
                    for y1 in range(sub_grid_size):
                        sub_to_image[sub_index] = image_index
                        sub_index += 1

                image_index += 1

    return sub_to_image


@numba.jit(nopython=True, cache=True)
def border_pixels_from_mask(mask):
    """Compute a 1D array listing all border pixel indexes in the mask. A border pixel is a pixel which is not fully \
    surrounding by False mask values i.e. it is on an edge."""

    border_pixel_total = total_border_pixels_from_mask(mask)

    border_pixels = np.zeros(border_pixel_total)
    border_index = 0
    image_index = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                if mask[x + 1, y] or mask[x - 1, y] or mask[x, y + 1] or mask[x, y - 1] or \
                        mask[x + 1, y + 1] or mask[x + 1, y - 1] or mask[x - 1, y + 1] or mask[x - 1, y - 1]:
                    border_pixels[border_index] = image_index
                    border_index += 1

                image_index += 1

    return border_pixels


@numba.jit(nopython=True, cache=True)
def border_sub_pixels_from_mask_pixel_scale_and_sub_grid_size(mask, pixel_scale, sub_grid_size):
    """Compute a 1D array listing all sub-pixel border pixel indexes in the mask. A border sub-pixel is a sub-pixel \
    whose _image pixel is not fully surrounded by False mask values and it is closest to the edge."""
    border_pixel_total = total_border_pixels_from_mask(mask)
    border_sub_pixels = np.zeros(border_pixel_total)

    image_index = 0

    x_cen = float(mask.shape[0] - 1) / 2
    y_cen = float(mask.shape[1] - 1) / 2

    sub_half = pixel_scale / 2
    sub_step = pixel_scale / (sub_grid_size + 1)

    border_index = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                if mask[x + 1, y] or mask[x - 1, y] or mask[x, y + 1] or mask[x, y - 1] or \
                        mask[x + 1, y + 1] or mask[x + 1, y - 1] or mask[x - 1, y + 1] or mask[x - 1, y - 1]:

                    x_arcsec = (x - x_cen) * pixel_scale
                    y_arcsec = (y - y_cen) * pixel_scale

                    sub_grid = np.zeros((sub_grid_size ** 2, 2))
                    sub_index = 0

                    for x1 in range(sub_grid_size):
                        for y1 in range(sub_grid_size):
                            sub_grid[sub_index, 0] = x_arcsec - sub_half + (x1 + 1) * sub_step
                            sub_grid[sub_index, 1] = y_arcsec - sub_half + (y1 + 1) * sub_step
                            sub_index += 1

                    sub_grid_radii = np.add(np.square(sub_grid[:, 0]), np.square(sub_grid[:, 1]))
                    border_sub_index = image_index * (sub_grid_size ** 2) + np.argmax(sub_grid_radii)
                    border_sub_pixels[border_index] = border_sub_index
                    border_index += 1

                image_index += 1

    return border_sub_pixels


@numba.jit(nopython=True, cache=True)
def mask_circular_from_shape_pixel_scale_and_radius(shape, pixel_scale, radius_arcsec, centre=(0.0, 0.0)):
    """Compute a circular mask from an input mask radius and _image shape."""

    mask = np.full(shape, True)

    x_cen = (float(mask.shape[0] - 1) / 2) + (centre[0] / pixel_scale)
    y_cen = (float(mask.shape[1] - 1) / 2) + (centre[1] / pixel_scale)

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):

            x_arcsec = (x - x_cen) * pixel_scale
            y_arcsec = (y - y_cen) * pixel_scale

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if r_arcsec <= radius_arcsec:
                mask[x, y] = False

    return mask


@numba.jit(nopython=True, cache=True)
def mask_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec,
                                                  centre=(0.0, 0.0)):
    """Compute an annular mask from an input inner and outer mask radius and _image shape."""

    mask = np.full(shape, True)

    x_cen = (float(mask.shape[0] - 1) / 2) + (centre[0] / pixel_scale)
    y_cen = (float(mask.shape[1] - 1) / 2) + (centre[1] / pixel_scale)

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):

            x_arcsec = (x - x_cen) * pixel_scale
            y_arcsec = (y - y_cen) * pixel_scale

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if outer_radius_arcsec >= r_arcsec >= inner_radius_arcsec:
                mask[x, y] = False

    return mask

@numba.jit(nopython=True, cache=True)
def mask_anti_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec,
                                                  outer_radius_2_arcsec, centre=(0.0, 0.0)):
    """Compute an annular mask from an input inner and outer mask radius and _image shape."""

    mask = np.full(shape, True)

    x_cen = (float(mask.shape[0] - 1) / 2) + (centre[0] / pixel_scale)
    y_cen = (float(mask.shape[1] - 1) / 2) + (centre[1] / pixel_scale)

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):

            x_arcsec = (x - x_cen) * pixel_scale
            y_arcsec = (y - y_cen) * pixel_scale

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if  inner_radius_arcsec >= r_arcsec or outer_radius_2_arcsec >= r_arcsec >= outer_radius_arcsec:
                mask[x, y] = False

    return mask

@numba.jit(nopython=True, cache=True)
def mask_blurring_from_mask_and_psf_shape(mask, psf_shape):
    """Compute a blurring mask from an input mask and psf shape.

    The blurring mask corresponds to all pixels which are outside of the mask but will have a fraction of their \
    light blur into the masked region due to PSF convolution."""

    blurring_mask = np.full(mask.shape, True)

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                for y1 in range((-psf_shape[1] + 1) // 2, (psf_shape[1] + 1) // 2):
                    for x1 in range((-psf_shape[0] + 1) // 2, (psf_shape[0] + 1) // 2):
                        if 0 <= x + x1 <= mask.shape[0] - 1 and 0 <= y + y1 <= mask.shape[1] - 1:
                            if mask[x + x1, y + y1]:
                                blurring_mask[x + x1, y + y1] = False
                        else:
                            raise exc.MaskException(
                                "setup_blurring_mask extends beyond the sub_grid_size of the mask - pad the "
                                "masked_image before masking")

    return blurring_mask


@numba.jit(nopython=True, cache=True)
def map_2d_array_to_masked_1d_array_from_array_2d_and_mask(mask, array_2d):
    """For a given 2D array and mask, mappers all unmasked pixels to a 1D array."""

    total_image_pixels = total_image_pixels_from_mask(mask)

    array_1d = np.zeros(shape=total_image_pixels)
    index = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                array_1d[index] = array_2d[x, y]
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
    for x in range(shape[0]):
        for y in range(shape[1]):
            array_2d[x, y] = array_1d[index]
            index += 1

    return array_2d


def trim_array_2d_around_centre(array_2d, new_shape):
    """
    Trim the data_vector array to a new sub_grid_size around its central pixel.

    NOTE: The centre of the array cannot be shifted. Therefore, even arrays must be trimmed to even arrays \
    (e.g. 8x8 -> 4x4) and odd to odd (e.g. 5x5 -> 3x3).

    Parameters
    ----------
    array_2d
    new_shape : (int, int)
        The (x,y) new pixel dimension of the trimmed data_vector-array.
    """

    if new_shape[0] > array_2d.shape[0]:
        raise ValueError(
            'grids.Grid2d.trim_data - You have specified a new x_size bigger than the data_vector array')
    elif new_shape[1] > array_2d.shape[1]:
        raise ValueError(
            'grids.Grid2d.trim_data - You have specified a new y_size bigger than the data_vector array')

    if array_2d.shape[0] % 2 == 0 and not new_shape[0] % 2 == 0:
        raise ValueError('You cannot trim an array from even shape to odd shape - change new_shape to even')

    if array_2d.shape[1] % 2 == 0 and not new_shape[1] % 2 == 0:
        raise ValueError('You cannot trim an array from even shape to odd shape - change new_shape to even')

    if not array_2d.shape[0] % 2 == 0 and new_shape[0] % 2 == 0:
        raise ValueError('You cannot trim an array from odd shape to even shape - change new_shape to odd')

    if not array_2d.shape[1] % 2 == 0 and new_shape[1] % 2 == 0:
        raise ValueError('You cannot trim an array from odd shape to even shape - change new_shape to odd')

    x_trim = int((array_2d.shape[0] - new_shape[0]) / 2)
    y_trim = int((array_2d.shape[1] - new_shape[1]) / 2)

    array = array_2d[x_trim:array_2d.shape[0] - x_trim, y_trim:array_2d.shape[1] - y_trim]

    return array


def trim_array_2d_around_region(array_2d, x0, x1, y0, y1):
    """
    Trim the data_vector array to a new sub_grid_size around its central pixel.

    NOTE: The centre of the array cannot be shifted. Therefore, even arrays must be trimmed to even arrays \
    (e.g. 8x8 -> 4x4) and odd to odd (e.g. 5x5 -> 3x3).

    Parameters
    ----------
    array_2d
    new_shape : (int, int)
        The (x,y) new pixel dimension of the trimmed data_vector-array.
    """

    if x1 > array_2d.shape[0]:
        raise ValueError(
            'grids.Grid2d.trim_data - You have specified a new x_size bigger than the data_vector array')
    elif y1 > array_2d.shape[1]:
        raise ValueError(
            'grids.Grid2d.trim_data - You have specified a new y_size bigger than the data_vector array')

    return array_2d[y0:y1, x0:x1]


def numpy_array_to_fits(array, path, overwrite=False):

    if overwrite and os.path.exists(path):
        os.remove(path)

    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(array, new_hdr)
    hdu.writeto(path)


def numpy_array_from_fits(path, hdu):
    hdu_list = fits.open(path)
    return np.array(hdu_list[hdu].data)


def compute_variances_from_noise(noise):
    """The variances are the signal_to_noise_ratio (standard deviations) squared."""
    return np.square(noise)

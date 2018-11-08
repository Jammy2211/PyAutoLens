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
def total_border_pixels_from_mask(mask):
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
def image_grid_2d_from_shape_and_pixel_scales(shape, pixel_scales):
    """
    Computes the (x,y) arc second coordinates of every pixel in an datas_ of shape (rows, columns).

    Coordinates are defined from the top-left corner, such that the first pixel at location [0, 0] has negative x \
    and y values in arc seconds.
    """

    grid_2d = np.zeros((shape[0], shape[1], 2))

    y_cen = float(shape[0] - 1) / 2
    x_cen = float(shape[1] - 1) / 2

    for y in range(shape[0]):
        for x in range(shape[1]):

            grid_2d[y, x, 0] = -(y - y_cen) * pixel_scales[0]
            grid_2d[y, x, 1] = (x - x_cen) * pixel_scales[1]

    return grid_2d


@numba.jit(nopython=True, cache=True)
def image_grid_1d_from_shape_and_pixel_scales(shape, pixel_scales):
    """
    Computes the (x,y) arc second coordinates of every pixel in an datas_ of shape (rows, columns).

    Coordinates are defined from the top-left corner, such that the first pixel at location [0, 0] has negative x \
    and y values in arc seconds.
    """

    grid_1d = np.zeros((shape[0]*shape[1], 2))

    y_cen = float(shape[0] - 1) / 2
    x_cen = float(shape[1] - 1) / 2

    i=0
    for y in range(shape[0]):
        for x in range(shape[1]):

            grid_1d[i, 0] = -(y - y_cen) * pixel_scales[0]
            grid_1d[i, 1] = (x - x_cen) * pixel_scales[1]
            i += 1

    return grid_1d

@numba.jit(nopython=True, cache=True)
def image_grid_1d_masked_from_mask_and_pixel_scales(mask, pixel_scales):
    """Compute a 1D grid of (x,y) coordinates, using the center of every unmasked pixel."""

    grid_2d = image_grid_2d_from_shape_and_pixel_scales(mask.shape, pixel_scales)

    total_image_pixels = total_image_pixels_from_mask(mask)
    image_grid = np.zeros(shape=(total_image_pixels, 2))
    pixel_count = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                image_grid[pixel_count, :] = grid_2d[y, x]
                pixel_count += 1

    return image_grid

@numba.jit(nopython=True, cache=True)
def sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask, pixel_scales, sub_grid_size):
    """Compute a 1D grid of (x,y) sub-pixel coordinates, using the sub-pixel centers of every unmasked pixel."""

    total_sub_pixels = total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size)

    sub_grid = np.zeros(shape=(total_sub_pixels, 2))

    y_cen = float(mask.shape[0] - 1) / 2
    x_cen = float(mask.shape[1] - 1) / 2

    sub_index = 0

    y_sub_half = pixel_scales[0] / 2
    y_sub_step = pixel_scales[0] / (sub_grid_size + 1)

    x_sub_half = pixel_scales[1] / 2
    x_sub_step = pixel_scales[1] / (sub_grid_size + 1)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            if not mask[y, x]:

                y_arcsec = (y - y_cen) * pixel_scales[0]
                x_arcsec = (x - x_cen) * pixel_scales[1]

                for y1 in range(sub_grid_size):
                    for x1 in range(sub_grid_size):

                        sub_grid[sub_index, 0] = -(y_arcsec - y_sub_half + (y1 + 1) * y_sub_step)
                        sub_grid[sub_index, 1] = x_arcsec - x_sub_half + (x1 + 1) * x_sub_step
                        sub_index += 1

    return sub_grid

@numba.jit(nopython=True, cache=True)
def grid_arc_seconds_1d_to_grid_pixels_1d(grid_arc_seconds, shape, pixel_scales):
    """ Converts a grid in coordinates of pixels to a grid in arc seconds.

    The pixel coordinate origin is at the top left corner of an image, whilst the arc-second coordinate origin \
    is at the centre start with negative x and y values from the top-left.

    This means that the top-left pixel coordinates, [0, 0], will give negative arc second coordinates.

    Parameters
    ----------
    grid_pixels : ndarray
        The grid of (x,y) coordinates in units of pixels
    """

    grid_pixels = np.zeros((grid_arc_seconds.shape[0], 2))

    y_cen = float(shape[0] - 1) / 2
    x_cen = float(shape[1] - 1) / 2

    for i in range(grid_arc_seconds.shape[0]):

        grid_pixels[i, 0] =(-grid_arc_seconds[i,0] / pixel_scales[0])  + y_cen + 0.5
        grid_pixels[i, 1] = (grid_arc_seconds[i,1] / pixel_scales[1])  + x_cen + 0.5

    return grid_pixels

@numba.jit(nopython=True, cache=True)
def grid_arc_seconds_1d_to_grid_pixel_centres_1d(grid_arc_seconds, shape, pixel_scales):
    """ Converts a grid in coordinates of pixels to a grid in arc seconds.

    The pixel coordinate origin is at the top left corner of an image, whilst the arc-second coordinate origin \
    is at the centre start with negative x and y values from the top-left.

    This means that the top-left pixel coordinates, [0, 0], will give negative arc second coordinates.

    Parameters
    ----------
    grid_pixels : ndarray
        The grid of (x,y) coordinates in units of pixels
    """

    grid_pixels = np.zeros((grid_arc_seconds.shape[0], 2))

    y_cen = float(shape[0] - 1) / 2
    x_cen = float(shape[1] - 1) / 2

    for i in range(grid_arc_seconds.shape[0]):

        grid_pixels[i, 0] = int((-grid_arc_seconds[i,0] / pixel_scales[0])  + y_cen + 0.5)
        grid_pixels[i, 1] = int((grid_arc_seconds[i,1] / pixel_scales[1])  + x_cen + 0.5)

    return grid_pixels

@numba.jit(nopython=True, cache=True)
def grid_pixels_1d_to_grid_arc_seconds_1d(grid_pixels, shape, pixel_scales):
    """ Converts a grid in coordinates of pixels to a grid in arc seconds.

    The pixel coordinate origin is at the top left corner of an image, whilst the arc-second coordinate origin \
    is at the centre start with negative x and y values from the top-left.

    This means that the top-left pixel coordinates, [0, 0], will give negative arc second coordinates.

    Parameters
    ----------
    grid_pixels : ndarray
        The grid of (x,y) coordinates in units of pixels
    """

    grid_arc_seconds = np.zeros((grid_pixels.shape[0], 2))

    y_cen = float(shape[0] - 1) / 2
    x_cen = float(shape[1] - 1) / 2

    for i in range(grid_arc_seconds.shape[0]):

        grid_arc_seconds[i, 0] = -(grid_pixels[i,0] - y_cen - 0.5) * pixel_scales[0]
        grid_arc_seconds[i, 1] = (grid_pixels[i,1] - x_cen - 0.5) * pixel_scales[1]

    return grid_arc_seconds

@numba.jit(nopython=True, cache=True)
def grid_to_pixel_from_mask(mask):
    """Compute a 1D array that maps every unmasked pixel to its corresponding 2d pixel using its (x,y) pixel indexes.

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
def sub_to_image_from_mask(mask, sub_grid_size):
    """Compute a 1D array that maps every unmasked pixel's sub-pixel to its corresponding 1d datas_-pixel.

    For howtolens, if sub-pixel 8 is in datas_-pixel 1, sub_to_image[7] = 1."""

    total_sub_pixels = total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size)

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
def border_pixels_from_mask(mask):
    """Compute a 1D array listing all border pixel indexes in the masks. A border pixel is a pixel which is not fully \
    surrounding by False masks values i.e. it is on an edge."""

    border_pixel_total = total_border_pixels_from_mask(mask)

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
def border_sub_pixels_from_mask_pixel_scales_and_sub_grid_size(mask, pixel_scales, sub_grid_size):
    """Compute a 1D array listing all sub-pixel border pixel indexes in the masks. A border sub-pixel is a sub-pixel \
    whose datas_ pixel is not fully surrounded by False masks values and it is closest to the edge."""
    border_pixel_total = total_border_pixels_from_mask(mask)
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
def map_2d_array_to_masked_1d_array_from_array_2d_and_mask(mask, array_2d):
    """For a given 2D array and masks, mappers all unmasked pixels to a 1D array."""

    total_image_pixels = total_image_pixels_from_mask(mask)

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
def resize_array_2d(array_2d, new_shape, new_centre=(-1, -1)):
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

    y_is_even = int(array_2d.shape[0]) % 2 == 0
    x_is_even = int(array_2d.shape[1]) % 2 == 0

    if new_centre is (-1, -1):

        if y_is_even:
            y_centre = int(array_2d.shape[0]/2)
        elif not y_is_even:
            y_centre = int(array_2d.shape[0]/2)

        if x_is_even:
            x_centre = int(array_2d.shape[1] / 2)
        elif not x_is_even:
            x_centre = int(array_2d.shape[1] / 2)

        new_centre = (y_centre, x_centre)

    resized_array = np.zeros(shape=new_shape)

    if y_is_even:
        ymin = new_centre[0] - int(new_shape[0]/2)
        ymax = new_centre[0] + int((new_shape[0]/2)) + 1
    elif not y_is_even:
        ymin = new_centre[0] - int(new_shape[0]/2)
        ymax = new_centre[0] + int((new_shape[0]/2)) + 1

    if x_is_even:
        xmin = new_centre[1] - int(new_shape[1] / 2)
        xmax = new_centre[1] + int((new_shape[1] / 2)) + 1
    elif not x_is_even:
        xmin = new_centre[1] - int(new_shape[1] / 2)
        xmax = new_centre[1] + int((new_shape[1] / 2)) + 1

    for y_resized, y in enumerate(range(ymin, ymax)):
        for x_resized, x in enumerate(range(xmin, xmax)):
            if y >= 0 and y < array_2d.shape[0] and x >= 0 and x < array_2d.shape[1]:
                if y_resized >= 0 and y_resized < new_shape[0] and x_resized >= 0 and x_resized < new_shape[1]:
                    resized_array[y_resized, x_resized] = array_2d[y,x]
            else:
                if y_resized >=0 and y_resized < new_shape[0] and x_resized >= 0 and x_resized < new_shape[1]:
                    resized_array[y_resized, x_resized] = 0.0

    return resized_array

def numpy_array_to_fits(array, path, overwrite=False):

    if overwrite and os.path.exists(path):
        os.remove(path)

    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(np.flipud(array), new_hdr)
    hdu.writeto(path)


def numpy_array_from_fits(path, hdu):
    hdu_list = fits.open(path)
    return np.flipud(np.array(hdu_list[hdu].data))


def compute_variances_from_noise(noise):
    """The variances are the signal_to_noise_ratio (standard deviations) squared."""
    return np.square(noise)

import logging

import os
import numba
import numpy as np
from astropy.io import fits

from autolens.imaging.util import mask_util

@numba.jit(nopython=True, cache=True)
def centres_from_shape_pixel_scales_and_origin(shape, pixel_scales, origin):

    y_cen = float(shape[0] - 1) / 2 + (origin[0] / pixel_scales[0])
    x_cen = float(shape[1] - 1) / 2 - (origin[1] / pixel_scales[1])

    return y_cen, x_cen

@numba.jit(nopython=True, cache=True)
def image_grid_2d_from_shape_pixel_scales_and_origin(shape, pixel_scales, origin=(0.0, 0.0)):
    """
    Computes the (y,x) arc second coordinates of every pixel in an datas_ of shape (rows, columns).

    Coordinates are defined from the top-left corner, such that the first pixel at location [0, 0] has negative x \
    and y values in arc seconds.
    """

    grid_2d = np.zeros((shape[0], shape[1], 2))

    y_cen, x_cen = centres_from_shape_pixel_scales_and_origin(shape=shape, pixel_scales=pixel_scales, origin=origin)

    for y in range(shape[0]):
        for x in range(shape[1]):

            grid_2d[y, x, 0] = -(y - y_cen) * pixel_scales[0]
            grid_2d[y, x, 1] = (x - x_cen) * pixel_scales[1]

    return grid_2d

@numba.jit(nopython=True, cache=True)
def image_grid_1d_from_shape_pixel_scales_and_origin(shape, pixel_scales, origin=(0.0, 0.0)):
    """
    Computes the (y,x) arc second coordinates of every pixel in an datas_ of shape (rows, columns).

    Coordinates are defined from the top-left corner, such that the first pixel at location [0, 0] has negative x \
    and y values in arc seconds.
    """

    grid_1d = np.zeros((shape[0]*shape[1], 2))

    y_cen, x_cen = centres_from_shape_pixel_scales_and_origin(shape=shape, pixel_scales=pixel_scales, origin=origin)

    i=0
    for y in range(shape[0]):
        for x in range(shape[1]):

            grid_1d[i, 0] = -(y - y_cen) * pixel_scales[0]
            grid_1d[i, 1] = (x - x_cen) * pixel_scales[1]
            i += 1

    return grid_1d

@numba.jit(nopython=True, cache=True)
def image_grid_1d_masked_from_mask_pixel_scales_and_origin(mask, pixel_scales, origin=(0.0, 0.0)):
    """Compute a 1D grid of (y,x) coordinates, using the center of every unmasked pixel."""

    grid_2d = image_grid_2d_from_shape_pixel_scales_and_origin(mask.shape, pixel_scales, origin)

    total_image_pixels = mask_util.total_image_pixels_from_mask(mask)
    image_grid = np.zeros(shape=(total_image_pixels, 2))
    pixel_count = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                image_grid[pixel_count, :] = grid_2d[y, x]
                pixel_count += 1

    return image_grid

@numba.jit(nopython=True, cache=True)
def sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask, pixel_scales, sub_grid_size, origin=(0.0, 0.0)):
    """Compute a 1D grid of (y,x) sub-pixel coordinates, using the sub-pixel centers of every unmasked pixel."""

    total_sub_pixels = mask_util.total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size)

    sub_grid = np.zeros(shape=(total_sub_pixels, 2))

    y_cen, x_cen = centres_from_shape_pixel_scales_and_origin(shape=mask.shape, pixel_scales=pixel_scales, origin=origin)

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
def grid_arc_seconds_1d_to_grid_pixels_1d(grid_arc_seconds, shape, pixel_scales, origin=(0.0, 0.0)):
    """ Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are
    returned as floats such that they include the decimal offset from each pixel's top-left corner.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
    higher y arc-second coordinate value and lowest x arc-second coordinate.

    The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
    origin before computing their 1D grid pixel indexes.

    Parameters
    ----------
    grid_arc_seconds: ndarray
        The grid of (y,x) coordinates in arc seconds.
    shape : (int, int)
        The (y,x) shape of the 2D grid the arc-second coordinates are converted to pixel indexes for.
    pixel_scales : (float, float)
        The (y,x) pixel scales of the 2D grid's array / pixels.
    origin : (float, flloat)
        The (y,x) origin of the 2D grid, which the arc-second grid is shifted too.
    """

    grid_pixels = np.zeros((grid_arc_seconds.shape[0], 2))

    y_cen, x_cen = centres_from_shape_pixel_scales_and_origin(shape=shape, pixel_scales=pixel_scales, origin=origin)

    for i in range(grid_arc_seconds.shape[0]):

        grid_pixels[i, 0] =(-grid_arc_seconds[i,0] / pixel_scales[0])  + y_cen + 0.5
        grid_pixels[i, 1] = (grid_arc_seconds[i,1] / pixel_scales[1])  + x_cen + 0.5

    return grid_pixels

@numba.jit(nopython=True, cache=True)
def grid_arc_seconds_1d_to_grid_pixel_centres_1d(grid_arc_seconds, shape, pixel_scales, origin=(0.0, 0.0)):
    """ Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
    returned as integers such that they map directly to the pixel they are contained within.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
    higher y arc-second coordinate value and lowest x arc-second coordinate.

    The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
    origin before computing their 1D grid pixel indexes.

    Parameters
    ----------
    grid_arc_seconds: ndarray
        The grid of (y,x) coordinates in arc seconds.
    shape : (int, int)
        The (y,x) shape of the 2D grid the arc-second coordinates are converted to pixel indexes for.
    pixel_scales : (float, float)
        The (y,x) pixel scales of the 2D grid's array / pixels.
    origin : (float, flloat)
        The (y,x) origin of the 2D grid, which the arc-second grid is shifted too.
    """

    grid_pixels = np.zeros((grid_arc_seconds.shape[0], 2))

    y_cen, x_cen = centres_from_shape_pixel_scales_and_origin(shape=shape, pixel_scales=pixel_scales, origin=origin)

    for i in range(grid_arc_seconds.shape[0]):

        grid_pixels[i, 0] = int((-grid_arc_seconds[i,0] / pixel_scales[0])  + y_cen + 0.5)
        grid_pixels[i, 1] = int((grid_arc_seconds[i,1] / pixel_scales[1])  + x_cen + 0.5)

    return grid_pixels

@numba.jit(nopython=True, cache=True)
def grid_arc_seconds_1d_to_grid_pixel_indexes_1d(grid_arc_seconds, shape, pixel_scales, origin=(0.0, 0.0)):
    """ Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel 1D indexes. Pixel coordinates are \
    returned as integers such that they are the pixel from the top-left of the 2D grid going rights and then \
    downwards.

    For example:

    The pixel at the top-left, whose 2D index is [0,0], corresponds to 1D index 0.
    The fifth pixel on the top row, whose 2D index is [0,5], corresponds to 1D index 4.
    The first pixel on the second row, whose 2D index is [0,1], has 1D index 10 if a row has 10 pixels.

    The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
    origin before computing their 1D grid pixel indexes.

    Parameters
    ----------
    grid_arc_seconds: ndarray
        The grid of (y,x) coordinates in arc seconds.
    shape : (int, int)
        The (y,x) shape of the 2D grid the arc-second coordinates are converted to pixel indexes for.
    pixel_scales : (float, float)
        The (y,x) pixel scales of the 2D grid's array / pixels.
    origin : (float, flloat)
        The (y,x) origin of the 2D grid, which the arc-second grid is shifted too.
    """

    grid_pixels = grid_arc_seconds_1d_to_grid_pixel_centres_1d(grid_arc_seconds=grid_arc_seconds, shape=shape,
                                                               pixel_scales=pixel_scales, origin=origin)

    grid_pixel_indexes = np.zeros(grid_pixels.shape[0])

    for i in range(grid_pixels.shape[0]):

        grid_pixel_indexes[i] = int(grid_pixels[i,0] * shape[1] + grid_pixels[i,1])

    return grid_pixel_indexes

@numba.jit(nopython=True, cache=True)
def grid_pixels_1d_to_grid_arc_seconds_1d(grid_pixels, shape, pixel_scales, origin=(0.0, 0.0)):
    """ Convert a grid of (y,x) pixel coordinates to a grid of (y,x) arc second values.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
    higher y arc-second coordinate value and lowest x arc-second coordinate.

    The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
    origin before computing their 1D grid pixel indexes.

    Parameters
    ----------
    grid_pixels : ndarray
        The grid of (y,x) coordinates in pixels.
    shape : (int, int)
        The (y,x) shape of the 2D grid the arc-second coordinates are converted to pixel indexes for.
    pixel_scales : (float, float)
        The (y,x) pixel scales of the 2D grid's array / pixels.
    origin : (float, flloat)
        The (y,x) origin of the 2D grid, which the arc-second grid is shifted too.
    """

    grid_arc_seconds = np.zeros((grid_pixels.shape[0], 2))

    y_cen, x_cen = centres_from_shape_pixel_scales_and_origin(shape=shape, pixel_scales=pixel_scales, origin=origin)

    for i in range(grid_arc_seconds.shape[0]):

        grid_arc_seconds[i, 0] = -(grid_pixels[i,0] - y_cen - 0.5) * pixel_scales[0]
        grid_arc_seconds[i, 1] = (grid_pixels[i,1] - x_cen - 0.5) * pixel_scales[1]

    return grid_arc_seconds
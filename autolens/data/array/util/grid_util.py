from autolens import decorator_util
import numpy as np
from skimage.transform import rescale

from autolens.data.array.util import mask_util


@decorator_util.jit()
def centres_from_shape_pixel_scales_and_origin(shape, pixel_scales, origin):
    """Determine the (y,x) arc-second central coordinates of an array from its shape, pixel-scales and origin.

     The coordinate system is defined such that the positive y axis is up and positive x axis is right.

    Parameters
     ----------
    shape : (int, int)
        The (y,x) shape of the 2D array the arc-second centre is computed for.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the 2D array.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the centre is shifted to.

    Returns
    --------
    tuple (float, float)
        The (y,x) arc-second central coordinates of the input array.

    Examples
    --------
    centres_arcsec = centres_from_shape_pixel_scales_and_origin(shape=(5,5), pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    y_centre_arcsec = float(shape[0] - 1) / 2 + (origin[0] / pixel_scales[0])
    x_centre_arcsec = float(shape[1] - 1) / 2 - (origin[1] / pixel_scales[1])

    return (y_centre_arcsec, x_centre_arcsec)

@decorator_util.jit()
def regular_grid_2d_from_shape_pixel_scales_and_origin(shape, pixel_scales, origin=(0.0, 0.0)):
    """Compute the (y,x) arc second coordinates at the centre of every pixel of an array of shape (rows, columns).

    Coordinates are defined from the top-left corner, such that the first pixel at location [0, 0] has negative x \
    and y values in arc seconds.

    The regular grid is returned on an array of shape (total_pixels, total_pixels, 2) where coordinate indexes match \
    those of the original 2D array. y coordinates are stored in the 0 index of the third dimension, x coordinates in \
    the 1 index.

    Parameters
     ----------
    shape : (int, int)
        The (y,x) shape of the 2D array the regular grid of coordinates is computed for.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the 2D array.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the regular grid is shifted around.

    Returns
    --------
    ndarray
        A regular grid of (y,x) arc-second coordinates at the centre of every pixel on a 2D array. The regular grid \
        array has dimensions (total_pixels, total_pixels, 2).

    Examples
    --------
    regular_grid_1d = regular_grid_2d_from_shape_pixel_scales_and_origin(shape=(5,5), pixel_scales=(0.5, 0.5), \
                                                                      origin=(0.0, 0.0))
    """

    regular_grid_2d = np.zeros((shape[0], shape[1], 2))

    centres_arcsec = centres_from_shape_pixel_scales_and_origin(shape=shape, pixel_scales=pixel_scales, origin=origin)

    for y in range(shape[0]):
        for x in range(shape[1]):

            regular_grid_2d[y, x, 0] = -(y - centres_arcsec[0]) * pixel_scales[0]
            regular_grid_2d[y, x, 1] = (x - centres_arcsec[1]) * pixel_scales[1]

    return regular_grid_2d

@decorator_util.jit()
def regular_grid_1d_from_shape_pixel_scales_and_origin(shape, pixel_scales, origin=(0.0, 0.0)):
    """Compute the (y,x) arc second coordinates at the centre of every pixel of an array of shape (rows, columns).

    Coordinates are defined from the top-left corner, such that the first pixel at location [0, 0] has negative x \
    and y values in arc seconds.

    The regular grid is returned on an array of shape (total_pixels**2, 2) where the 2D dimension of the original 2D \
    array are reduced to one dimension. y coordinates are stored in the 0 index of the second dimension, x coordinates
    in the 1 index.

    Parameters
     ----------
    shape : (int, int)
        The (y,x) shape of the 2D array the regular grid of coordinates is computed for.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the 2D array.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the regular grid is shifted around.

    Returns
    --------
    ndarray
        A regular grid of (y,x) arc-second coordinates at the centre of every pixel on a 2D array. The regular grid
        array has dimensions (total_pixels**2, 2).

    Examples
    --------
    regular_grid_1d = regular_grid_1d_from_shape_pixel_scales_and_origin(shape=(5,5), pixel_scales=(0.5, 0.5), \
                                                                      origin=(0.0, 0.0))
    """

    regular_grid_1d = np.zeros((shape[0]*shape[1], 2))

    centres_arcsec = centres_from_shape_pixel_scales_and_origin(shape=shape, pixel_scales=pixel_scales, origin=origin)

    i=0
    for y in range(shape[0]):
        for x in range(shape[1]):

            regular_grid_1d[i, 0] = -(y - centres_arcsec[0]) * pixel_scales[0]
            regular_grid_1d[i, 1] = (x - centres_arcsec[1]) * pixel_scales[1]
            i += 1

    return regular_grid_1d

@decorator_util.jit()
def regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask, pixel_scales, origin=(0.0, 0.0)):
    """Compute the (y,x) arc second coordinates at the centre of every pixel of a 2D mask array of shape (rows, columns).

    Coordinates are defined from the top-left corner, where the first unmasked pixel corresponds to index 0. The pixel \
    at the top-left of the array has negative x and y values in arc seconds.

    The regular grid is returned on an array of shape (total_unmasked_pixels, 2). y coordinates are stored in the 0 \
    index of the second dimension, x coordinates in the 1 index.

    Parameters
     ----------
    mask : ndarray
        A 2D array of bools, where *False* values mean unmasked and are therefore included as part of the calculated \
        regular grid.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the 2D mask array.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the regular grid is shifted around.

    Returns
    --------
    ndarray
        A regular grid of (y,x) arc-second coordinates at the centre of every pixel unmasked pixel on the 2D mask \
        array. The regular grid array has dimensions (total_unmasked_pixels, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    regular_grid_1d = regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=(0.5, 0.5),
                                                                            origin=(0.0, 0.0))
    """

    grid_2d = regular_grid_2d_from_shape_pixel_scales_and_origin(mask.shape, pixel_scales, origin)

    total_regular_pixels = mask_util.total_regular_pixels_from_mask(mask)
    regular_grid_1d = np.zeros(shape=(total_regular_pixels, 2))
    pixel_count = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                regular_grid_1d[pixel_count, :] = grid_2d[y, x]
                pixel_count += 1

    return regular_grid_1d

def sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask, pixel_scales, sub_grid_size, origin=(0.0, 0.0),
                                                                optimal_sub_grid=True):

    if optimal_sub_grid:
        return sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size_optimal_spacing(
            mask=mask, pixel_scales=pixel_scales, sub_grid_size=sub_grid_size, origin=origin)
    else:
        return sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size_non_optimal_spacing(
            mask=mask, pixel_scales=pixel_scales, sub_grid_size=sub_grid_size, origin=origin)

@decorator_util.jit()
def sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size_optimal_spacing(mask, pixel_scales, sub_grid_size,
                                                                                origin=(0.0, 0.0)):
    """ For the sub-grid, every unmasked pixel of a 2D mask array of shape (rows, columns) is divided into a finer \
    uniform grid of shape (sub_grid_size, sub_grid_size). This routine computes the (y,x) arc second coordinates at \
    the centre of every sub-pixel defined by this grid.

    Coordinates are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0. \
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second \
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_unmasked_pixels*sub_grid_size**2, 2). y coordinates are \
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Parameters
     ----------
    mask : ndarray
        A 2D array of bools, where *False* values mean unmasked and are therefore included as part of the calculated \
        regular grid.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the 2D mask array.
    sub_grid_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    --------
    ndarray
        A sub grid of (y,x) arc-second coordinates at the centre of every pixel unmasked pixel on the 2D mask \
        array. The sub grid array has dimensions (total_unmasked_pixels*sub_grid_size**2, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    sub_grid_1d = sub_grid_1d_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    total_sub_pixels = mask_util.total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size)

    sub_grid_1d = np.zeros(shape=(total_sub_pixels, 2))

    centres_arcsec = centres_from_shape_pixel_scales_and_origin(shape=mask.shape, pixel_scales=pixel_scales,
                                                                origin=origin)

    sub_index = 0

    y_sub_half = pixel_scales[0] / 2
    y_sub_step = pixel_scales[0] / (sub_grid_size)

    x_sub_half = pixel_scales[1] / 2
    x_sub_step = pixel_scales[1] / (sub_grid_size)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            if not mask[y, x]:

                y_arcsec = (y - centres_arcsec[0]) * pixel_scales[0]
                x_arcsec = (x - centres_arcsec[1]) * pixel_scales[1]

                for y1 in range(sub_grid_size):
                    for x1 in range(sub_grid_size):

                        sub_grid_1d[sub_index, 0] = -(y_arcsec - y_sub_half + y1 * y_sub_step + (y_sub_step/2.0))
                        sub_grid_1d[sub_index, 1] = x_arcsec - x_sub_half + x1 * x_sub_step + (x_sub_step/2.0)
                        sub_index += 1

    return sub_grid_1d


@decorator_util.jit()
def sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size_non_optimal_spacing(mask, pixel_scales, sub_grid_size,
                                                                                    origin=(0.0, 0.0)):
    """ For the sub-grid, every unmasked pixel of a 2D mask array of shape (rows, columns) is divided into a finer \
    uniform grid of shape (sub_grid_size, sub_grid_size). This routine computes the (y,x) arc second coordinates at \
    the centre of every sub-pixel defined by this grid.

    Coordinates are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0. \
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second \
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_unmasked_pixels*sub_grid_size**2, 2). y coordinates are \
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Parameters
     ----------
    mask : ndarray
        A 2D array of bools, where *False* values mean unmasked and are therefore included as part of the calculated \
        regular grid.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the 2D mask array.
    sub_grid_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin : (float, flloat)
        The (y,x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    --------
    ndarray
        A sub grid of (y,x) arc-second coordinates at the centre of every pixel unmasked pixel on the 2D mask \
        array. The sub grid array has dimensions (total_unmasked_pixels*sub_grid_size**2, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    sub_grid_1d = sub_grid_1d_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    total_sub_pixels = mask_util.total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size)

    sub_grid_1d = np.zeros(shape=(total_sub_pixels, 2))

    centres_arcsec = centres_from_shape_pixel_scales_and_origin(shape=mask.shape, pixel_scales=pixel_scales,
                                                                origin=origin)

    sub_index = 0

    y_sub_half = pixel_scales[0] / 2
    y_sub_step = pixel_scales[0] / (sub_grid_size + 1)

    x_sub_half = pixel_scales[1] / 2
    x_sub_step = pixel_scales[1] / (sub_grid_size + 1)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            if not mask[y, x]:

                y_arcsec = (y - centres_arcsec[0]) * pixel_scales[0]
                x_arcsec = (x - centres_arcsec[1]) * pixel_scales[1]

                for y1 in range(sub_grid_size):
                    for x1 in range(sub_grid_size):

                        sub_grid_1d[sub_index, 0] = -(y_arcsec - y_sub_half + (y1 + 1) * y_sub_step)
                        sub_grid_1d[sub_index, 1] = x_arcsec - x_sub_half + (x1 + 1) * x_sub_step
                        sub_index += 1

    return sub_grid_1d

@decorator_util.jit()
def grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d, shape, pixel_scales, origin=(0.0, 0.0)):
    """ Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel coordinate values. Pixel coordinates \
    are returned as floats such that they include the decimal offset from each pixel's top-left corner relative to \
    the input arc-second coordinate.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
    the highest y arc-second coordinate and lowest x arc-second coordinate on the gird.

    The arc-second grid is defined by an origin and coordinates are shifted to this origin before computing their \
    1D grid pixel coordinate values.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_arcsec_1d: ndarray
        The grid of (y,x) coordinates in arc seconds which is converted to pixel value coordinates.
    shape : (int, int)
        The (y,x) shape of the original 2D array the arc-second coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the arc-second grid is shifted to.

    Returns
    --------
    ndarray
        A grid of (y,x) pixel-value coordinates with dimensions (total_pixels, 2).

    Examples
    --------
    grid_arcsec_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_1d = grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=grid_arcsec_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_1d = np.zeros((grid_arcsec_1d.shape[0], 2))

    centres_arcsec = centres_from_shape_pixel_scales_and_origin(shape=shape, pixel_scales=pixel_scales, origin=origin)

    for i in range(grid_arcsec_1d.shape[0]):

        grid_pixels_1d[i, 0] = (-grid_arcsec_1d[i, 0] / pixel_scales[0]) + centres_arcsec[0] + 0.5
        grid_pixels_1d[i, 1] = (grid_arcsec_1d[i, 1] / pixel_scales[1]) + centres_arcsec[1] + 0.5

    return grid_pixels_1d

@decorator_util.jit()
def grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d, shape, pixel_scales, origin=(0.0, 0.0)):
    """ Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
    returned as integers such that they map directly to the pixel they are contained within.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
    higher y arc-second coordinate value and lowest x arc-second coordinate.

    The arc-second coordinate grid is defined by the class attribute origin, and coordinates are shifted to this \
    origin before computing their 1D grid pixel indexes.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_arcsec_1d: ndarray
        The grid of (y,x) coordinates in arc seconds which is converted to pixel indexes.
    shape : (int, int)
        The (y,x) shape of the original 2D array the arc-second coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the arc-second grid is shifted

    Returns
    --------
    ndarray
        A grid of (y,x) pixel indexes with dimensions (total_pixels, 2).

    Examples
    --------
    grid_arcsec_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_1d = grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_1d = np.zeros((grid_arcsec_1d.shape[0], 2))

    centres_arcsec = centres_from_shape_pixel_scales_and_origin(shape=shape, pixel_scales=pixel_scales, origin=origin)

    for i in range(grid_arcsec_1d.shape[0]):

        grid_pixels_1d[i, 0] = int((-grid_arcsec_1d[i, 0] / pixel_scales[0]) + centres_arcsec[0] + 0.5)
        grid_pixels_1d[i, 1] = int((grid_arcsec_1d[i, 1] / pixel_scales[1]) + centres_arcsec[1] + 0.5)

    return grid_pixels_1d

@decorator_util.jit()
def grid_arcsec_1d_to_grid_pixel_indexes_1d(grid_arcsec_1d, shape, pixel_scales, origin=(0.0, 0.0)):
    """ Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel 1D indexes. Pixel coordinates are \
    returned as integers such that they are the pixel from the top-left of the 2D grid going rights and then \
    downwards.

    For example:

    The pixel at the top-left, whose 2D index is [0,0], corresponds to 1D index 0.
    The fifth pixel on the top row, whose 2D index is [0,5], corresponds to 1D index 4.
    The first pixel on the second row, whose 2D index is [0,1], has 1D index 10 if a row has 10 pixels.

    The arc-second coordinate grid is defined by the class attribute origin, and coordinates are shifted to this \
    origin before computing their 1D grid pixel indexes.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_arcsec_1d: ndarray
        The grid of (y,x) coordinates in arc seconds which is converted to 1D pixel indexes.
    shape : (int, int)
        The (y,x) shape of the original 2D array the arc-second coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the arc-second grid is shifted.

    Returns
    --------
    ndarray
        A grid of 1d pixel indexes with dimensions (total_pixels, 2).

    Examples
    --------
    grid_arcsec_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_1d = grid_arcsec_1d_to_grid_pixel_indexes_1d(grid_arcsec_1d=grid_arcsec_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_1d = grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec_1d, shape=shape,
                                                               pixel_scales=pixel_scales, origin=origin)

    grid_pixel_indexes_1d = np.zeros(grid_pixels_1d.shape[0])

    for i in range(grid_pixels_1d.shape[0]):

        grid_pixel_indexes_1d[i] = int(grid_pixels_1d[i,0] * shape[1] + grid_pixels_1d[i,1])

    return grid_pixel_indexes_1d

@decorator_util.jit()
def grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d, shape, pixel_scales, origin=(0.0, 0.0)):
    """ Convert a grid of (y,x) pixel coordinates to a grid of (y,x) arc second values.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
    higher y arc-second coordinate value and lowest x arc-second coordinate.

    The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
    origin after computing their values from the 1D grid pixel indexes.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_pixels_1d: ndarray
        The grid of (y,x) coordinates in pixel values which is converted to arc-second coordinates.
    shape : (int, int)
        The (y,x) shape of the original 2D array the arc-second coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the arc-second grid is shifted.

    Returns
    --------
    ndarray
        A grid of 1d arc-second coordinates with dimensions (total_pixels, 2).

    Examples
    --------
    grid_pixels_1d = np.array([[0,0], [0,1], [1,0], [1,1])
    grid_pixels_1d = grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=grid_pixels_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_arcsec_1d = np.zeros((grid_pixels_1d.shape[0], 2))

    centres_arcsec = centres_from_shape_pixel_scales_and_origin(shape=shape, pixel_scales=pixel_scales, origin=origin)

    for i in range(grid_arcsec_1d.shape[0]):

        grid_arcsec_1d[i, 0] = -(grid_pixels_1d[i, 0] - centres_arcsec[0] - 0.5) * pixel_scales[0]
        grid_arcsec_1d[i, 1] = (grid_pixels_1d[i, 1] - centres_arcsec[1] - 0.5) * pixel_scales[1]

    return grid_arcsec_1d

@decorator_util.jit()
def grid_arcsec_2d_to_grid_pixel_centres_2d(grid_arcsec_2d, shape, pixel_scales, origin=(0.0, 0.0)):
    """ Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
    returned as integers such that they map directly to the pixel they are contained within.

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
    higher y arc-second coordinate value and lowest x arc-second coordinate.

    The arc-second coordinate grid is defined by the class attribute origin, and coordinates are shifted to this \
    origin before computing their 1D grid pixel indexes.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_arcsec_1d: ndarray
        The grid of (y,x) coordinates in arc seconds which is converted to pixel indexes.
    shape : (int, int)
        The (y,x) shape of the original 2D array the arc-second coordinates were computed on.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the original 2D array.
    origin : (float, flloat)
        The (y,x) origin of the grid, which the arc-second grid is shifted

    Returns
    --------
    ndarray
        A grid of (y,x) pixel indexes with dimensions (total_pixels, 2).

    Examples
    --------
    grid_arcsec_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_1d = grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec_1d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_2d = np.zeros((grid_arcsec_2d.shape[0], grid_arcsec_2d.shape[1], 2))

    centres_arcsec = centres_from_shape_pixel_scales_and_origin(shape=shape, pixel_scales=pixel_scales, origin=origin)

    for y in range(grid_arcsec_2d.shape[0]):
        for x in range(grid_arcsec_2d.shape[1]):
            grid_pixels_2d[y, x, 0] = int((-grid_arcsec_2d[y, x, 0] / pixel_scales[0]) + centres_arcsec[0] + 0.5)
            grid_pixels_2d[y, x, 1] = int((grid_arcsec_2d[y, x, 1] / pixel_scales[1]) + centres_arcsec[1] + 0.5)

    return grid_pixels_2d
def get_dimensions_pixels(array):
    """Get the x and y dimensions of a 2D data array in pixels"""

    if (array.ndim != 2):
        raise IndexError('ArrayTools.get_dimensions_pixels - array supplied to code was not 2D.')

    xdim = array.shape[0]
    ydim = array.shape[1]

    return xdim, ydim
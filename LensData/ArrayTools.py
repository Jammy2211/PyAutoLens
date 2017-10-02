def get_dimensions_pixels(array):
    """Get the x and y dimensions of a 2D data array in pixels"""

    if (array.ndim != 2):
        raise IndexError('ArrayTools.get_dimensions_pixels - array supplied to code was not 2D.')

    xdim = array.shape[0]
    ydim = array.shape[1]

    return xdim, ydim

def get_dimensions_arcsec(xdim, ydim, pixel_scale):
    """Get the x and y dimensions of a 2D array in arc seconds, given its number of pixels and pixel scale"""

    if (xdim <= 0):
        raise ValueError('ArrayTools.get_dimensions_arcsec - xdim = {} - negative x dimension supplied'.format(xdim))
    elif (ydim <= 0):
        raise ValueError('ArrayTools.get_dimensions_arcsec - ydim = {} - negative y dimension supplied'.format(ydim))
    elif (pixel_scale <= 0):
        raise ValueError('ArrayTools.get_dimensions_arcsec - pixel_scale = {} - negative pixel scale supplied'.format(pixel_scale))

    xarc = xdim*pixel_scale
    yarc = ydim*pixel_scale

    return xarc, yarc
import inspect
import os

from autolens import decorator_util
import numpy as np
from astropy.io import fits
from functools import wraps


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
        decorated : function
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


@decorator_util.jit()
def extracted_array_2d_from_array_2d_and_coordinates(array_2d, y0, y1, x0, x1):
    """Resize an array to a new size by extracting a sub-set of the array.

    The extracted input coordinates use NumPy convention, such that the upper values should be specified as +1 the \
    dimensions of the extracted array.

    In the example below, an array of size (5,5) is extracted using the coordinates y0=1, y1=4, x0=1, x1=4. This
    extracts an array of dimensions (3,3) and is equivalent to array_2d[1:4, 1:4].

    This function is necessary work with numba jit tags and is why a standard Numpy array extraction is not used.

    Parameters
    ----------
    array_2d : ndarray
        The 2D array that an array is extracted from.
    y0 : int
        The top row number (e.g. the higher y-coodinate) of the array that is extracted for the resize.
    y1 : int
        The bottom row number (e.g. the lower y-coodinate) of the array that is extracted for the resize.
    x0 : int
        The left column number (e.g. the lower x-coodinate) of the array that is extracted for the resize.
    x1 : int
        The right column number (e.g. the higher x-coodinate) of the array that is extracted for the resize.

    Returns
    -------
    ndarray
        The extracted 2D array from the input 2D array.

    Examples
    --------
    array_2d = np.ones((5,5))
    extracted_array = extract_array_2d(array_2d=array_2d, y0=1, y1=4, x0=1, x1=4)
    """

    new_shape = (y1-y0, x1-x0)

    resized_array = np.zeros(shape=new_shape)

    for y_resized, y in enumerate(range(y0, y1)):
        for x_resized, x in enumerate(range(x0, x1)):
                resized_array[y_resized, x_resized] = array_2d[y, x]

    return resized_array


@decorator_util.jit()
def resized_array_2d_from_array_2d_and_resized_shape(array_2d, resized_shape, origin=(-1, -1), pad_value=0.0):
    """Resize an array to a new size around a central pixel.

    If the origin (e.g. the central pixel) of the resized array is not specified, the central pixel of the array is \
    calculated automatically. For example, a (5,5) array's central pixel is (2,2). For even dimensions the central \
    pixel is assumed to be the lower indexed value, e.g. a (6,4) array's central pixel is calculated as (2,1).

    The default origin is (-1, -1) because numba requires that the function input is the same type throughout the \
    function, thus a default 'None' value cannot be used.

    Parameters
    ----------
    array_2d : ndarray
        The 2D array that is resized.
    resized_shape : (int, int)
        The (y,x) new pixel dimension of the trimmed array.
    origin : (int, int)
        The oigin of the resized array, e.g. the central pixel around which the array is extracted.
    pad_value : float
        If the reszied array is bigger in size than the input array, the value the padded edge values are filled in \
        using.

    Returns
    -------
    ndarray
        The resized 2D array from the input 2D array.

    Examples
    --------
    array_2d = np.ones((5,5))
    resize_array = resize_array_2d(array_2d=array_2d, new_shape=(2,2), origin=(2, 2))
    """

    y_is_even = int(array_2d.shape[0]) % 2 == 0
    x_is_even = int(array_2d.shape[1]) % 2 == 0

    if origin is (-1, -1):

        if y_is_even:
            y_centre = int(array_2d.shape[0] / 2)
        elif not y_is_even:
            y_centre = int(array_2d.shape[0] / 2)

        if x_is_even:
            x_centre = int(array_2d.shape[1] / 2)
        elif not x_is_even:
            x_centre = int(array_2d.shape[1] / 2)

        origin = (y_centre, x_centre)

    resized_array = np.zeros(shape=resized_shape)

    if y_is_even:
        y_min = origin[0] - int(resized_shape[0] / 2)
        y_max = origin[0] + int((resized_shape[0] / 2)) + 1
    elif not y_is_even:
        y_min = origin[0] - int(resized_shape[0] / 2)
        y_max = origin[0] + int((resized_shape[0] / 2)) + 1

    if x_is_even:
        x_min = origin[1] - int(resized_shape[1] / 2)
        x_max = origin[1] + int((resized_shape[1] / 2)) + 1
    elif not x_is_even:
        x_min = origin[1] - int(resized_shape[1] / 2)
        x_max = origin[1] + int((resized_shape[1] / 2)) + 1

    for y_resized, y in enumerate(range(y_min, y_max)):
        for x_resized, x in enumerate(range(x_min, x_max)):
            if y >= 0 and y < array_2d.shape[0] and x >= 0 and x < array_2d.shape[1]:
                if y_resized >= 0 and y_resized < resized_shape[0] and x_resized >= 0 and x_resized < resized_shape[1]:
                    resized_array[y_resized, x_resized] = array_2d[y, x]
            else:
                if y_resized >= 0 and y_resized < resized_shape[0] and x_resized >= 0 and x_resized < resized_shape[1]:
                    resized_array[y_resized, x_resized] = pad_value

    return resized_array

@decorator_util.jit()
def replace_noise_map_2d_values_where_image_2d_values_are_negative(image_2d, noise_map_2d, target_signal_to_noise=2.0):
    """If the values of a 2D image array are negative, this function replaces the corresponding 2D noise-map array \
    values to meet a specified target to noise value.

    This routine is necessary because of anomolous values in images which come from our HST ACS data-reduction \
    pipeline, where image-pixels with negative values (e.g. due to the background sky subtraction) have extremely \
    small noise values, which inflate their signal-to-noise values and chi-squared contributions in the modeling.

    Parameters
    ----------
    image_2d : ndarray
        The 2D image array used to locate the pixel indexes in the noise-map which are replaced.
    noise_map_2d : ndarray
        The 2D noise-map array whose values are replaced.
    target_signal_to_noise : float
        The target signal-to-noise the noise-map valueus are changed to.

    Returns
    -------
    ndarray
        The 2D noise-map with values changed.

    Examples
    --------
    image_2d = np.ones((5,5))
    image_2d[2,2] = -1.0
    noise_map_2d = np.ones((5,5))

    noise_map_2d_replaced = replace_noise_map_2d_values_where_image_2d_values_are_negative(
        image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=2.0):
    """
    for y in range(image_2d.shape[0]):
        for x in range(image_2d.shape[1]):
            if image_2d[y, x] < 0.0:
                absolute_signal_to_noise = np.abs(image_2d[y, x]) / noise_map_2d[y, x]
                if absolute_signal_to_noise >= target_signal_to_noise:
                    noise_map_2d[y,x] = np.abs(image_2d[y, x]) / target_signal_to_noise

    return noise_map_2d


def numpy_array_2d_to_fits(array_2d, file_path, overwrite=False):
    """Write a 2D NumPy array to a .fits file.

    Before outputting a NumPy array, the array is flipped upside-down using np.flipud. This is so that the arrays \
    appear the same orientation as .fits files loaded in DS9.

    Parameters
    ----------
    array_2d : ndarray
        The 2D array that is written to fits.
    file_path : str
        The full path of the file that is output, including the file name and '.fits' extension.
    overwrite : bool
        If True and a file already exists with the input file_path the .fits file is overwritten. If False, an error \
        will be raised.

    Returns
    -------
    None

    Examples
    --------
    array_2d = np.ones((5,5))
    numpy_array_to_fits(array_2d=array_2d, file_path='/path/to/file/filename.fits', overwrite=True)
    """
    if overwrite and os.path.exists(file_path):
        os.remove(file_path)

    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(np.flipud(array_2d), new_hdr)
    hdu.writeto(file_path)


def numpy_array_2d_from_fits(file_path, hdu):
    """Read a 2D NumPy array to a .fits file.

    After loading the NumPy array, the array is flipped upside-down using np.flipud. This is so that the arrays \
    appear the same orientation as .fits files loaded in DS9.

    Parameters
    ----------
    file_path : str
        The full path of the file that is loaded, including the file name and '.fits' extension.
    hdu : int
        The HDU extension of the array that is loaded from the .fits file.

    Returns
    -------
    ndarray
        The NumPy array that is loaded from the .fits file.

    Examples
    --------
    array_2d = numpy_array_from_fits(file_path='/path/to/file/filename.fits', hdu=0)
    """
    hdu_list = fits.open(file_path)
    return np.flipud(np.array(hdu_list[hdu].data))
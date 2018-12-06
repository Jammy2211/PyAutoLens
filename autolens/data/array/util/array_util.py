import inspect
from functools import wraps

import os
import numba
import numpy as np
from astropy.io import fits

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
def resize_array_2d(array_2d, new_shape, origin=(-1, -1)):
    """Resize an array to a new size around its a central pixel defined by the array's origin..

    Parameters
    ----------
    array_2d : ndarray
        The 2D array that is to be resized.
    new_shape : (int, int)
        The (y,x) new pixel dimension of the trimmed array-array.
    origin : (int, int)
        The new centre of the resized array
    """

    y_is_even = int(array_2d.shape[0]) % 2 == 0
    x_is_even = int(array_2d.shape[1]) % 2 == 0

    if origin is (-1, -1):

        if y_is_even:
            y_centre = int(array_2d.shape[0]/2)
        elif not y_is_even:
            y_centre = int(array_2d.shape[0]/2)

        if x_is_even:
            x_centre = int(array_2d.shape[1] / 2)
        elif not x_is_even:
            x_centre = int(array_2d.shape[1] / 2)

        origin = (y_centre, x_centre)

    resized_array = np.zeros(shape=new_shape)

    if y_is_even:
        ymin = origin[0] - int(new_shape[0] / 2)
        ymax = origin[0] + int((new_shape[0] / 2)) + 1
    elif not y_is_even:
        ymin = origin[0] - int(new_shape[0] / 2)
        ymax = origin[0] + int((new_shape[0] / 2)) + 1

    if x_is_even:
        xmin = origin[1] - int(new_shape[1] / 2)
        xmax = origin[1] + int((new_shape[1] / 2)) + 1
    elif not x_is_even:
        xmin = origin[1] - int(new_shape[1] / 2)
        xmax = origin[1] + int((new_shape[1] / 2)) + 1

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

import logging

import numpy as np

from autolens.imaging import imaging_util

logging.basicConfig()
logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class ArrayGeometry(object):
    pixel_scales = None

    @property
    def shape_arc_seconds(self):
        return (float(self.pixel_scales[0] * self.shape[0]), float(self.pixel_scales[1] * self.shape[1]))

    @property
    def central_pixel_coordinates(self):
        return float(self.shape[0] - 1) / 2, float(self.shape[1] - 1) / 2

    def grid_arc_seconds_to_grid_pixels(self, grid_arc_seconds):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are
        returned as floats such that they include the decimal offset from each pixel's top-left corner.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        higher y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is at the centre and will correspond to the dimensions of each image divided \
        by two.

        Parameters
        ----------
        grid_arc_seconds: ndarray
            The grid of (y,x) coordinates in arc seconds.
        """
        return imaging_util.grid_arc_seconds_1d_to_grid_pixels_1d(grid_arc_seconds=grid_arc_seconds, shape=self.shape,
                                                                  pixel_scales=self.pixel_scales)

    def grid_arc_seconds_to_grid_pixel_centres(self, grid_arc_seconds):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
        returned as integers such that they map directly to the pixel they are contained within.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        higher y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is at the centre and will correspond to the dimensions of each image divided \
        by two.

        Parameters
        ----------
        grid_arc_seconds: ndarray
            The grid of (y,x) coordinates in arc seconds.
        """
        return imaging_util.grid_arc_seconds_1d_to_grid_pixel_centres_1d(grid_arc_seconds=grid_arc_seconds,
                                                                         shape=self.shape,
                                                                         pixel_scales=self.pixel_scales)

    def grid_pixels_to_grid_arc_seconds(self, grid_pixels):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are
        returned as floats such that they include the decimal offset from each pixel's top-left corner.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        higher y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is at the centre and will correspond to the dimensions of each image divided \
        by two.

        Parameters
        ----------
        grid_arc_seconds: ndarray
            The grid of (y,x) coordinates in arc seconds.
        """
        return imaging_util.grid_pixels_1d_to_grid_arc_seconds_1d(grid_pixels=grid_pixels, shape=self.shape,
                                                                  pixel_scales=self.pixel_scales)

    @property
    def grid_1d(self):
        """ The arc second-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value y value in arc seconds.
        """
        return imaging_util.image_grid_1d_from_shape_and_pixel_scales(shape=self.shape, pixel_scales=self.pixel_scales)

    @property
    def grid_2d(self):
        """ The arc second-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value y value in arc seconds.
        """
        return imaging_util.image_grid_2d_from_shape_and_pixel_scales(shape=self.shape, pixel_scales=self.pixel_scales)

    @property
    def arc_second_maxima(self):
        return (self.shape_arc_seconds[0]) / 2.0, (self.shape_arc_seconds[1]) / 2.0

    @property
    def arc_second_minima(self):
        return -(self.shape_arc_seconds[0]) / 2.0, -(self.shape_arc_seconds[1]) / 2.0

    @property
    def yticks(self):
        """Compute the yticks labels of this grid, used for plotting the y-axis ticks when visualizing an _data-grid"""
        return np.linspace(-self.shape_arc_seconds[0] / 2.0, self.shape_arc_seconds[0] / 2.0, 4)

    @property
    def xticks(self):
        """Compute the xticks labels of this grid, used for plotting the x-axis ticks when visualizing an _data-grid"""
        return np.linspace(-self.shape_arc_seconds[1] / 2.0, self.shape_arc_seconds[1] / 2.0, 4)


class Array(np.ndarray):

    def __new__(cls, array, *args, **kwargs):
        return np.array(array, dtype='float64').view(cls)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Array, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super(Array, self).__setstate__(state[0:-1])

    def resize_around_centre(self, new_shape):

        if new_shape[0] < self.shape[0] and new_shape[1] < self.shape[1]:
            return self.trim_around_centre(new_shape=new_shape)
        elif new_shape[0] > self.shape[0] and new_shape[1] > self.shape[1]:
            return self.pad_around_centre(new_shape=new_shape)
        else:
            raise ValueError('resize_around_centre - one dimensions of the new_shape pads the image, whereas the other'
                             'trims it - make the trimmed dimension larger.')

    def pad_around_centre(self, new_shape):
        """Pad the array to a new shape.

        Parameters
        -----------
        new_shape : (int, int)
            The new two-dimensional shape of the array.
        """
        return self.new_with_array(imaging_util.pad_array_2d_around_centre(self, new_shape))

    def trim_around_centre(self, new_shape):
        """Trim the array to a new shape.

        Parameters
        -----------
        new_shape : (int, int)
            The new two-dimensional shape of the array.
        """
        return self.new_with_array(imaging_util.trim_array_2d_around_centre(self, new_shape))

    def trim_around_region(self, x0, x1, y0, y1):
        """Trim the array to a new shape.
        """
        return self.new_with_array(imaging_util.trim_array_2d_around_region(self, x0, x1, y0, y1))

    def new_with_array(self, array):
        """
        Parameters
        ----------
        array: ndarray
            An ndarray

        Returns
        -------
        new_array: ScaledSquarePixelArray
            A new instance of this class that shares all of this instances attributes with a new ndarray.
        """
        arguments = vars(self)
        arguments.update({"array": array})

        return self.__class__(**arguments)

    @classmethod
    def from_fits(cls, file_path, hdu):
        """
        Loads the data from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the _data data.
        """
        return cls(imaging_util.numpy_array_from_fits(file_path, hdu))


class ScaledArray(Array, ArrayGeometry):

    # noinspection PyUnusedLocal
    def __init__(self, array):
        """
        Parameters
        ----------
        array: ndarray
            An array representing data (e.g. an _data, noise-mappers, etc.)
        """
        # noinspection PyArgumentList
        super(ScaledArray, self).__init__()

    def map(self, func):
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                func(y, x)


class ScaledSquarePixelArray(ScaledArray):
    """
    Class storing the grids for 2D pixel grids (e.g. _data, PSF, signal_to_noise_ratio).
    """

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scale):
        """
        Parameters
        ----------
        array: ndarray
            An array representing data (e.g. an _data, noise-mappers, etc.)
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        # noinspection PyArgumentList
        self.pixel_scale = pixel_scale
        super(ScaledSquarePixelArray, self).__init__(array=array)

    def __array_finalize__(self, obj):
        if hasattr(obj, "pixel_scale"):
            self.pixel_scale = obj.pixel_scale

    @property
    def pixel_scales(self):
        return self.pixel_scale, self.pixel_scale

    @classmethod
    def from_fits_with_pixel_scale(cls, file_path, hdu, pixel_scale):
        """
        Loads the data from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the _data data.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls(imaging_util.numpy_array_from_fits(file_path, hdu), pixel_scale)

    @classmethod
    def single_value(cls, value, shape, pixel_scale):
        """
        Creates an instance of Array and fills it with a single value

        Parameters
        ----------
        value: float
            The value with which the array should be filled
        shape: (int, int)
            The shape of the array
        pixel_scale: float
            The scale of a pixel in arc seconds

        Returns
        -------
        array: ScaledSquarePixelArray
            An array filled with a single value
        """
        array = np.ones(shape) * value
        return cls(array, pixel_scale)

    def flatten(self, order='C'):
        """
        Returns
        -------
        flat_scaled_array: ScaledSquarePixelArray
            A copy of this array flattened to 1D
        """
        return self.new_with_array(super(ScaledSquarePixelArray, self).flatten(order))

    def __eq__(self, other):
        super_result = super(ScaledSquarePixelArray, self).__eq__(other)
        try:
            return super_result.all()
        except AttributeError:
            return super_result


class ScaledRectangularPixelArray(ScaledArray):
    """
    Class storing the grids for 2D pixel grids (e.g. _data, PSF, signal_to_noise_ratio).
    """

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scales):
        """
        Parameters
        ----------
        array: ndarray
            An array representing data (e.g. an _data, noise-mappers, etc.)
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        self.pixel_scales = pixel_scales
        # noinspection PyArgumentList
        super(ScaledRectangularPixelArray, self).__init__(array=array)

    def __array_finalize__(self, obj):
        if hasattr(obj, "pixel_scales"):
            self.pixel_scales = obj.pixel_scales

    @classmethod
    def single_value(cls, value, shape, pixel_scales):
        """
        Creates an instance of Array and fills it with a single value

        Parameters
        ----------
        value: float
            The value with which the array should be filled
        shape: (int, int)
            The shape of the array
        pixel_scales: (float, float)
            The scale of a pixel in arc seconds

        Returns
        -------
        array: ScaledSquarePixelArray
            An array filled with a single value
        """
        array = np.ones(shape) * value
        return cls(array, pixel_scales)

    @classmethod
    def from_fits_with_pixel_scale(cls, file_path, hdu, pixel_scales):
        """
        Loads the data from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the _data data.
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls(imaging_util.numpy_array_from_fits(file_path, hdu), pixel_scales)

    def flatten(self, order='C'):
        """
        Returns
        -------
        flat_scaled_array: ScaledSquarePixelArray
            A copy of this array flattened to 1D
        """
        return self.new_with_array(super().flatten(order))

    def __eq__(self, other):
        super_result = super(ScaledRectangularPixelArray, self).__eq__(other)
        try:
            return super_result.all()
        except AttributeError:
            return super_result

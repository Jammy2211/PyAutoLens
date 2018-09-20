import numpy as np
import logging

from autolens.imaging import imaging_util

logging.basicConfig()
logger = logging.getLogger(__name__)


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

    def trim(self, new_shape):
        """Trim the array to a new shape.
        
        Parameters
        -----------
        new_shape : (int, int)
            The new two-dimensional shape of the array.
        """
        return self.new_with_array(imaging_util.trim_array_2d_to_new_shape(self, new_shape))

    def new_with_array(self, array):
        """
        Parameters
        ----------
        array: ndarray
            An ndarray

        Returns
        -------
        new_array: ScaledArray
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
            The HDU number in the fits file containing the image data.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls(imaging_util.numpy_array_from_fits(file_path, hdu))


class ScaledArray(Array):
    """
    Class storing the grids for 2D pixel grids (e.g. image, PSF, signal_to_noise_ratio).
    """

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scale=1):
        """
        Parameters
        ----------
        array: ndarray
            An array representing data (e.g. an image, noise-mappers, etc.)
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        # noinspection PyArgumentList
        super(ScaledArray, self).__init__()
        self.pixel_scale = pixel_scale

    def __array_finalize__(self, obj):
        if isinstance(obj, ScaledArray):
            self.pixel_scale = obj.pixel_scale

    def map(self, func):
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                func(x, y)

    @classmethod
    def from_fits_with_scale(cls, file_path, hdu, pixel_scale):
        """
        Loads the data from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image data.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls(imaging_util.numpy_array_from_fits(file_path, hdu), pixel_scale)

    @property
    def central_pixel_coordinates(self):
        """
        Returns
        -------
        central_pixel_coordinates:
            The coordinates of the central pixel in the image. If a dimension of the image are odd then the \
            corresponding coordinate will be fractional values in the centre.
        """
        return float(self.shape[0] - 1) / 2, float(self.shape[1] - 1) / 2

    def pixels_to_arc_seconds(self, pixels):
        """Converts coordinate values from pixels to arc seconds."""
        return self.pixel_scale * pixels

    def arc_seconds_to_pixels(self, arc_seconds):
        """Converts coordinate values from arc seconds to pixels."""
        return arc_seconds / self.pixel_scale

    def pixel_coordinates_to_arc_second_coordinates(self, pixel_coordinates):
        """ Converts a pixel coordinate pair to an arc seconds coordinate pair.

        The pixel coordinate origin is at the top left corner of the image, whilst the arc-second coordinate origin \
        is at the centre start with negative x and y values from the top-left.

        This means that the top-left pixel coordinates, [0, 0], will give negative arc second coordinates.

        Parameters
        ----------
        pixel_coordinates: (float, float)
            The coordinates of a point in pixels

        Returns
        -------
        arc_second_coordinates: (float, float)
            The coordinates of a point in arc seconds
        """
        return tuple(map(lambda coord, centre: self.pixels_to_arc_seconds(coord - centre), pixel_coordinates,
                         self.central_pixel_coordinates))

    def arc_second_coordinates_to_pixel_coordinates(self, arc_second_coordinates):
        """
        Converts an arc second coordinate pair to a pixel coordinate pair.

        The pixel coordinate origin is at the top left corner of the image, whilst the arc-second coordinate origin \
        is at the centre start with negative x and y values from the top-left.

        This means that the top-left pixel coordinates, [0, 0], will give negative arc second coordinates.

        Parameters
        ----------
        arc_second_coordinates: (float, float)
            The coordinates of a point in arc seconds

        Returns
        -------
        pixel_coordinates: (float, float)
            The coordinates of a point in pixels
        """
        return tuple(map(lambda coord, centre: self.arc_seconds_to_pixels(coord) + centre,
                         arc_second_coordinates,
                         self.central_pixel_coordinates))

    @property
    def shape_arc_seconds(self):
        """The shape of the image in arc seconds"""
        return tuple(map(lambda d: self.pixels_to_arc_seconds(d), self.shape))

    def flatten(self, order='C'):
        """
        Returns
        -------
        flat_scaled_array: ScaledArray
            A copy of this array flattened to 1D
        """
        return self.new_with_array(super(ScaledArray, self).flatten(order))

    def sub_pixel_to_coordinate(self, sub_pixel, arcsec, sub_grid_size):
        """Convert a sub-pixel coordinate in an image-pixel to a sub-coordinate, using the pixel scale sub_grid_size."""

        half = self.pixel_scale / 2
        step = self.pixel_scale / (sub_grid_size + 1)

        return arcsec - half + (sub_pixel + 1) * step

    @property
    def grid_2d(self):
        """ The arc second-grid of (x,y) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value y value in arc seconds.
        """
        return imaging_util.image_grid_2d_from_shape_and_pixel_scale(self.shape, self.pixel_scale)

    @classmethod
    def single_value(cls, value, shape, pixel_scale=1):
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
        array: ScaledArray
            An array filled with a single value
        """
        array = np.ones(shape) * value
        return cls(array, pixel_scale)

    def __eq__(self, other):
        super_result = super(ScaledArray, self).__eq__(other)
        try:
            return super_result.all()
        except AttributeError:
            return super_result

    @property
    def xticks(self):
        """Compute the xticks labels of this grid, used for plotting the x-axis ticks when visualizing an image-grid"""
        return np.around(np.linspace(-self.shape_arc_seconds[1]/2.0, self.shape_arc_seconds[1]/2.0, 4), 2)

    @property
    def yticks(self):
        """Compute the yticks labels of this grid, used for plotting the y-axis ticks when visualizing an image-grid"""
        return np.around(np.linspace(-self.shape_arc_seconds[0]/2.0, self.shape_arc_seconds[0]/2.0, 4), 2)
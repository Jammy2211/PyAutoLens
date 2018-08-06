import numpy as np
import logging

from autolens.imaging import array_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractArray(np.ndarray):
    def __new__(cls, array, *args, **kwargs):
        return np.array(array, dtype='float64').view(cls)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(AbstractArray, self).__reduce__()
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
        super(AbstractArray, self).__setstate__(state[0:-1])

    @classmethod
    def from_fits(cls, file_path, hdu, pixel_scale):
        """
        Loads the weighted_data from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image weighted_data.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls(array_util.numpy_array_from_fits(file_path, hdu), pixel_scale)

    def pad(self, new_dimensions, pad_value=0):
        """ Pad the weighted_data array with zeros (or an input value) around its central pixel.

        NOTE: The centre of the array cannot be shifted. Therefore, even arrays must be padded to even arrays \
        (e.g. 8x8 -> 4x4) and odd to odd (e.g. 5x5 -> 3x3).

        Parameters
        ----------
        new_dimensions : (int, int)
            The (x,y) new pixel dimension of the padded weighted_data-array.
        pad_value : float
            The value to pad the array with.
        """
        if new_dimensions[0] < self.shape[0]:
            raise ValueError('grids.Grid2d.pad - You have specified a new x_size smaller than the weighted_data array')
        elif new_dimensions[1] < self.shape[1]:
            raise ValueError('grids.Grid2d.pad - You have specified a new y_size smaller than the weighted_data array')

        x_pad = int((new_dimensions[0] - self.shape[0] + 1) / 2)
        y_pad = int((new_dimensions[1] - self.shape[1] + 1) / 2)

        array = np.pad(self, ((x_pad, y_pad), (x_pad, y_pad)), 'constant', constant_values=pad_value)

        return self.new_with_array(array)

    def trim(self, new_dimensions):
        """
        Trim the weighted_data array to a new sub_grid_size around its central pixel.

        NOTE: The centre of the array cannot be shifted. Therefore, even arrays must be trimmed to even arrays \
        (e.g. 8x8 -> 4x4) and odd to odd (e.g. 5x5 -> 3x3).

        Parameters
        ----------
        new_dimensions : (int, int)
            The (x,y) new pixel dimension of the trimmed weighted_data-array.
        """
        if new_dimensions[0] > self.shape[0]:
            raise ValueError(
                'grids.Grid2d.trim_data - You have specified a new x_size bigger than the weighted_data array')
        elif new_dimensions[1] > self.shape[1]:
            raise ValueError(
                'grids.Grid2d.trim_data - You have specified a new y_size bigger than the weighted_data array')

        x_trim = int((self.shape[0] - new_dimensions[0]) / 2)
        y_trim = int((self.shape[1] - new_dimensions[1]) / 2)

        array = self[x_trim:self.shape[0] - x_trim, y_trim:self.shape[1] - y_trim]

        if self.shape[0] != new_dimensions[0]:
            logger.debug(
                'image.weighted_data.trim_data - Your specified x_size was odd (even) when the image x dimension is '
                'even (odd)')
            logger.debug(
                'The method has automatically used x_size+1 to ensure the image is not miscentred by a half-pixel.')
        elif self.shape[1] != new_dimensions[1]:
            logger.debug(
                'image.weighted_data.trim_data - Your specified y_size was odd (even) when the image y dimension is '
                'even (odd)')
            logger.debug(
                'The method has automatically used y_size+1 to ensure the image is not miscentred by a half-pixel.')

        return self.new_with_array(array)

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


class ScaledArray(AbstractArray):
    """
    Class storing the grids for 2D pixel grids (e.g. image, PSF, signal_to_noise_ratio).
    """

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scale=1):
        """
        Parameters
        ----------
        array: ndarray
            An array representing the weighted_data
        pixel_scale: float
            The scale of one pixel in arc seconds
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

    @property
    def central_pixel_coordinates(self):
        """
        Returns
        -------
        central_pixel_coordinates:
            The coordinates of the central pixel in the image. If a dimension of the image are odd then the
            corresponding coordinate will be fra
    @propertyctional.
        """
        return float(self.shape[0] - 1) / 2, float(self.shape[1] - 1) / 2

    def pixels_to_arc_seconds(self, pixels):
        """
        Converts a value from pixels to arc seconds.
        """
        return self.pixel_scale * pixels

    def arc_seconds_to_pixels(self, arc_seconds):
        """
        Converts a value from arc seconds to pixels.
        """
        return arc_seconds / self.pixel_scale

    def pixel_coordinates_to_arc_second_coordinates(self, pixel_coordinates):
        """
        Converts a pixel coordinate pair to an arc seconds coordinate pair. The pixel coordinate origin is at the top
        left corner of the image whilst the arc second coordinate origin is at the centre. This means that the original
        pixel coordinates, (0, 0), will give negative arc second coordinates.

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
        Converts an arc second coordinate pair to a pixel coordinate pair. The pixel coordinate origin is at the top
        left corner of the image whilst the arc second coordinate origin is at the centre. This means that the original
        pixel coordinates, (0, 0), will give negative arc second coordinates.

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
        """
        The shape of the image in arc seconds
        """
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
        """Convert a coordinate on the regular image-pixel grid_coords to a sub-coordinate, using the pixel scale and
        sub-grid_coords sub_grid_size """

        half = self.pixel_scale / 2
        step = self.pixel_scale / (sub_grid_size + 1)

        return arcsec - half + (sub_pixel + 1) * step

    @property
    def grid_coordinates(self):
        """
        Computes the arc second grids of every pixel on the weighted_data-grid_coords.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value and positive y value in arc seconds.
        """

        coordinates_array = np.zeros((self.shape[0], self.shape[1], 2))

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                arc_second_coordinates = self.pixel_coordinates_to_arc_second_coordinates((x, y))
                coordinates_array[x, y, 0] = arc_second_coordinates[0]
                coordinates_array[x, y, 1] = arc_second_coordinates[1]

        return coordinates_array

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

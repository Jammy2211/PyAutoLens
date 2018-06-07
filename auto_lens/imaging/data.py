import numpy as np
import logging
from astropy.io import fits

logging.basicConfig()
logger = logging.getLogger(__name__)


class Array(np.ndarray):
    """
    Class storing the grids for 2D pixel grids (e.g. image, PSF, signal_to_noise_ratio).
    """

    def __new__(cls, array, pixel_scale=1, **kwargs):
        return np.array(array).view(cls)

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scale=1):
        """
        Parameters
        ----------
        array: ndarray
            An array representing the data
        pixel_scale: float
            The scale of one pixel in arc seconds
        """
        # noinspection PyArgumentList
        super(Array, self).__init__()
        self.pixel_scale = pixel_scale

    @property
    def central_pixel_coordinates(self):
        """
        Returns
        -------

        """
        return float(self.shape[0] - 1) / 2, float(self.shape[1] - 1) / 2

    def pixels_to_arc_seconds(self, pixels):
        return self.pixel_scale * pixels

    def arc_seconds_to_pixels(self, arc_seconds):
        return arc_seconds / self.pixel_scale

    def pixel_coordinates_to_arc_second_coordinates(self, pixel_coordinates):
        return tuple(map(lambda coord, centre: self.pixels_to_arc_seconds(coord - centre), pixel_coordinates,
                         self.central_pixel_coordinates))

    def arc_second_coordinates_to_pixel_coordinates(self, arc_second_coordinates):
        return tuple(map(lambda coord, centre: self.arc_seconds_to_pixels(coord) + centre,
                         arc_second_coordinates,
                         self.central_pixel_coordinates))

    @property
    def shape_arc_seconds(self):
        return tuple(map(lambda d: self.pixels_to_arc_seconds(d), self.shape))

    def new_with_array(self, array):
        """
        Parameters
        ----------
        array: ndarray
            An ndarray

        Returns
        -------
        new_array: Array
            A new instance of this class that shares all of this instances attributes with a new ndarray.
        """
        arguments = vars(self)
        arguments.update({"array": array})

        return self.__class__(**arguments)

    def pad(self, new_dimensions):
        """ Pad the data array with zeros around its central pixel.

        NOTE: The centre of the array cannot be shifted. Therefore, even arrays must be padded to even arrays \
        (e.g. 8x8 -> 4x4) and odd to odd (e.g. 5x5 -> 3x3).

        Parameters
        ----------
        new_dimensions : (int, int)
            The (x,y) new pixel dimension of the padded data-array.
        """
        if new_dimensions[0] < self.shape[0]:
            raise ValueError('grids.Grid2d.pad - You have specified a new x_size smaller than the data array')
        elif new_dimensions[1] < self.shape[1]:
            raise ValueError('grids.Grid2d.pad - You have specified a new y_size smaller than the data array')

        x_pad = int((new_dimensions[0] - self.shape[0] + 1) / 2)
        y_pad = int((new_dimensions[1] - self.shape[1] + 1) / 2)

        array = np.pad(self, ((x_pad, y_pad), (x_pad, y_pad)), 'constant')

        return self.new_with_array(array)

    def trim(self, new_dimensions):
        """ Trim the data array to a new size around its central pixel.

        NOTE: The centre of the array cannot be shifted. Therefore, even arrays must be trimmed to even arrays \
        (e.g. 8x8 -> 4x4) and odd to odd (e.g. 5x5 -> 3x3).

        Parameters
        ----------
        new_dimensions : (int, int)
            The (x,y) new pixel dimension of the trimmed data-array.
        """
        if new_dimensions[0] > self.shape[0]:
            raise ValueError('grids.Grid2d.trim_data - You have specified a new x_size bigger than the data array')
        elif new_dimensions[1] > self.shape[1]:
            raise ValueError('grids.Grid2d.trim_data - You have specified a new y_size bigger than the data array')

        x_trim = int((self.shape[0] - new_dimensions[0]) / 2)
        y_trim = int((self.shape[1] - new_dimensions[1]) / 2)

        array = self[x_trim:self.shape[0] - x_trim, y_trim:self.shape[1] - y_trim]

        if self.shape[0] != new_dimensions[0]:
            logger.debug(
                'image.data.trim_data - Your specified x_size was odd (even) when the image x dimension is even (odd)')
            logger.debug(
                'The method has automatically used x_size+1 to ensure the image is not miscentred by a half-pixel.')
        elif self.shape[1] != new_dimensions[1]:
            logger.debug(
                'image.data.trim_data - Your specified y_size was odd (even) when the image y dimension is even (odd)')
            logger.debug(
                'The method has automatically used y_size+1 to ensure the image is not miscentred by a half-pixel.')

        return self.new_with_array(array)

    def sub_pixel_to_coordinate(self, sub_pixel, arcsec, sub_grid_size):
        """Convert a coordinate on the regular image-pixel grid_coords to a sub-coordinate, using the pixel scale and sub-grid_coords \
        size"""

        half = self.pixel_scale / 2
        step = self.pixel_scale / (sub_grid_size + 1)

        return arcsec - half + (sub_pixel + 1) * step

    @property
    def grid_coordinates(self):
        """
        Computes the arc second grids of every pixel on the data-grid_coords.

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
    def from_fits(cls, file_path, hdu, pixel_scale):
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
        hdu_list = fits.open(file_path)  # Open the fits file
        array = np.array(hdu_list[hdu].data)
        return cls(array, pixel_scale)

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
        array: Array
            An array filled with a single value
        """
        array = np.ones(shape) * value
        return cls(array, pixel_scale)

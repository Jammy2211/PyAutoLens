import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)


class DataGrid(np.ndarray):
    """
    Class storing the grids for 2D pixel grids (e.g. image, PSF, signal_to_noise_ratio).
    """

    def __new__(cls, array, pixel_scale=1):
        return np.array(array).view(cls)

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scale):
        """
        Parameters
        ----------
        array: ndarray
            An array representing the data
        pixel_scale: float
            The scale of one pixel in arc seconds
        """
        # noinspection PyArgumentList
        super(DataGrid, self).__init__()
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

        arguments = vars(self)
        arguments.update({"array": array})

        return self.__class__(**arguments)

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

        arguments = vars(self)
        arguments.update({"array": array})

        return self.__class__(**arguments)

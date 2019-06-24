import logging

import numpy as np

from autolens import exc
from autolens.data.array.util import array_util, grid_util, binning_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class ArrayGeometry(object):

    shape_arc_sec = None
    origin = None

    @property
    def shape_arcsec(self):
        return self.shape_arc_sec

    @property
    def arc_second_maxima(self):
        return ((self.shape_arcsec[0] / 2.0) + self.origin[0],
                (self.shape_arcsec[1] / 2.0) + self.origin[1])

    @property
    def arc_second_minima(self):
        return ((-(self.shape_arcsec[0] / 2.0)) + self.origin[0],
                (-(self.shape_arcsec[1] / 2.0)) + self.origin[1])

    @property
    def yticks(self):
        """Compute the yticks labels of this grid, used for plotting the y-axis ticks when visualizing an image-grid"""
        return np.linspace(self.arc_second_minima[0], self.arc_second_maxima[0], 4)

    @property
    def xticks(self):
        """Compute the xticks labels of this grid, used for plotting the x-axis ticks when visualizing an image-grid"""
        return np.linspace(self.arc_second_minima[1], self.arc_second_maxima[1], 4)


# noinspection PyUnresolvedReferences
class RectangularArrayGeometry(ArrayGeometry):

    pixel_scales = None

    @property
    def shape_arcsec(self):
        return (float(self.pixel_scales[0] * self.shape[0]), float(self.pixel_scales[1] * self.shape[1]))

    @property
    def central_pixel_coordinates(self):
        return float(self.shape[0] - 1) / 2, float(self.shape[1] - 1) / 2

    def arc_second_coordinates_to_pixel_coordinates(self, arc_second_coordinates):
        return (int(((-arc_second_coordinates[0] + self.origin[0]) / self.pixel_scales[0]) + self.central_pixel_coordinates[0] + 0.5),
                int(((arc_second_coordinates[1] - self.origin[1]) / self.pixel_scales[1]) + self.central_pixel_coordinates[1] + 0.5))

    def grid_arcsec_to_grid_pixels(self, grid_arcsec):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
        returned as floats such that they include the decimal offset from each pixel's top-left corner.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        highest y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_arcsec: ndarray
            A grid of (y,x) coordinates in arc seconds.
        """
        return grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=grid_arcsec, shape=self.shape,
                                                               pixel_scales=self.pixel_scales, origin=self.origin)

    def grid_arcsec_to_grid_pixel_centres(self, grid_arcsec):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
        returned as integers such that they map directly to the pixel they are contained within.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        higher y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_arcsec: ndarray
            The grid of (y,x) coordinates in arc seconds.
        """
        return grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec,
                                                                      shape=self.shape,
                                                                      pixel_scales=self.pixel_scales,
                                                                      origin=self.origin).astype('int')

    def grid_arcsec_to_grid_pixel_indexes(self, grid_arcsec):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel 1D indexes. Pixel coordinates are \
        returned as integers such that they are the pixel from the top-left of the 2D grid going rights and then \
        downwards.

        For example:

        The pixel at the top-left, whose 2D index is [0,0], corresponds to 1D index 0.
        The fifth pixel on the top row, whose 2D index is [0,5], corresponds to 1D index 4.
        The first pixel on the second row, whose 2D index is [0,1], has 1D index 10 if a row has 10 pixels.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_arcsec: ndarray
            The grid of (y,x) coordinates in arc seconds.
        """
        return grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(grid_arcsec_1d=grid_arcsec,
                                                                      shape=self.shape,
                                                                      pixel_scales=self.pixel_scales,
                                                                      origin=self.origin).astype('int')

    def grid_pixels_to_grid_arcsec(self, grid_pixels):
        """Convert a grid of (y,x) pixel coordinates to a grid of (y,x) arc second values.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        higher y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_pixels : ndarray
            The grid of (y,x) coordinates in pixels.
        """
        return grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=grid_pixels, shape=self.shape,
                                                               pixel_scales=self.pixel_scales, origin=self.origin)

    @property
    def grid_1d(self):
        """ The arc second-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value y value in arc seconds.
        """
        return grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=self.shape,
                                                                            pixel_scales=self.pixel_scales,
                                                                            origin=self.origin)

    @property
    def grid_2d(self):
        """ The arc second-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value y value in arc seconds.
        """
        return grid_util.regular_grid_2d_from_shape_pixel_scales_and_origin(shape=self.shape,
                                                                            pixel_scales=self.pixel_scales,
                                                                            origin=self.origin)


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


    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

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
        if 'centre' in arguments:
            arguments.pop("centre")
        return self.__class__(**arguments)

    @classmethod
    def from_fits(cls, file_path, hdu):
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        """
        return cls(array_util.numpy_array_2d_from_fits(file_path, hdu))


class ScaledArray(Array, RectangularArrayGeometry):

    # noinspection PyUnusedLocal
    def __init__(self, array):
        """ A scaled-array is a 2D array with a pixel-scale, such that all pixels on the array has associated with \
        them arc-second coordinates.
        
        Parameters
        ----------
        array: ndarray
            An array representing image (e.g. an image, noise-map, etc.)
        origin : (float, float)
            The arc-second origin of the scaled array's coordinate system.
        """
        # noinspection PyArgumentList
        super(ScaledArray, self).__init__()

    def map(self, func):
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                func(y, x)


class ScaledSquarePixelArray(ScaledArray):
    """
    Class storing the grid_stacks for 2D pixel grid_stacks (e.g. image, PSF, signal_to_noise_ratio).
    """

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scale, origin=(0.0, 0.0)):
        """ A scaled array with square-pixels.
        
        Parameters
        ----------
        array: ndarray
            An array representing image (e.g. an image, noise-map, etc.)
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The arc-second origin of the scaled array's coordinate system.
        """

        pixel_scale_sanity_checks(pixel_scales=(pixel_scale, pixel_scale))

        # noinspection PyArgumentList
        self.pixel_scale = pixel_scale
        self.origin = origin
        super(ScaledSquarePixelArray, self).__init__(array=array)

    def __array_finalize__(self, obj):
        if hasattr(obj, "pixel_scale"):
            self.pixel_scale = obj.pixel_scale
        if hasattr(obj, 'origin'):
            self.origin = obj.origin

    @property
    def pixel_scales(self):
        return self.pixel_scale, self.pixel_scale

    @classmethod
    def from_fits_with_pixel_scale(cls, file_path, hdu, pixel_scale, origin=(0.0, 0.0)):
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls(array_util.numpy_array_2d_from_fits(file_path, hdu), pixel_scale, origin)

    @classmethod
    def single_value(cls, value, shape, pixel_scale, origin=(0.0, 0.0)):
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
        return cls(array, pixel_scale, origin)

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

    def zoomed_scaled_array_around_mask(self, mask, buffer=1):
        """Extract the 2D region of an array corresponding to the rectangle encompassing all unmasked values.

        This is used to extract and visualize only the region of an image that is used in an analysis.

        Parameters
        ----------
        mask : mask.Mask
            The mask around which the scaled array is extracted.
        buffer : int
            The buffer of pixels around the extraction.
        """
        return self.new_with_array(array=array_util.extracted_array_2d_from_array_2d_and_coordinates(
            array_2d=self,  y0=mask.zoom_region[0]-buffer, y1=mask.zoom_region[1]+buffer,
            x0=mask.zoom_region[2]-buffer, x1=mask.zoom_region[3]+buffer))

    def resized_scaled_array_from_array(self, new_shape, new_centre_pixels=None, new_centre_arcsec=None):
        """resized the array to a new shape and at a new origin.

        Parameters
        -----------
        new_shape : (int, int)
            The new two-dimensional shape of the array.
        """
        if new_centre_pixels is None and new_centre_arcsec is None:
            new_centre = (-1, -1)  # In Numba, the input origin must be the same image type as the origin, thus we cannot
            # pass 'None' and instead use the tuple (-1, -1).
        elif new_centre_pixels is not None and new_centre_arcsec is None:
            new_centre = new_centre_pixels
        elif new_centre_pixels is None and new_centre_arcsec is not None:
            new_centre = self.arc_second_coordinates_to_pixel_coordinates(arc_second_coordinates=new_centre_arcsec)
        else:
            raise exc.DataException('You have supplied two centres (pixels and arc-seconds) to the resize scaled'
                                       'array function')

        return self.new_with_array(array=array_util.resized_array_2d_from_array_2d_and_resized_shape(
            array_2d=self, resized_shape=new_shape, origin=new_centre))

    def binned_up_array_from_array(self, bin_up_factor, method):

        if method is 'mean':
            return ScaledSquarePixelArray(array=binning_util.binned_up_array_2d_using_mean_from_array_2d_and_bin_up_factor(array_2d=self,
                                                                                                                         bin_up_factor=bin_up_factor),
                                          pixel_scale=self.pixel_scale*bin_up_factor)
        elif method is 'quadrature':
            return ScaledSquarePixelArray(array=binning_util.binned_array_2d_using_quadrature_from_array_2d_and_bin_up_factor(array_2d=self,
                                                                                                                            bin_up_factor=bin_up_factor),
                                          pixel_scale=self.pixel_scale*bin_up_factor)
        elif method is 'sum':
            return ScaledSquarePixelArray(array=binning_util.binned_array_2d_using_sum_from_array_2d_and_bin_up_factor(array_2d=self,
                                                                                                                     bin_up_factor=bin_up_factor),
                                          pixel_scale=self.pixel_scale*bin_up_factor)
        else:
            raise exc.DataException('The method used in binned_up_array_from_array is not a valid method '
                                       '[mean | quadrature | sum]')


class ScaledRectangularPixelArray(ScaledArray):
    """
    Class storing the grid_stacks for 2D pixel grid_stacks (e.g. image, PSF, signal_to_noise_ratio).
    """

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scales, origin=(0.0, 0.0)):
        """
        Parameters
        ----------
        array: ndarray
            An array representing image (e.g. an image, noise-map, etc.)
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The arc-second origin of the scaled array's coordinate system.
        """

        pixel_scale_sanity_checks(pixel_scales=pixel_scales)

        self.pixel_scales = pixel_scales
        self.origin = origin
        # noinspection PyArgumentList
        super(ScaledRectangularPixelArray, self).__init__(array=array)

    def __array_finalize__(self, obj):
        if hasattr(obj, "pixel_scales"):
            self.pixel_scales = obj.pixel_scales
        if hasattr(obj, 'origin'):
            self.origin = obj.origin

    @classmethod
    def single_value(cls, value, shape, pixel_scales, origin=(0.0, 0.0)):
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
        return cls(array, pixel_scales, origin=origin)

    @classmethod
    def from_fits_with_pixel_scale(cls, file_path, hdu, pixel_scales, origin=(0.0, 0.0)):
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the hyper.
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls(array_util.numpy_array_2d_from_fits(file_path, hdu), pixel_scales, origin)

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


def pixel_scale_sanity_checks(pixel_scales):

    if pixel_scales[0] <= 0.0 or pixel_scales[1] <= 0:
        raise exc.ScaledArrayException('A pixel scale supplied to ScaledRectangularPixelArray (and therefore the Image) '
                                   'is zero or negative')
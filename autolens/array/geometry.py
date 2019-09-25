import numpy as np

from autolens import exc
from autolens.array.util import grid_util


class Geometry(object):

    def __init__(self, shape, pixel_scales, sub_size, origin=(0.0, 0.0)):

        if pixel_scales[0] <= 0.0 or pixel_scales[1] <= 0:
            raise exc.GeometryException(
                "A pixel scale supplied to a Mask (and therefore the Image) "
                "is zero or negative"
            )

        self.shape = shape
        self.pixel_scales = pixel_scales
        self.sub_size = sub_size
        self.sub_length = int(self.sub_size ** 2.0)
        self.sub_fraction = 1.0 / self.sub_length
        self.origin = origin

    @property
    def pixel_scale(self):
        if self.pixel_scales[0] == self.pixel_scales[1]:
            return self.pixel_scales[0]
        else:
            raise exc.ScaledArrayException("Cannot return a pixel_scale for a a grid where each dimension has a "
                                           "different pixel scale (e.g. pixel_scales[0] != pixel_scales[1]")

    @property
    def shape_arcsec(self):
        return (
            float(self.pixel_scales[0] * self.shape[0]),
            float(self.pixel_scales[1] * self.shape[1]),
        )

    @property
    def central_pixel_coordinates(self):
        return float(self.shape[0] - 1) / 2, float(self.shape[1] - 1) / 2

    @property
    def arc_second_maxima(self):
        return (
            (self.shape_arcsec[0] / 2.0) + self.origin[0],
            (self.shape_arcsec[1] / 2.0) + self.origin[1],
        )

    @property
    def arc_second_minima(self):
        return (
            (-(self.shape_arcsec[0] / 2.0)) + self.origin[0],
            (-(self.shape_arcsec[1] / 2.0)) + self.origin[1],
        )

    @property
    def yticks(self):
        """Compute the yticks labels of this grid, used for plotting the y-axis ticks when visualizing an image-grid"""
        return np.linspace(self.arc_second_minima[0], self.arc_second_maxima[0], 4)

    @property
    def xticks(self):
        """Compute the xticks labels of this grid, used for plotting the x-axis ticks when visualizing an image-grid"""
        return np.linspace(self.arc_second_minima[1], self.arc_second_maxima[1], 4)

    def arc_second_coordinates_to_pixel_coordinates(self, arc_second_coordinates):
        return (
            int(
                ((-arc_second_coordinates[0] + self.origin[0]) / self.pixel_scales[0])
                + self.central_pixel_coordinates[0]
                + 0.5
            ),
            int(
                ((arc_second_coordinates[1] - self.origin[1]) / self.pixel_scales[1])
                + self.central_pixel_coordinates[1]
                + 0.5
            ),
        )

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
        return grid_util.grid_arcsec_1d_to_grid_pixels_1d(
            grid_arcsec_1d=grid_arcsec,
            shape=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

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
        return grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(
            grid_arcsec_1d=grid_arcsec,
            shape=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        ).astype("int")

    def grid_arcsec_1d_to_grid_pixel_indexes_1d(self, grid_arcsec):
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
        return grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(
            grid_arcsec_1d=grid_arcsec,
            shape=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        ).astype("int")

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
        return grid_util.grid_pixels_1d_to_grid_arcsec_1d(
            grid_pixels_1d=grid_pixels,
            shape=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def grid_1d(self):
        """ The arc second-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value y value in arc seconds.
        """
        return grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
            shape=self.shape,
            pixel_scales=self.pixel_scales,
            sub_size=1,
            origin=self.origin,
        )

    @property
    def grid_2d(self):
        """ The arc second-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value y value in arc seconds.
        """
        return grid_util.grid_2d_from_shape_pixel_scales_sub_size_and_origin(
            shape=self.shape,
            pixel_scales=self.pixel_scales,
            sub_size=1,
            origin=self.origin,
        )

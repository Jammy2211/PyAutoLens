import logging

import numpy as np

from autolens import exc
from autolens.array import scaled_array, grids
from autolens.array.util import grid_util, array_util, mask_util, binning_util
from autolens.array.mapping_util import array_mapping_util, grid_mapping_util, mask_mapping_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask(np.ndarray):

    # noinspection PyUnusedLocal
    def __new__(
            cls, array_2d, pixel_scales, sub_size, origin=(0.0, 0.0), *args, **kwargs
    ):
        """ A mask, which is applied to a 2D array of hyper_galaxies to extract a set of unmasked image pixels (i.e. mask entry \
        is *False* or 0) which are then fitted in an analysis.

        The mask retains the pixel scale of the array and has a centre and origin.

        Parameters
        ----------
        array_2d: ndarray
            An array of bools representing the mask.
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The (y,x) arc-second origin of the mask's coordinate system.
        centre : (float, float)
            The (y,x) arc-second centre of the mask provided it is a standard geometric shape (e.g. a circle).
        """
        # noinspection PyArgumentList

        obj = array_2d.view(cls)

        if pixel_scales[0] <= 0.0 or pixel_scales[1] <= 0:
            raise exc.GeometryException(
                "A pixel scale supplied to a Mask (and therefore the Image) "
                "is zero or negative"
            )

        obj.pixel_scales = pixel_scales
        obj.sub_size = sub_size
        obj.sub_length = int(obj.sub_size ** 2.0)
        obj.sub_fraction = 1.0 / obj.sub_length
        obj.origin = origin

        return obj

    def __array_finalize__(self, obj):
        if hasattr(obj, 'pixel_scales'):
            self.pixel_scales = obj.pixel_scales
        if hasattr(obj, 'sub_size'):
            self.sub_size = obj.sub_size
            self.sub_length = int(obj.sub_size ** 2.0)
            self.sub_fraction = 1.0 / obj.sub_length
        if hasattr(obj, 'origin'):
            self.origin = obj.origin

    @classmethod
    def unmasked_from_shape_pixel_scales_and_sub_size(
        cls, shape, pixel_scales, sub_size, origin=(0.0, 0.0), invert=False
    ):
        """Setup a mask where all pixels are unmasked.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        mask = np.full(tuple(map(lambda d: int(d), shape)), False)
        if invert:
            mask = np.invert(mask)
        return Mask(
            array_2d=mask, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

    @classmethod
    def circular(
        cls,
        shape,
        pixel_scales,
        radius_arcsec,
        sub_size,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
        """Setup a mask where unmasked pixels are within a circle of an input arc second radius and centre.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        radius_arcsec : float
            The radius (in arc seconds) of the circle within which pixels unmasked.
        centre: (float, float)
            The centre of the circle used to mask pixels.
        """
        mask = mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=shape,
            pixel_scales=pixel_scales,
            radius_arcsec=radius_arcsec,
            centre=centre,
        )
        if invert:
            mask = np.invert(mask)
        return Mask(
            array_2d=mask.astype("bool"),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def circular_annular(
        cls,
        shape,
        pixel_scales,
        sub_size,
        inner_radius_arcsec,
        outer_radius_arcsec,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
        """Setup a mask where unmasked pixels are within an annulus of input inner and outer arc second radii and \
         centre.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        inner_radius_arcsec : float
            The radius (in arc seconds) of the inner circle outside of which pixels are unmasked.
        outer_radius_arcsec : float
            The radius (in arc seconds) of the outer circle within which pixels are unmasked.
        centre: (float, float)
            The centre of the annulus used to mask pixels.
        """

        mask = mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
            shape=shape,
            pixel_scales=pixel_scales,
            inner_radius_arcsec=inner_radius_arcsec,
            outer_radius_arcsec=outer_radius_arcsec,
            centre=centre,
        )

        if invert:
            mask = np.invert(mask)

        return Mask(
            array_2d=mask.astype("bool"),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def circular_anti_annular(
        cls,
        shape,
        pixel_scales,
        sub_size,
        inner_radius_arcsec,
        outer_radius_arcsec,
        outer_radius_2_arcsec,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
        """Setup a mask where unmasked pixels are outside an annulus of input inner and outer arc second radii, but \
        within a second outer radius, and at a given centre.

        This mask there has two distinct unmasked regions (an inner circle and outer annulus), with an inner annulus \
        of masked pixels.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        inner_radius_arcsec : float
            The radius (in arc seconds) of the inner circle inside of which pixels are unmasked.
        outer_radius_arcsec : float
            The radius (in arc seconds) of the outer circle within which pixels are masked and outside of which they \
            are unmasked.
        outer_radius_2_arcsec : float
            The radius (in arc seconds) of the second outer circle within which pixels are unmasked and outside of \
            which they masked.
        centre: (float, float)
            The centre of the anti-annulus used to mask pixels.
        """

        mask = mask_util.mask_circular_anti_annular_from_shape_pixel_scales_and_radii(
            shape=shape,
            pixel_scales=pixel_scales,
            inner_radius_arcsec=inner_radius_arcsec,
            outer_radius_arcsec=outer_radius_arcsec,
            outer_radius_2_arcsec=outer_radius_2_arcsec,
            centre=centre,
        )

        if invert:
            mask = np.invert(mask)

        return Mask(
            array_2d=mask.astype("bool"),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def elliptical(
        cls,
        shape,
        pixel_scales,
        major_axis_radius_arcsec,
        axis_ratio,
        phi,
        sub_size,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
        """ Setup a mask where unmasked pixels are within an ellipse of an input arc second major-axis and centre.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        major_axis_radius_arcsec : float
            The major-axis (in arc seconds) of the ellipse within which pixels are unmasked.
        axis_ratio : float
            The axis-ratio of the ellipse within which pixels are unmasked.
        phi : float
            The rotation angle of the ellipse within which pixels are unmasked, (counter-clockwise from the positive \
             x-axis).
        centre: (float, float)
            The centre of the ellipse used to mask pixels.
        """

        mask = mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=shape,
            pixel_scales=pixel_scales,
            major_axis_radius_arcsec=major_axis_radius_arcsec,
            axis_ratio=axis_ratio,
            phi=phi,
            centre=centre,
        )

        if invert:
            mask = np.invert(mask)

        return Mask(
            array_2d=mask.astype("bool"),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def elliptical_annular(
        cls,
        shape,
        pixel_scales,
        sub_size,
        inner_major_axis_radius_arcsec,
        inner_axis_ratio,
        inner_phi,
        outer_major_axis_radius_arcsec,
        outer_axis_ratio,
        outer_phi,
        origin=(0.0, 0.0),
        centre=(0.0, 0.0),
        invert=False,
    ):
        """Setup a mask where unmasked pixels are within an elliptical annulus of input inner and outer arc second \
        major-axis and centre.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        inner_major_axis_radius_arcsec : float
            The major-axis (in arc seconds) of the inner ellipse within which pixels are masked.
        inner_axis_ratio : float
            The axis-ratio of the inner ellipse within which pixels are masked.
        inner_phi : float
            The rotation angle of the inner ellipse within which pixels are masked, (counter-clockwise from the \
            positive x-axis).
        outer_major_axis_radius_arcsec : float
            The major-axis (in arc seconds) of the outer ellipse within which pixels are unmasked.
        outer_axis_ratio : float
            The axis-ratio of the outer ellipse within which pixels are unmasked.
        outer_phi : float
            The rotation angle of the outer ellipse within which pixels are unmasked, (counter-clockwise from the \
            positive x-axis).
        centre: (float, float)
            The centre of the elliptical annuli used to mask pixels.
        """

        mask = mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=shape,
            pixel_scales=pixel_scales,
            inner_major_axis_radius_arcsec=inner_major_axis_radius_arcsec,
            inner_axis_ratio=inner_axis_ratio,
            inner_phi=inner_phi,
            outer_major_axis_radius_arcsec=outer_major_axis_radius_arcsec,
            outer_axis_ratio=outer_axis_ratio,
            outer_phi=outer_phi,
            centre=centre,
        )

        if invert:
            mask = np.invert(mask)

        return Mask(
            array_2d=mask.astype("bool"),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def mask_from_fits(cls, file_path, hdu, pixel_scales, sub_size, origin=(0.0, 0.0)):
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scales : (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        return Mask(
            array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @property
    def pixel_scale(self):
        if self.pixel_scales[0] == self.pixel_scales[1]:
            return self.pixel_scales[0]
        else:
            raise exc.ScaledException(
                "Cannot return a pixel_scale for a a grid where each dimension has a "
                "different pixel scale (e.g. pixel_scales[0] != pixel_scales[1]"
            )

    @property
    def sub_mask(self):

        sub_shape = (
            self.shape[0] * self.sub_size,
            self.shape[1] * self.sub_size,
        )

        return mask_util.mask_from_shape_and_mask_2d_index_for_mask_1d_index(
            shape=sub_shape,
            mask_2d_index_for_mask_1d_index=self._sub_mask_2d_index_for_sub_mask_1d_index,
        ).astype("bool")

    @property
    def unmasked_grid(self):
        """ The arc second-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value y value in arc seconds.
        """
        grid_1d = grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
            shape=self.shape,
            pixel_scales=self.pixel_scales,
            sub_size=1,
            origin=self.origin,
        )
        return self.grid_from_grid_1d(grid_1d=grid_1d)

    @property
    def masked_grid(self):
        grid_1d = grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=self,
            pixel_scales=self.pixel_scales,
            sub_size=1,
            origin=self.origin,
        )
        return self.grid_from_grid_1d(grid_1d=grid_1d)

    @property
    def masked_sub_grid(self):
        sub_grid_1d = grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=self,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )
        return self.grid_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

    def resized_mask_from_new_shape(
        self, new_shape, new_centre_pixels=None, new_centre_arcsec=None
    ):
        """resized the array to a new shape and at a new origin.

        Parameters
        -----------
        new_shape : (int, int)
            The new two-dimensional shape of the array.
        """
        if new_centre_pixels is None and new_centre_arcsec is None:

            new_centre = (
                -1,
                -1,
            )  # In Numba, the input origin must be the same image type as the origin, thus we cannot
            # pass 'None' and instead use the tuple (-1, -1).

        elif new_centre_pixels is not None and new_centre_arcsec is None:

            new_centre = new_centre_pixels

        elif new_centre_pixels is None and new_centre_arcsec is not None:

            new_centre = self.pixel_coordinates_from_arcsec_coordinates(
                arcsec_coordinates=new_centre_arcsec
            )

        else:

            raise exc.DataException(
                "You have supplied two centres (pixels and arc-seconds) to the resize hyper"
                "array function"
            )

        resized_mask_2d = array_util.resized_array_2d_from_array_2d_and_resized_shape(
            array_2d=self, resized_shape=new_shape, origin=new_centre
        ).astype("bool")
        return Mask(
            array_2d=resized_mask_2d,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=new_centre,
        )

    def new_mask_with_new_sub_size(self, sub_size):
        return Mask(
            array_2d=self,
            pixel_scales=self.pixel_scales,
            sub_size=sub_size,
            origin=self.origin,
        )

    def binned_up_mask_from_mask(self, bin_up_factor):

        binned_up_mask = binning_util.binned_up_mask_from_mask_2d_and_bin_up_factor(
            mask_2d=self, bin_up_factor=bin_up_factor
        )

        return Mask(
            array_2d=binned_up_mask,
            pixel_scales=(
                self.pixel_scales[0] * bin_up_factor,
                self.pixel_scales[1] * bin_up_factor,
            ),
            sub_size=self.sub_size,
            origin=self.origin,
        )

    def binned_up_mask_sub_size_1_from_mask(self, bin_up_factor):

        binned_up_mask = binning_util.binned_up_mask_from_mask_2d_and_bin_up_factor(
            mask_2d=self, bin_up_factor=bin_up_factor
        )

        return Mask(
            array_2d=binned_up_mask,
            pixel_scales=(
                self.pixel_scales[0] * bin_up_factor,
                self.pixel_scales[1] * bin_up_factor,
            ),
            sub_size=1,
            origin=self.origin,
        )

    @property
    @array_util.Memoizer()
    def mask_centre(self):
        return grid_util.grid_centre_from_grid_1d(grid_1d=self.masked_grid)

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))

    @array_util.Memoizer()
    def blurring_mask_from_kernel_shape(self, kernel_shape):
        """Compute a blurring mask, which represents all masked pixels whose light will be blurred into unmasked \
        pixels via PSF convolution (see grid.Grid.blurring_grid_from_mask_and_psf_shape).

        Parameters
        ----------
        kernel_shape : (int, int)
           The shape of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = mask_util.blurring_mask_from_mask_and_kernel_shape(
            self, kernel_shape
        )

        return Mask(
            array_2d=blurring_mask, pixel_scales=self.pixel_scales, sub_size=1
        )

    @property
    def edge_mask(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        mask = np.full(fill_value=True, shape=self.shape)
        mask[self._edge_2d_indexes[:,0], self._edge_2d_indexes[:,1]] = False
        return Mask(array_2d=mask, pixel_scales=self.pixel_scales, sub_size=1, origin=self.origin)

    @property
    def edge_grid(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        edge_grid_1d = self.masked_grid[self._edge_1d_indexes]
        return self.edge_mask.grid_from_grid_1d(grid_1d=edge_grid_1d)

    @property
    def border_mask(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        mask = np.full(fill_value=True, shape=self.shape)
        mask[self._border_2d_indexes[:,0], self._border_2d_indexes[:,1]] = False
        return Mask(array_2d=mask, pixel_scales=self.pixel_scales, sub_size=1, origin=self.origin)

    @property
    def border_grid(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        border_grid_1d = self.masked_grid[self._border_1d_indexes]
        return self.border_mask.grid_from_grid_1d(grid_1d=border_grid_1d)

    @property
    def sub_border_grid_1d(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return self.masked_sub_grid[self._sub_border_1d_indexes]

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

    def pixel_coordinates_from_arcsec_coordinates(self, arcsec_coordinates):
        return (
            int(
                ((-arcsec_coordinates[0] + self.origin[0]) / self.pixel_scales[0])
                + self.central_pixel_coordinates[0]
                + 0.5
            ),
            int(
                ((arcsec_coordinates[1] - self.origin[1]) / self.pixel_scales[1])
                + self.central_pixel_coordinates[1]
                + 0.5
            ),
        )

    def grid_pixels_from_grid_arcsec(self, grid_arcsec_1d):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
        returned as floats such that they include the decimal offset from each pixel's top-left corner.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        highest y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_arcsec_1d: ndarray
            A grid of (y,x) coordinates in arc seconds.
        """
        grid_pixels_1d = grid_util.grid_pixels_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec_1d,
            shape=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )
        return grid_arcsec_1d.mask.grid_from_grid_1d(grid_1d=grid_pixels_1d)

    def grid_pixel_centres_from_grid_arcsec_1d(self, grid_arcsec_1d):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
        returned as integers such that they map directly to the pixel they are contained within.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        higher y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_arcsec_1d: ndarray
            The grid of (y,x) coordinates in arc seconds.
        """
        grid_pixel_centres_1d = grid_util.grid_pixel_centres_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec_1d,
            shape=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        ).astype("int")
        return grid_arcsec_1d.mask.grid_from_grid_1d(grid_1d=grid_pixel_centres_1d)

    def grid_pixel_indexes_from_grid_arcsec_1d(self, grid_arcsec_1d):
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
        grid_arcsec_1d: ndarray
            The grid of (y,x) coordinates in arc seconds.
        """
        grid_pixel_indexes_1d = grid_util.grid_pixel_indexes_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec_1d,
            shape=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        ).astype("int")
        return grid_arcsec_1d.mask.grid_from_grid_1d(grid_1d=grid_pixel_indexes_1d)

    def grid_arcsec_from_grid_pixels_1d(self, grid_pixels_1d):
        """Convert a grid of (y,x) pixel coordinates to a grid of (y,x) arc second values.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        higher y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_pixels_1d : ndarray
            The grid of (y,x) coordinates in pixels.
        """
        grid_arcsec_1d = grid_util.grid_arcsec_1d_from_grid_pixels_1d_shape_and_pixel_scales(
            grid_pixels_1d=grid_pixels_1d,
            shape=self.shape,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )
        return grid_pixels_1d.mask.grid_from_grid_1d(grid_1d=grid_arcsec_1d)

    def grid_arcsec_from_grid_pixels_1d_for_marching_squares(self, grid_pixels_1d, shape):

        grid_arcsec_1d = grid_util.grid_arcsec_1d_from_grid_pixels_1d_shape_and_pixel_scales(
            grid_pixels_1d=grid_pixels_1d,
            shape=shape,
            pixel_scales=(
                self.pixel_scales[0] / self.sub_size,
                self.pixel_scales[1] / self.sub_size,
            ),
            origin=self.origin,
        )

        grid_arcsec_1d[:, 0] -= self.pixel_scales[0] / (2.0 * self.sub_size)
        grid_arcsec_1d[:, 1] += self.pixel_scales[1] / (2.0 * self.sub_size)

        return grid_arcsec_1d

    def scaled_array_from_array_1d(self, array_1d):
        """ Map a 1D array the same dimension as the grid to its original 2D array.

        Values which were masked in the mapping_util to the 1D array are returned as zeros.

        Parameters
        -----------
        array_1d : ndarray
            The 1D array which is mapped to its masked 2D array.
        """
        mask = self.new_mask_with_new_sub_size(sub_size=1)
        return scaled_array.Scaled(sub_array_1d=array_1d, mask=mask)

    def scaled_array_from_array_2d(self, array_2d):
        """For a 2D array (e.g. an image, noise_map, etc.) map it to a masked 1D array of valuees using this mask.

        Parameters
        ----------
        array_2d : ndarray | None | float
            The 2D array to be mapped to a masked 1D array.
        """
        array_1d = array_mapping_util.sub_array_1d_for_sub_array_2d_mask_and_sub_size(
            mask=self, sub_array_2d=array_2d, sub_size=1
        )
        return self.scaled_array_from_array_1d(array_1d=array_1d)

    def scaled_array_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        return scaled_array.Scaled(sub_array_1d=sub_array_1d, mask=self)

    def scaled_array_from_sub_array_2d(self, sub_array_2d):
        """ Map a 2D sub-array to its masked 1D sub-array.

        Values which are masked in the mapping_util to the 1D array are returned as zeros.

        Parameters
        -----------
        su_array_2d : ndarray
            The 2D sub-array which is mapped to its masked 1D sub-array.
        """
        sub_array_1d = array_mapping_util.sub_array_1d_for_sub_array_2d_mask_and_sub_size(
            sub_array_2d=sub_array_2d, mask=self, sub_size=self.sub_size
        )
        return self.scaled_array_from_sub_array_1d(sub_array_1d=sub_array_1d)

    def scaled_array_binned_from_sub_array_1d(self, sub_array_1d):
        """For an input 1D sub-array, map its values to a 1D array of values by summing each set \of sub-pixel \
        values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_array_1d : ndarray
            A 1D sub-array of values (e.g. image, convergence, potential) which is mapped to
            a 1d array.
        """
        binned_array_1d = np.multiply(
            self.sub_fraction,
            sub_array_1d.reshape(-1, self.sub_length).sum(axis=1),
        )

        mask = self.new_mask_with_new_sub_size(sub_size=1)
        return scaled_array.Scaled(sub_array_1d=binned_array_1d, mask=mask)

    def sub_array_2d_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        return array_mapping_util.sub_array_2d_for_sub_array_1d_mask_and_sub_size(
            sub_array_1d=sub_array_1d, mask=self, sub_size=self.sub_size
        )

    def array_2d_binned_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid to its original masked 2D sub-array and return it as
        a hyper array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub-array of which is mapped to a 2D hyper sub-array the dimensions.
        """
        binned_array_1d = np.multiply(
            self.sub_fraction,
            sub_array_1d.reshape(-1, self.sub_length).sum(axis=1),
        )
        return array_mapping_util.sub_array_2d_for_sub_array_1d_mask_and_sub_size(
            sub_array_1d=binned_array_1d, mask=self, sub_size=1
        )

    def grid_from_grid_1d(self, grid_1d):
        """ Map a 1D grid the same dimension as the grid to its original 2D grid.

        Values which were masked in the mapping_util to the 1D grid are returned as zeros.

        Parameters
        -----------
        grid_1d : ndgrid
            The 1D grid which is mapped to its masked 2D grid.
        """
        mask = self.new_mask_with_new_sub_size(sub_size=1)
        return grids.Grid(sub_grid_1d=grid_1d, mask=mask)

    def grid_from_grid_2d(self, grid_2d):
        """For a 2D grid (e.g. an image, noise_map, etc.) map it to a masked 1D grid of valuees using this mask.

        Parameters
        ----------
        grid_2d : ndgrid | None | float
            The 2D grid to be mapped to a masked 1D grid.
        """
        grid_1d = grid_mapping_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            mask=self, sub_grid_2d=grid_2d, sub_size=1
        )
        return self.grid_from_grid_1d(grid_1d=grid_1d)

    def grid_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub_grid which is mapped to its masked 2D sub-grid.
        """
        return grids.Grid(sub_grid_1d=sub_grid_1d, mask=self)

    def grid_from_sub_grid_2d(self, sub_grid_2d):
        """ Map a 2D sub-grid to its masked 1D sub-grid.

        Values which are masked in the mapping_util to the 1D grid are returned as zeros.

        Parameters
        -----------
        su_grid_2d : ndgrid
            The 2D sub-grid which is mapped to its masked 1D sub-grid.
        """
        sub_grid_1d = grid_mapping_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=sub_grid_2d, mask=self, sub_size=self.sub_size
        )
        return self.grid_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

    def grid_binned_from_sub_grid_1d(self, sub_grid_1d):
        """For an input 1D sub-grid, map its values to a 1D grid of values by summing each set \of sub-pixel \
        values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            A 1D sub-grid of values (e.g. image, convergence, potential) which is mapped to
            a 1d grid.
        """

        grid_1d_y = self.scaled_array_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 0]
        )

        grid_1d_x = self.scaled_array_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 1]
        )

        mask = self.new_mask_with_new_sub_size(sub_size=1)
        return grids.Grid(
            sub_grid_1d=np.stack((grid_1d_y, grid_1d_x), axis=-1), mask=mask
        )

    def sub_grid_2d_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub_grid which is mapped to its masked 2D sub-grid.
        """
        return grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=sub_grid_1d, mask=self, sub_size=self.sub_size
        )

    def grid_2d_binned_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid to its original masked 2D sub-grid and return it as
        a hyper grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub-grid of which is mapped to a 2D hyper sub-grid the dimensions.
        """

        grid_1d_y = self.scaled_array_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 0]
        )

        grid_1d_x = self.scaled_array_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 1]
        )

        binned_grid_1d = np.stack((grid_1d_y, grid_1d_x), axis=-1)

        return grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=binned_grid_1d, mask=self, sub_size=1
        )

    def trimmed_array_2d_from_padded_array_1d_and_image_shape(
        self, padded_array_1d, image_shape
    ):
        """ Map a padded 1D array of values to its original 2D array, trimming all edge values.

        Parameters
        -----------
        padded_array_1d : ndarray
            A 1D array of values which were computed using a padded grid
        """

        padded_array_2d = self.scaled_array_from_array_1d(
            array_1d=padded_array_1d
        ).in_2d
        pad_size_0 = self.shape[0] - image_shape[0]
        pad_size_1 = self.shape[1] - image_shape[1]
        return padded_array_2d[
            pad_size_0 // 2 : self.shape[0] - pad_size_0 // 2,
            pad_size_1 // 2 : self.shape[1] - pad_size_1 // 2,
        ]

    def convolve_padded_array_1d_with_psf(self, padded_array_1d, psf):
        """Convolve a 1d padded array of values (e.g. image before PSF blurring) with a PSF, and then trim \
        the convolved array to its original 2D shape.

        Parameters
        -----------
        padded_array_1d: ndarray
            A 1D array of values which were computed using the a padded grid.
        psf : ndarray
            An array describing the PSF kernel of the image.
        """

        padded_array_2d = array_mapping_util.sub_array_2d_for_sub_array_1d_mask_and_sub_size(
            sub_array_1d=padded_array_1d,
            mask=np.full(fill_value=False, shape=self.shape),
            sub_size=1,
        )

        # noinspection PyUnresolvedReferences
        blurred_padded_array_2d = psf.convolved_array_2d_from_array_2d(array_2d=padded_array_2d)

        return array_mapping_util.sub_array_1d_for_sub_array_2d_mask_and_sub_size(
            sub_array_2d=blurred_padded_array_2d,
            mask=np.full(self.shape, False),
            sub_size=1,
        )

    def unmasked_blurred_array_2d_from_padded_array_1d_psf_and_image_shape(
        self, padded_array_1d, psf, image_shape
    ):
        """For a padded grid and psf, compute an unmasked blurred image from an unmasked unblurred image.

        This relies on using the lens data's padded-grid, which is a grid of (y,x) coordinates which extends over the \
        entire image as opposed to just the masked region.

        Parameters
        ----------
        psf : abstract_data.PSF
            The PSF of the image used for convolution.
        unmasked_image_1d : ndarray
            The 1D unmasked image which is blurred.
        """

        padded_array_2d = self.scaled_array_from_array_1d(array_1d=padded_array_1d)

        blurred_image_2d = psf.convolved_array_2d_from_array_2d(array_2d=padded_array_2d.in_2d)

        blurred_image_1d = self.scaled_array_from_array_2d(array_2d=blurred_image_2d)

        return self.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=blurred_image_1d, image_shape=image_shape
        )

    @property
    def _edge_1d_indexes(self):
        """The indicies of the mask's edge pixels, where an edge pixel is any unmasked pixel on its edge \
        (next to at least one pixel with a *True* value).
        """
        return mask_util.edge_1d_indexes_from_mask(mask=self).astype("int")

    @property
    def _edge_2d_indexes(self):
        """The indicies of the mask's edge pixels, where an edge pixel is any unmasked pixel on its edge \
        (next to at least one pixel with a *True* value).
        """
        return self._mask_2d_index_for_mask_1d_index[self._edge_1d_indexes].astype('int')

    @property
    def _border_1d_indexes(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return mask_util.border_1d_indexes_from_mask(mask=self).astype("int")

    @property
    def _border_2d_indexes(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return self._mask_2d_index_for_mask_1d_index[self._border_1d_indexes].astype('int')

    @property
    def _sub_border_1d_indexes(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=self, sub_size=self.sub_size
        ).astype("int")

    @property
    def _mask_2d_index_for_mask_1d_index(self):
        """A 1D array of mappings between every unmasked pixel and its 2D pixel coordinates."""
        return mask_mapping_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=self, sub_size=1
        ).astype(
            "int"
        )

    @property
    def _sub_mask_2d_index_for_sub_mask_1d_index(self):
        """A 1D array of mappings between every unmasked sub pixel and its 2D sub-pixel coordinates."""
        return mask_mapping_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=self, sub_size=self.sub_size
        ).astype(
            "int"
        )

    @property
    @array_util.Memoizer()
    def _mask_1d_index_for_sub_mask_1d_index(self):
        """The mapping_util between every sub-pixel and its host pixel.

        For example:

        - sub_to_pixel[8] = 2 -  The ninth sub-pixel is within the 3rd pixel.
        - sub_to_pixel[20] = 4 -  The twenty first sub-pixel is within the 5th pixel.
        """
        return mask_mapping_util.mask_1d_index_for_sub_mask_1d_index_from_mask(
            mask=self, sub_size=self.sub_size
        ).astype("int")

    @property
    def _zoom_centre(self):

        extraction_grid_1d = self.grid_pixels_from_grid_arcsec(
            grid_arcsec_1d=self.masked_grid.in_1d
        )
        y_pixels_max = np.max(extraction_grid_1d[:, 0])
        y_pixels_min = np.min(extraction_grid_1d[:, 0])
        x_pixels_max = np.max(extraction_grid_1d[:, 1])
        x_pixels_min = np.min(extraction_grid_1d[:, 1])
        return (
            ((y_pixels_max + y_pixels_min - 1.0) / 2.0),
            ((x_pixels_max + x_pixels_min - 1.0) / 2.0),
        )

    @property
    def _zoom_offset_pixels(self):
        return (
            self._zoom_centre[0] - self.central_pixel_coordinates[0],
            self._zoom_centre[1] - self.central_pixel_coordinates[1],
        )

    @property
    def _zoom_offset_arcsec(self):
        return (
            -self.pixel_scales * self._zoom_offset_pixels[0],
            self.pixel_scales * self._zoom_offset_pixels[1],
        )

    @property
    def _zoom_region(self):
        """The zoomed rectangular region corresponding to the square encompassing all unmasked values. This zoomed
        extraction region is a squuare, even if the mask is rectangular.

        This is used to zoom in on the region of an image that is used in an analysis for visualization."""

        # Have to convert mask to bool for invert function to work.
        where = np.array(np.where(np.invert(self.astype("bool"))))
        y0, x0 = np.amin(where, axis=1)
        y1, x1 = np.amax(where, axis=1)

        ylength = y1 - y0
        xlength = x1 - x0

        if ylength > xlength:
            length_difference = ylength - xlength
            x1 += int(length_difference / 2.0)
            x0 -= int(length_difference / 2.0)
        elif xlength > ylength:
            length_difference = xlength - ylength
            y1 += int(length_difference / 2.0)
            y0 -= int(length_difference / 2.0)

        return [y0, y1 + 1, x0, x1 + 1]


def load_mask_from_fits(mask_path, pixel_scale, sub_size=1, mask_hdu=0):
    return Mask.mask_from_fits(
        file_path=mask_path,
        hdu=mask_hdu,
        pixel_scales=(pixel_scale, pixel_scale),
        sub_size=sub_size,
    )


def output_mask_to_fits(mask, mask_path, overwrite=False):
    array_util.numpy_array_2d_to_fits(
        array_2d=mask, file_path=mask_path, overwrite=overwrite
    )

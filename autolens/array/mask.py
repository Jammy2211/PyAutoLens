import logging

import numpy as np

from autolens import exc
from autolens.array.util import grid_util, array_util, mask_util, binning_util
from autolens.array import mapping
from autolens.array import scaled_array

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask(scaled_array.ScaledSquarePixelArray):
    """
    A mask represented by an ndarray where True is masked.
    """

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scale, sub_size, origin=(0.0, 0.0)):
        """ A mask, which is applied to a 2D array of hyper_galaxies to extract a set of unmasked image pixels (i.e. mask entry \
        is *False* or 0) which are then fitted in an analysis.
        
        The mask retains the pixel scale of the array and has a centre and origin.
        
        Parameters
        ----------
        array: ndarray
            An array of bools representing the mask.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The (y,x) arc-second origin of the mask's coordinate system.
        centre : (float, float)
            The (y,x) arc-second centre of the mask provided it is a standard geometric shape (e.g. a circle).
        """
        # noinspection PyArgumentList
        super(Mask, self).__init__(array=array, pixel_scale=pixel_scale, origin=origin)
        self.sub_size = sub_size
        self.sub_length = int(self.sub_size ** 2.0)
        self.sub_fraction = 1.0 / self.sub_length
        self.mapping = mapping.Mapping(mask=self)

    def __array_finalize__(self, obj):
        if hasattr(obj, "sub_size"):
            self.sub_size = obj.sub_size
            self.sub_length = int(obj.sub_size ** 2.0)
            self.sub_fraction = 1.0 / obj.sub_length
        if hasattr(obj, "mapping"):
            self.mapping = obj.mapping
        if hasattr(obj, "pixel_scale"):
            self.pixel_scale = obj.pixel_scale
        if hasattr(obj, "origin"):
            self.origin = obj.origin

    def __getitem__(self, coords):
        try:
            return super(Mask, self).__getitem__(coords)
        except IndexError:
            return True

    @classmethod
    def unmasked_from_shape_pixel_scale_and_sub_size(
        cls, shape, pixel_scale, sub_size, invert=False
    ):
        """Setup a mask where all pixels are unmasked.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        mask = np.full(tuple(map(lambda d: int(d), shape)), False)
        if invert:
            mask = np.invert(mask)
        return cls(array=mask, pixel_scale=pixel_scale, sub_size=sub_size)

    @classmethod
    def circular(
        cls,
        shape,
        pixel_scale,
        radius_arcsec,
        sub_size,
        centre=(0.0, 0.0),
        invert=False,
    ):
        """Setup a mask where unmasked pixels are within a circle of an input arc second radius and centre.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        radius_arcsec : float
            The radius (in arc seconds) of the circle within which pixels unmasked.
        centre: (float, float)
            The centre of the circle used to mask pixels.
        """
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(
            shape=shape,
            pixel_scale=pixel_scale,
            radius_arcsec=radius_arcsec,
            centre=centre,
        )
        if invert:
            mask = np.invert(mask)
        return cls(
            array=mask.astype("bool"), pixel_scale=pixel_scale, sub_size=sub_size
        )

    @classmethod
    def circular_annular(
        cls,
        shape,
        pixel_scale,
        sub_size,
        inner_radius_arcsec,
        outer_radius_arcsec,
        centre=(0.0, 0.0),
        invert=False,
    ):
        """Setup a mask where unmasked pixels are within an annulus of input inner and outer arc second radii and \
         centre.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        inner_radius_arcsec : float
            The radius (in arc seconds) of the inner circle outside of which pixels are unmasked.
        outer_radius_arcsec : float
            The radius (in arc seconds) of the outer circle within which pixels are unmasked.
        centre: (float, float)
            The centre of the annulus used to mask pixels.
        """

        mask = mask_util.mask_circular_annular_from_shape_pixel_scale_and_radii(
            shape=shape,
            pixel_scale=pixel_scale,
            inner_radius_arcsec=inner_radius_arcsec,
            outer_radius_arcsec=outer_radius_arcsec,
            centre=centre,
        )

        if invert:
            mask = np.invert(mask)

        return cls(
            array=mask.astype("bool"), pixel_scale=pixel_scale, sub_size=sub_size
        )

    @classmethod
    def circular_anti_annular(
        cls,
        shape,
        pixel_scale,
        sub_size,
        inner_radius_arcsec,
        outer_radius_arcsec,
        outer_radius_2_arcsec,
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
        pixel_scale: float
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

        mask = mask_util.mask_circular_anti_annular_from_shape_pixel_scale_and_radii(
            shape=shape,
            pixel_scale=pixel_scale,
            inner_radius_arcsec=inner_radius_arcsec,
            outer_radius_arcsec=outer_radius_arcsec,
            outer_radius_2_arcsec=outer_radius_2_arcsec,
            centre=centre,
        )

        if invert:
            mask = np.invert(mask)

        return cls(
            array=mask.astype("bool"), pixel_scale=pixel_scale, sub_size=sub_size
        )

    @classmethod
    def elliptical(
        cls,
        shape,
        pixel_scale,
        major_axis_radius_arcsec,
        axis_ratio,
        phi,
        sub_size,
        centre=(0.0, 0.0),
        invert=False,
    ):
        """ Setup a mask where unmasked pixels are within an ellipse of an input arc second major-axis and centre.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scale: float
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

        mask = mask_util.mask_elliptical_from_shape_pixel_scale_and_radius(
            shape, pixel_scale, major_axis_radius_arcsec, axis_ratio, phi, centre
        )

        if invert:
            mask = np.invert(mask)

        return cls(
            array=mask.astype("bool"), pixel_scale=pixel_scale, sub_size=sub_size
        )

    @classmethod
    def elliptical_annular(
        cls,
        shape,
        pixel_scale,
        sub_size,
        inner_major_axis_radius_arcsec,
        inner_axis_ratio,
        inner_phi,
        outer_major_axis_radius_arcsec,
        outer_axis_ratio,
        outer_phi,
        centre=(0.0, 0.0),
        invert=False,
    ):
        """Setup a mask where unmasked pixels are within an elliptical annulus of input inner and outer arc second \
        major-axis and centre.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scale: float
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

        mask = mask_util.mask_elliptical_annular_from_shape_pixel_scale_and_radius(
            shape=shape,
            pixel_scale=pixel_scale,
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

        return cls(
            array=mask.astype("bool"), pixel_scale=pixel_scale, sub_size=sub_size
        )

    @classmethod
    def mask_from_fits(cls, file_path, hdu, pixel_scale, sub_size, origin=(0.0, 0.0)):
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
        return Mask(
            array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu),
            pixel_scale=pixel_scale,
            sub_size=sub_size,
            origin=origin,
        )

    def new_mask_with_new_sub_size(self, sub_size):
        return Mask(
            array=self,
            pixel_scale=self.pixel_scale,
            sub_size=sub_size,
            origin=self.origin,
        )

    @property
    def sub_mask(self):

        sub_shape = (self.shape[0] * self.sub_size, self.shape[1] * self.sub_size)

        return mask_util.mask_from_shape_and_mask_1d_index_tomask_index(
            shape=sub_shape,
            mask_1d_index_tomask_index=self.mapping.sub_mask_1d_index_to_submask_index,
        )

    def binned_up_mask_from_mask(self, bin_up_factor):

        binned_up_mask = binning_util.binned_upmask_frommask_and_bin_up_factor(
            mask_2d=self, bin_up_factor=bin_up_factor
        )

        return Mask(
            array=binned_up_mask,
            pixel_scale=self.pixel_scale * bin_up_factor,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    def binned_up_mask_sub_size_1_from_mask(self, bin_up_factor):

        binned_up_mask = binning_util.binned_upmask_frommask_and_bin_up_factor(
            mask_2d=self, bin_up_factor=bin_up_factor
        )

        return Mask(
            array=binned_up_mask,
            pixel_scale=self.pixel_scale * bin_up_factor,
            sub_size=1,
            origin=self.origin,
        )

    @property
    @array_util.Memoizer()
    def centre(self):
        return grid_util.grid_centre_from_grid_1d(grid_1d=self.masked_grid_1d)

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))

    @array_util.Memoizer()
    def blurring_mask_from_psf_shape(self, psf_shape):
        """Compute a blurring mask, which represents all masked pixels whose light will be blurred into unmasked \
        pixels via PSF convolution (see grid.Grid.blurring_grid_from_mask_and_psf_shape).

        Parameters
        ----------
        psf_shape : (int, int)
           The shape of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        if psf_shape[0] % 2 == 0 or psf_shape[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = mask_util.blurring_mask_from_mask_and_psf_shape(self, psf_shape)

        return Mask(array=blurring_mask, pixel_scale=self.pixel_scale, sub_size=1)

    @property
    def edge_1d_indexes(self):
        """The indicies of the mask's edge pixels, where an edge pixel is any unmasked pixel on its edge \
        (next to at least one pixel with a *True* value).
        """
        return mask_util.edge_1d_indexes_from_mask(mask=self).astype("int")

    @property
    def border_1d_indexes(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return mask_util.border_1d_indexes_from_mask(mask=self).astype("int")

    @property
    def sub_border_1d_indexes(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=self, sub_size=self.sub_size
        ).astype("int")

    @property
    def border_grid_1d(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return self.masked_grid_1d[self.border_1d_indexes]

    @property
    def sub_border_grid_1d(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return self.masked_sub_grid_1d[self.sub_border_1d_indexes]

    @property
    def masked_grid_1d(self):
        return grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=self, pixel_scales=self.pixel_scales, sub_size=1, origin=self.origin
        )

    @property
    def masked_sub_grid_1d(self):
        return grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=self,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

    @property
    def zoom_centre(self):
        extraction_grid_1d = self.grid_arcsec_to_grid_pixels(
            grid_arcsec=self.masked_grid_1d
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
    def zoom_offset_pixels(self):
        return (
            self.zoom_centre[0] - self.central_pixel_coordinates[0],
            self.zoom_centre[1] - self.central_pixel_coordinates[1],
        )

    @property
    def zoom_offset_arcsec(self):
        return (
            -self.pixel_scale * self.zoom_offset_pixels[0],
            self.pixel_scale * self.zoom_offset_pixels[1],
        )

    @property
    def zoom_region(self):
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
        file_path=mask_path, hdu=mask_hdu, pixel_scale=pixel_scale, sub_size=sub_size
    )


def output_mask_to_fits(mask, mask_path, overwrite=False):
    array_util.numpy_array_2d_to_fits(
        array_2d=mask, file_path=mask_path, overwrite=overwrite
    )

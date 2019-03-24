import logging

import numpy as np

from autolens import exc
from autolens.data.array.util import grid_util, mapping_util, array_util, mask_util
from autolens.data.array import scaled_array

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask(scaled_array.ScaledSquarePixelArray):
    """
    A mask represented by an ndarray where True is masked.
    """

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scale, centre=(0.0, 0.0), origin=(0.0, 0.0)):
        """ A mask, which is applied to a 2D array of hyper to extract a set of unmasked image pixels (i.e. mask entry \
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
        self.centre = centre
        super(Mask, self).__init__(array=array, pixel_scale=pixel_scale, origin=origin)

    def __array_finalize__(self, obj):
        if hasattr(obj, "pixel_scale"):
            self.pixel_scale = obj.pixel_scale
        if hasattr(obj, "centre"):
            self.centre = obj.centre
        if hasattr(obj, 'origin'):
            self.origin = obj.origin

    def __getitem__(self, coords):
        try:
            return super(Mask, self).__getitem__(coords)
        except IndexError:
            return True

    @classmethod
    def unmasked_for_shape_and_pixel_scale(cls, shape, pixel_scale, invert=False):
        """Setup a mask where all pixels are unmasked.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        mask = np.full(tuple(map(lambda d: int(d), shape)), False)
        if invert: mask = np.invert(mask)
        return cls(array=mask, pixel_scale=pixel_scale)

    @classmethod
    def circular(cls, shape, pixel_scale, radius_arcsec, centre=(0., 0.), invert=False):
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
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape, pixel_scale, radius_arcsec,
                                                                         centre)
        if invert: mask = np.invert(mask)
        return cls(array=mask.astype('bool'), pixel_scale=pixel_scale, centre=centre)

    @classmethod
    def circular_annular(cls, shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec, centre=(0., 0.),
                         invert=False):
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
        mask = mask_util.mask_circular_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec,
                                                                                outer_radius_arcsec, centre)
        if invert: mask = np.invert(mask)
        return cls(array=mask.astype('bool'), pixel_scale=pixel_scale, centre=centre)

    @classmethod
    def circular_anti_annular(cls, shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec, outer_radius_2_arcsec,
                              centre=(0., 0.), invert=False):
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
        mask = mask_util.mask_circular_anti_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec,
                                                                                     outer_radius_arcsec,
                                                                                     outer_radius_2_arcsec, centre)
        if invert: mask = np.invert(mask)
        return cls(array=mask.astype('bool'), pixel_scale=pixel_scale, centre=centre)

    @classmethod
    def elliptical(cls, shape, pixel_scale, major_axis_radius_arcsec, axis_ratio, phi, centre=(0., 0.),
                   invert=False):
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
        mask = mask_util.mask_elliptical_from_shape_pixel_scale_and_radius(shape, pixel_scale, major_axis_radius_arcsec,
                                                                          axis_ratio, phi, centre)
        if invert: mask = np.invert(mask)
        return cls(array=mask.astype('bool'), pixel_scale=pixel_scale, centre=centre)

    @classmethod
    def elliptical_annular(cls, shape, pixel_scale,inner_major_axis_radius_arcsec, inner_axis_ratio, inner_phi,
                           outer_major_axis_radius_arcsec, outer_axis_ratio, outer_phi, centre=(0.0, 0.0),
                           invert=False):
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
        mask = mask_util.mask_elliptical_annular_from_shape_pixel_scale_and_radius(shape, pixel_scale,
                           inner_major_axis_radius_arcsec, inner_axis_ratio, inner_phi,
                           outer_major_axis_radius_arcsec, outer_axis_ratio, outer_phi, centre)
        if invert: mask = np.invert(mask)
        return cls(array=mask.astype('bool'), pixel_scale=pixel_scale, centre=centre)

    def binned_up_mask_from_mask(self, bin_up_factor):
        return Mask(array=mask_util.bin_up_mask_2d(mask_2d=self, bin_up_factor=bin_up_factor),
                    pixel_scale=self.pixel_scale*bin_up_factor, centre=self.centre, origin=self.origin)

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))

    @property
    def masked_grid_index_to_pixel(self):
        """A 1D array of mappings between every unmasked pixel and its 2D pixel coordinates."""
        return mask_util.masked_grid_1d_index_to_2d_pixel_index_from_mask(self).astype('int')

    def masked_sub_grid_index_to_sub_pixel(self, sub_grid_size):
        """A 1D array of mappings between every unmasked sub pixel and its 2D sub-pixel coordinates."""
        return mask_util.masked_sub_grid_1d_index_to_2d_sub_pixel_index_from_mask(
            self, sub_grid_size=sub_grid_size).astype('int')

    def map_2d_array_to_masked_1d_array(self, array_2d):
        """For a 2D array (e.g. an image, noise_map, etc.) map it to a masked 1D array of valuees using this mask.

        Parameters
        ----------
        array_2d : ndarray | None | float
            The 2D array to be mapped to a masked 1D array.
        """
        if array_2d is None or isinstance(array_2d, float):
            return array_2d
        return mapping_util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(self, array_2d)

    @array_util.Memoizer()
    def blurring_mask_for_psf_shape(self, psf_shape):
        """Compute a blurring mask, which represents all masked pixels whose light will be blurred into unmasked \
        pixels via PSF convolution (see grid_stack.RegularGrid.blurring_grid_from_mask_and_psf_shape).

        Parameters
        ----------
        psf_shape : (int, int)
           The shape of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        if psf_shape[0] % 2 == 0 or psf_shape[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(self, psf_shape)

        return Mask(blurring_mask, self.pixel_scale)

    @property
    def edge_pixels(self):
        """The indicies of the mask's edge pixels, where an edge pixel is any unmasked pixel on its edge \
        (next to at least one pixel with a *True* value).
        """
        return mask_util.edge_pixels_from_mask(self).astype('int')

    @property
    def border_pixels(self):
        """The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a *True* value but not central pixels like those within \
        an annulus mask).
        """
        return mask_util.border_pixels_from_mask(self).astype('int')

    @property
    def masked_grid_1d(self):
        return grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=self,
                                                                                  pixel_scales=self.pixel_scales,
                                                                                  origin=self.origin)

    @property
    def extraction_centre(self):
        extraction_grid_1d = self.grid_arcsec_to_grid_pixel_centres(grid_arcsec=self.masked_grid_1d)
        y_pixels_max = np.max(extraction_grid_1d[:,0])
        y_pixels_min = np.min(extraction_grid_1d[:,0])
        x_pixels_max = np.max(extraction_grid_1d[:,1])
        x_pixels_min = np.min(extraction_grid_1d[:,1])
        return (round((y_pixels_max + y_pixels_min) / 2.0), round((x_pixels_max + x_pixels_min) / 2.0))

    @property
    def extraction_offset(self):
        return (int(self.extraction_centre[0] - round(self.central_pixel_coordinates[0])),
                int(self.extraction_centre[1] - round(self.central_pixel_coordinates[1])))

    @property
    def extraction_region(self):
        """The rectangular region corresponding to the rectangle encompassing all unmasked values.

        This is used to extract and visualize only the region of an image that is used in an analysis."""
        # Have to convert mask to bool for invert function to work.
        where = np.array(np.where(np.invert(self.astype('bool'))))
        y0, x0 = np.amin(where, axis=1)
        y1, x1 = np.amax(where, axis=1)
        return [y0, y1+1, x0, x1+1]

def load_mask_from_fits(mask_path, pixel_scale, mask_hdu=0):
    return Mask.from_fits_with_pixel_scale(file_path=mask_path, hdu=mask_hdu, pixel_scale=pixel_scale)

def output_mask_to_fits(mask, mask_path, overwrite=False):
    array_util.numpy_array_2d_to_fits(array_2d=mask, file_path=mask_path, overwrite=overwrite)
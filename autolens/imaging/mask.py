from autolens.imaging import imaging_util
from autolens.imaging import scaled_array
from autolens import exc
import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask(scaled_array.ScaledArray):
    """
    A mask represented by an ndarray where True is masked.
    """

    def __getitem__(self, coords):
        try:
            return super(Mask, self).__getitem__(coords)
        except IndexError:
            return True

    @classmethod
    def empty_for_shape_arc_seconds_and_pixel_scale(cls, shape_arc_seconds, pixel_scale):
        return cls(np.full(tuple(map(lambda d: int(d / pixel_scale), shape_arc_seconds)), True), pixel_scale)

    @classmethod
    def circular(cls, shape, pixel_scale, radius_mask_arcsec, centre=(0., 0.)):
        """
        Setup the mask as a circle, using a specified arc second radius.

        Parameters
        ----------
        shape: (float, float)
            The (x,y) image_shape
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        radius_mask_arcsec : float
            The radius of the circular mask in arc seconds.
        centre: (float, float)
            The centre of the mask.
        """
        mask = imaging_util.mask_circular_from_shape_pixel_scale_and_radius(shape, pixel_scale, radius_mask_arcsec,
                                                                            centre)
        return cls(mask, pixel_scale)

    @classmethod
    def annular(cls, shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec, centre=(0., 0.)):
        """
        Setup the mask as a circle, using a specified inner and outer radius in arc seconds.

        Parameters
        ----------
        shape_arc_seconds : (float, float)
            The (x,y) image_shape of the mask
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        inner_radius : float
            The inner radius of the annulus mask in arc seconds.
        outer_radius : float
            The outer radius of the annulus mask in arc seconds.
        centre: (float, float)
            The centre of the mask.
        """
        mask = imaging_util.mask_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec,
                                                                          outer_radius_arcsec, centre)
        return cls(mask, pixel_scale)

    @classmethod
    def unmasked(cls, shape_arc_seconds, pixel_scale):
        """
        Setup the mask such that all values are unmasked, thus corresponding to the entire masked_image.

        Parameters
        ----------
        shape_arc_seconds : (float, float)
            The (x,y) image_shape of the mask
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        msk = Mask.empty_for_shape_arc_seconds_and_pixel_scale(shape_arc_seconds, pixel_scale)
        return Mask(np.ma.make_mask_none(msk.shape), pixel_scale)

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))

    @property
    def grid_to_pixel(self):
        """
        Compute the mapping_matrix of every pixel in the mask to its 2D pixel coordinates.
        """
        return imaging_util.grid_to_pixel_from_mask(self).astype('int')

    def map_2d_array_to_masked_1d_array(self, array):
        """Compute a data grid, which represents the data values of a data-set (e.g. an masked_image, noise, in the mask.

        Parameters
        ----------
        array: ndarray | float | None

        """
        if array is None or isinstance(array, float):
            return array
        return imaging_util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(self, array)

    def map_masked_1d_array_to_2d_array(self, data):
        """Use mapper to map an input data-set from a *GridData* to its original 2D masked_image.
        Parameters
        -----------
        data : ndarray
            The grid-data which is mapped to its 2D masked_image.
        """
        return imaging_util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(data, self.shape,
                                                                                               self.grid_to_pixel)

    @property
    def masked_image_grid(self):
        """
        Compute the masked_image grid_coords grids from a mask, using the center of every unmasked pixel.
        """
        return imaging_util.image_grid_masked_from_mask_and_pixel_scale(self, self.pixel_scale)

    @imaging_util.Memoizer()
    def blurring_mask_for_psf_shape(self, psf_shape):
        """Compute the blurring mask, which represents all data_to_pixels not in the mask but close enough to it that a
        fraction of their light will be blurring in the masked_image.

        Parameters
        ----------
        psf_shape : (int, int)
           The sub_grid_size of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        if psf_shape[0] % 2 == 0 or psf_shape[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = imaging_util.mask_blurring_from_mask_and_psf_shape(self, psf_shape)

        return Mask(blurring_mask, self.pixel_scale)

    @property
    def border_pixel_indices(self):
        """Compute the border masked_image data_to_pixels from a mask, where a border pixel is a pixel inside the mask but on
        its edge, therefore neighboring a pixel with a *True* value.
        """
        return imaging_util.border_pixels_from_mask(self).astype('int')

    def border_sub_pixel_indices(self, sub_grid_size):
        """Compute the border masked_image data_to_pixels from a mask, where a border pixel is a pixel inside the mask but on
        its edge, therefore neighboring a pixel with a *True* value.
        """
        return imaging_util.border_sub_pixels_from_mask_pixel_scale_and_sub_grid_size(self, self.pixel_scale,
                                                                                      sub_grid_size).astype('int')
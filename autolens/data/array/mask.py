import logging

import numpy as np

from autolens import exc
from autolens.data.array.util import mapping_util, array_util, mask_util
from autolens.data.array import scaled_array

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask(scaled_array.ScaledSquarePixelArray):
    """
    A masks represented by an ndarray where True is masked.
    """

    def __getitem__(self, coords):
        try:
            return super(Mask, self).__getitem__(coords)
        except IndexError:
            return True

    @classmethod
    def padded_for_shape_and_pixel_scale(cls, shape, pixel_scale):
        """
        Setup the masks such that all pixels are padded.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the masks in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls(np.full(tuple(map(lambda d: int(d), shape)), False, dtype='bool'), pixel_scale, origin=(0.0, 0.0))

    @classmethod
    def masked_for_shape_and_pixel_scale(cls, shape, pixel_scale):
        """
        Setup the masks such that all pixels are masked.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the masks in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls(np.full(tuple(map(lambda d: int(d), shape)), True, dtype='bool'), pixel_scale, origin=(0.0, 0.0))

    @classmethod
    def circular(cls, shape, pixel_scale, radius_mask_arcsec, centre=(0., 0.)):
        """
        Setup the masks as a circle, using a specified arc second radius.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the masks in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        radius_mask_arcsec : float
            The radius of the circular masks in arc seconds.
        centre: (float, float)
            The origin of the masks.
        """
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape, pixel_scale, radius_mask_arcsec,
                                                                         centre)
        return cls(mask.astype('bool'), pixel_scale, origin=(0.0, 0.0))

    @classmethod
    def annular(cls, shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec, centre=(0., 0.)):
        """
        Setup the masks as an annulus, using a specified inner and outer radius in arc seconds.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the masks in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        inner_radius_arcsec : float
            The inner radius of the annulus masks in arc seconds.
        outer_radius_arcsec : float
            The outer radius of the annulus masks in arc seconds.
        centre: (float, float)
            The origin of the masks.
        """
        mask = mask_util.mask_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec,
                                                                       outer_radius_arcsec, centre)
        return cls(mask.astype('bool'), pixel_scale, origin=(0.0, 0.0))

    @classmethod
    def anti_annular(cls, shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec, outer_radius_2_arcsec,
                     origin=(0., 0.)):
        """
        Setup the masks as an annulus, using a specified inner and outer radius in arc seconds.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the masks in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        inner_radius_arcsec : float
            The inner radius of the annulus masks in arc seconds.
        outer_radius_arcsec : float
            The outer radius of the annulus masks in arc seconds.
        outer_radius_2_arcsec
        origin: (float, float)
            The origin of the masks.
        """
        mask = mask_util.mask_anti_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec,
                                                                            outer_radius_arcsec,
                                                                            outer_radius_2_arcsec, origin)
        return cls(mask, pixel_scale)

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))

    @property
    def masked_grid_index_to_pixel(self):
        """A 1D array of mappings between every masked pixel and its 2D pixel coordinates."""
        return mask_util.masked_grid_1d_index_to_2d_pixel_index_from_mask(self).astype('int')

    def map_2d_array_to_masked_1d_array(self, array_2d):
        """For a 2D datas-array (e.g. the datas_, noise-mappers, etc.) mappers it to a masked 1D array of values usin this masks.

        Parameters
        ----------
        array_2d : ndarray | None | float
            The datas to be mapped to a masked 1D array.
        """
        if array_2d is None or isinstance(array_2d, float):
            return array_2d
        return mapping_util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(self, array_2d)

    @array_util.Memoizer()
    def blurring_mask_for_psf_shape(self, psf_shape):
        """Compute the blurring masks, which represents all masked pixels whose light will be blurred into padded \
        pixels via PSF convolution.

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
        """The indicies of the masks's edge pixels, where an edge pixel is any pixel inside the masks but on its edge \
        (next to at least one pixel with a *True* value).
        """
        return mask_util.edge_pixels_from_mask(self).astype('int')

    @property
    def border_pixels(self):
        """The indicies of the masks's edge pixels, where a border pixel is any pixel inside the masks but on an
         exterior edge (e.g. not the central pixels of an annulus mask).
        """
        return mask_util.border_pixels_from_mask(self).astype('int')
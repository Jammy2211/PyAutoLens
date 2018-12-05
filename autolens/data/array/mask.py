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
    def circular(cls, shape, pixel_scale, radius_arcsec, centre=(0., 0.)):
        """
        Setup the masks as a circle, using a specified arc second radius.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the masks in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        radius_arcsec : float
            The radius (in arc seconds) of the circle within which pixels are not masked.
        centre: (float, float)
            The centre of the circle used to mask pixels.
        """
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape, pixel_scale, radius_arcsec,
                                                                         centre)
        return cls(mask.astype('bool'), pixel_scale, origin=(0.0, 0.0))

    @classmethod
    def circular_annular(cls, shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec, centre=(0., 0.)):
        """
        Setup the masks as an annulus, using a specified inner and outer radius in arc seconds.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the masks in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        inner_radius_arcsec : float
            The radius (in arc seconds) of the inner circle outside of which pixels are not masked.
        outer_radius_arcsec : float
            The radius (in arc seconds) of the outer circle within which pixels are not masked.
        centre: (float, float)
            The origin of the masks.
        """
        mask = mask_util.mask_circular_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec,
                                                                                outer_radius_arcsec, centre)
        return cls(mask.astype('bool'), pixel_scale, origin=(0.0, 0.0))

    @classmethod
    def circular_anti_annular(cls, shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec, outer_radius_2_arcsec,
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
            The radius (in arc seconds) of the inner circle inside of which pixels are not masked.
        outer_radius_arcsec : float
            The radius (in arc seconds) of the outer circle within which pixels are masked and outside of which they \
            are not.
        outer_radius_2_arcsec : float
            The radius (in arc seconds) of the second outer circle within which pixels are not masked and outside of \
            which they are.
        origin: (float, float)
            The origin of the masks.
        """
        mask = mask_util.mask_circular_anti_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec,
                                                                                     outer_radius_arcsec,
                                                                                     outer_radius_2_arcsec, origin)
        return cls(mask, pixel_scale)

    @classmethod
    def elliptical(cls, shape, pixel_scale, major_axis_radius_arcsec, axis_ratio, phi, centre=(0., 0.)):
        """
        Setup the masks as a ellipse, using a specified arc second major axis, axis-ratio and rotation angle phi \
        defined counter-clockwise from the positive x-axis.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the masks in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        major_axis_radius_arcsec : float
            The major-axis (in arc seconds) of the ellipse within which pixels are not masked.
        axis_ratio : float
            The axis-ratio of the ellipse within which pixels are not masked.
        phi : float
            The rotation angle of the ellipse within which pixels are not masked, defined counter-clockwise from the \
             positive x-axis.
        centre: (float, float)
            The origin of the masks.
        """
        mask = mask_util.mask_elliptical_from_shape_pixel_scale_and_radius(shape, pixel_scale, major_axis_radius_arcsec,
                                                                          axis_ratio, phi, centre)
        return cls(mask.astype('bool'), pixel_scale, origin=(0.0, 0.0))

    @classmethod
    def elliptical_annular(cls, shape, pixel_scale,inner_major_axis_radius_arcsec, inner_axis_ratio, inner_phi,
                           outer_major_axis_radius_arcsec, outer_axis_ratio, outer_phi, centre=(0.0, 0.0)):
        """
        Setup the masks as a ellipse, using a specified arc second major axis, axis-ratio and rotation angle phi \
        defined counter-clockwise from the positive x-axis.

        Parameters
        ----------
        shape: (int, int)
            The (y,x) shape of the masks in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        inner_major_axis_radius_arcsec : float
            The major-axis (in arc seconds) of the inner ellipse within which pixels are masked.
        inner_axis_ratio : float
            The axis-ratio of the inner ellipse within which pixels are masked.
        inner_phi : float
            The rotation angle of the inner ellipse within which pixels are masked, defined counter-clockwise from the \
            positive x-axis.
        outer_major_axis_radius_arcsec : float
            The major-axis (in arc seconds) of the outer ellipse within which pixels are not masked.
        outer_axis_ratio : float
            The axis-ratio of the outer ellipse within which pixels are not masked.
        outer_phi : float
            The rotation angle of the outer ellipse within which pixels are not masked, defined counter-clockwise from
            the positive x-axis.
        centre: (float, float)
            The origin of the masks.
        """
        mask = mask_util.mask_elliptical_annular_from_shape_pixel_scale_and_radius(shape, pixel_scale,
                           inner_major_axis_radius_arcsec, inner_axis_ratio, inner_phi,
                           outer_major_axis_radius_arcsec, outer_axis_ratio, outer_phi, centre)
        return cls(mask.astype('bool'), pixel_scale, origin=(0.0, 0.0))

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
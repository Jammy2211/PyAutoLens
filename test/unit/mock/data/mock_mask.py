import autolens as al

import numpy as np

from autolens import exc
from autolens.array import mapping
from autolens.array.util import mask_util


class MockMask(al.Mask):
    def __new__(cls, array, pixel_scale=1.0, sub_size=1, *args, **kwargs):

        obj = np.array(array, dtype="bool").view(cls)
        obj.pixel_scale = pixel_scale
        obj.sub_size = sub_size
        obj.sub_length = int(sub_size ** 2.0)
        obj.sub_fraction = 1.0 / obj.sub_length
        obj.origin = (0.0, 0.0)
        obj.mapping = mapping.Mapping(mask=obj)

        return obj

    def __init__(self, array, pixel_scale=1.0, sub_size=1):
        pass

    def blurring_mask_from_psf_shape(self, psf_shape):
        """Compute a blurring mask, which represents all masked pixels whose light will be blurred into unmasked \
        pixels via PSF convolution (see grid.Grid.blurring_grid_from_mask_and_psf_shape).

        Parameters
        ----------
        psf_shape : (int, int)
           The shape of the psf which defines the blurring region (e.al. the shape of the PSF)
        """

        if psf_shape[0] % 2 == 0 or psf_shape[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = al.mask_util.blurring_mask_from_mask_and_psf_shape(
            self, psf_shape
        )

        return MockMask(array=blurring_mask, pixel_scale=self.pixel_scale)


class MockMask1D(np.ndarray):
    def __new__(cls, shape, pixel_scale=1.0, *args, **kwargs):

        array = np.full(fill_value=False, shape=shape)

        obj = np.array(array, dtype="bool").view(cls)
        obj.pixel_scale = pixel_scale
        obj.origin = (0.0, 0.0)

        return obj

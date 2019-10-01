import autolens as al

import numpy as np

from autolens import exc
from autoarray.util import mask_util


class MockMask(al.Mask):
    def __new__(
        cls,
        array_2d,
        pixel_scales=(1.0, 1.0),
        sub_size=1,
        origin=(0.0, 0.0),
        *args,
        **kwargs
    ):

        obj = array_2d.view(cls)
        obj.pixel_scales = pixel_scales
        obj.sub_size = sub_size
        obj.sub_length = int(obj.sub_size ** 2.0)
        obj.sub_fraction = 1.0 / obj.sub_length
        obj.origin = origin

        return obj

    def __init__(
        self,
        array_2d,
        pixel_scales=(1.0, 1.0),
        sub_size=1,
        origin=(0.0, 0.0),
        *args,
        **kwargs
    ):
        pass

    def blurring_mask_from_kernel_shape(self, kernel_shape):
        """Compute a blurring mask, which represents all masked pixels whose light will be blurred into unmasked \
        pixels via PSF convolution (see grid.Grid.blurring_grid_from_mask_and_psf_shape).

        Parameters
        ----------
        kernel_shape : (int, int)
           The shape of the psf which defines the blurring region (e.al. the shape of the PSF)
        """

        if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = al.mask_util.blurring_mask_from_mask_and_kernel_shape(
            self, kernel_shape
        )

        return MockMask(array_2d=blurring_mask, pixel_scales=self.pixel_scales)


class MockMask1D(np.ndarray):
    def __new__(cls, shape, pixel_scale=1.0, *args, **kwargs):

        array = np.full(fill_value=False, shape=shape)

        obj = np.array(array, dtype="bool").view(cls)
        obj.pixel_scale = pixel_scale
        obj.origin = (0.0, 0.0)

        return obj

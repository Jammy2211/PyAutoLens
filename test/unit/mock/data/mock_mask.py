from autolens.data.array import mask as msk

import numpy as np

class MockMask(msk.Mask):

    def __new__(cls, array, pixel_scale=1.0, *args, **kwargs):

        obj = np.array(array, dtype='float64').view(cls)
        obj.pixel_scale = pixel_scale
        obj.origin = (0.0, 0.0)

        return obj

    def __init__(self, array, pixel_scale=1.0):
        pass


class MockMask1D(np.ndarray):

    def __new__(cls, shape, pixel_scale=1.0, *args, **kwargs):

        array = np.full(fill_value=False, shape=shape)

        obj = np.array(array, dtype='bool').view(cls)
        obj.pixel_scale = pixel_scale
        obj.origin = (0.0, 0.0)

        return obj
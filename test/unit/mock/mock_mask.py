from autolens.data.array import mask as msk

import numpy as np

class MockMask(msk.Mask):

    def __new__(cls, *args, **kwargs):

        array = np.array([[True, True,  True,  True,  True],
                          [True, False, False, False, True],
                          [True, False, False, False, True],
                          [True, False, False, False, True],
                          [True,  True,  True,  True, True]])

        obj = np.array(array, dtype='float64').view(cls)
        obj.pixel_scale = 2.0
        obj.origin = (0.0, 0.0)

        return obj

    def __init__(self):
        pass


class MockBlurringMask(msk.Mask):

    def __new__(cls, *args, **kwargs):

        array = np.array([[False, False, False, False, False],
                          [False,  True,  True,  True, False],
                          [False,  True,  True,  True, False],
                          [False,  True,  True,  True, False],
                          [False, False, False, False, False]])

        obj = np.array(array, dtype='float64').view(cls)
        obj.pixel_scale = 2.0
        obj.origin = (0.0, 0.0)

        return obj

    def __init__(self):
        pass
import numpy as np
import scipy.signal

from autolens.data import convolution
from autolens.model.inversion import convolution as inversion_convolution

from test.unit.mock.mock_mask import MockMask, MockBlurringMask

class MockImage(np.ndarray):

    def __new__(cls, *args, **kwargs):

        array = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 2.0, 3.0, 0.0],
                          [0.0, 4.0, 5.0, 6.0, 0.0],
                          [0.0, 7.0, 8.0, 9.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0]])

        obj = np.array(array, dtype='float64').view(cls)
        obj.pixel_scale = 2.0
        obj.origin = (0.0, 0.0)

        return obj

class MockImage1D(np.ndarray):

    def __new__(cls, *args, **kwargs):

        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        obj = np.array(array, dtype='float64').view(cls)
        obj.pixel_scale = 2.0
        obj.origin = (0.0, 0.0)

        return obj

class MockNoiseMap(np.ndarray):

    def __new__(cls, *args, **kwargs):

        array = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 1.0, 1.0, 0.0],
                          [0.0, 1.0, 1.0, 1.0, 0.0],
                          [0.0, 1.0, 1.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0]])


        obj = np.array(array, dtype='float64').view(cls)
        obj.pixel_scale = 2.0
        obj.origin = (0.0, 0.0)

        return obj


class MockNoiseMap1D(np.ndarray):

    def __new__(cls, *args, **kwargs):

        array = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        obj = np.array(array, dtype='float64').view(cls)
        obj.pixel_scale = 2.0
        obj.origin = (0.0, 0.0)

        return obj


class MockPSF(np.ndarray):

    def __new__(cls, *args, **kwargs):

        array = np.array([[1.0, 5.0, 9.0],
                          [2.0, 5.0, 1.0],
                          [3.0, 4.0, 0.0]])

        obj = np.array(array, dtype='float64').view(cls)
        obj.pixel_scale = 2.0
        obj.origin = (0.0, 0.0)

        return obj

    def convolve(self, array):
        return scipy.signal.convolve2d(array, self, mode='same')


class MockConvolverImage(convolution.ConvolverImage):

    def __init__(self):

        super(MockConvolverImage, self).__init__(mask=MockMask(), blurring_mask=MockBlurringMask(), psf=MockPSF())


class MockConvolverMappingMatrix(inversion_convolution.ConvolverMappingMatrix):

    def __init__(self):
        super(MockConvolverMappingMatrix, self).__init__(mask=MockMask(), psf=MockPSF())
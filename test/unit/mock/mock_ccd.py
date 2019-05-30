import numpy as np
import scipy.signal

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
import numpy as np
import scipy.signal

from autolens.data import ccd
from autolens.data.array import scaled_array

class MockImage(object):

    def __new__(cls, shape, value, pixel_scale=1.0):

        return scaled_array.ScaledSquarePixelArray(array=value*np.ones(shape=shape), pixel_scale=pixel_scale)


class MockNoiseMap(object):

    def __new__(cls, shape, value, pixel_scale=1.0):

        return scaled_array.ScaledSquarePixelArray(array=value*np.ones(shape=shape), pixel_scale=pixel_scale)


class MockBackgroundNoiseMap(object):

    def __new__(cls, shape, value, pixel_scale=1.0):

        return scaled_array.ScaledSquarePixelArray(array=value*np.ones(shape=shape), pixel_scale=pixel_scale)


class MockPoissonNoiseMap(object):

    def __new__(cls, shape, value, pixel_scale=1.0):

        return scaled_array.ScaledSquarePixelArray(array=value*np.ones(shape=shape), pixel_scale=pixel_scale)

class MockExposureTimeMap(object):

    def __new__(cls, shape, value, pixel_scale=1.0):

        return scaled_array.ScaledSquarePixelArray(array=value*np.ones(shape=shape), pixel_scale=pixel_scale)


class MockBackgrondSkyMap(object):

    def __new__(cls, shape, value, pixel_scale=1.0):

        return scaled_array.ScaledSquarePixelArray(array=value*np.ones(shape=shape), pixel_scale=pixel_scale)


class MockPSF(object):

    def __new__(cls, shape, value, pixel_scale=1.0, *args, **kwargs):

        return ccd.PSF(array=value*np.ones(shape=shape), pixel_scale=pixel_scale, origin=(0.0, 0.0))


class MockImage1D(np.ndarray):

    def __new__(cls, shape, value, pixel_scale=1.0):

        array = value*np.ones(shape=shape)

        obj = np.array(array, dtype='float64').view(cls)
        obj.pixel_scale = pixel_scale
        obj.origin = (0.0, 0.0)

        return obj


class MockNoiseMap1D(np.ndarray):

    def __new__(cls, shape, value, pixel_scale=1.0):

        array = value*np.ones(shape=shape)

        obj = np.array(array, dtype='float64').view(cls)
        obj.pixel_scale = pixel_scale
        obj.origin = (0.0, 0.0)

        return obj


class MockCCDData(ccd.CCDData):

    def __init__(self, image, pixel_scale, psf, noise_map, background_noise_map, poisson_noise_map,
                 exposure_time_map, background_sky_map, name):

        super(MockCCDData, self).__init__(
            image=image, pixel_scale=pixel_scale, psf=psf, noise_map=noise_map,
            background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
            exposure_time_map=exposure_time_map, background_sky_map=background_sky_map, name=name)
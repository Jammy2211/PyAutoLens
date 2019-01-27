import time
from autolens.data import ccd


import os

path = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))

def pixel_scale_from_image_type(image_type):

    if image_type == 'LSST':
        return 0.2
    elif image_type == 'Euclid':
        return 0.1
    elif image_type == 'HST':
        return 0.05
    elif image_type == 'HST_Up':
        return 0.03
    elif image_type == 'AO':
        return 0.01

def image_type_from_pixel_scale(pixel_scale):

    if pixel_scale == 0.2:
        return 'LSST'
    elif pixel_scale == 0.1:
        return 'Euclid'
    elif pixel_scale == 0.05:
        return 'HST'
    elif pixel_scale == 0.03:
        return 'HST_Up'
    elif pixel_scale == 0.01:
        return 'AO'

def load_profiling_ccd_data(image_type, lens_name, psf_shape):

    pixel_scale = pixel_scale_from_image_type(image_type=image_type)

    return ccd.load_ccd_data_from_fits(image_path=path + '/data/' + lens_name + '/' + image_type + '/image.fits',
                                       psf_path=path + '/data/' + lens_name + '/' + image_type + '/psf.fits',
                                       noise_map_path=path + '/data/' + lens_name + '/' + image_type + '/noise_map.fits',
                                       pixel_scale=pixel_scale, resized_psf_shape=psf_shape)


def tick_toc_x1(func):
    def wrapper():
        start = time.time()
        for _ in range(1):
            func()

        diff = time.time() - start
        print("{}: {}".format(func.__name__, diff / 1.0))

    return wrapper


def tick_toc_x10(func):
    def wrapper():
        start = time.time()
        for _ in range(10):
            func()

        diff = time.time() - start
        print("{}: {}".format(func.__name__, diff / 10.0))

    return wrapper


def tick_toc_x20(func):
    def wrapper():
        start = time.time()
        for _ in range(10):
            func()

        diff = time.time() - start
        print("{}: {}".format(func.__name__, diff / 20.0))

    return wrapper

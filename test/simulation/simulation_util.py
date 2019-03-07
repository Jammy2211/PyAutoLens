import time
from autolens.data import ccd


import os

path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

def pixel_scale_from_data_type(data_type):

    if data_type == 'LSST':
        return 0.2
    elif data_type == 'Euclid':
        return 0.1
    elif data_type == 'HST':
        return 0.05
    elif data_type == 'HST_Up':
        return 0.03
    elif data_type == 'AO':
        return 0.01
    else:
        raise ValueError('An invalid data-type was entered when generating the test-data suite - ', data_type)

def shape_from_data_type(data_type):

    if data_type == 'LSST':
        return (100, 100)
    elif data_type == 'Euclid':
        return (150, 150)
    elif data_type == 'HST':
        return (250, 250)
    elif data_type == 'HST_Up':
        return (320, 320)
    elif data_type == 'AO':
        return (750, 750)
    else:
        raise ValueError('An invalid data-type was entered when generating the test-data suite - ', data_type)

def data_type_from_pixel_scale(pixel_scale):

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
    else:
        raise ValueError('An invalid pixel-scale was entered when generating the data-type - ', pixel_scale)

def load_test_ccd_data(data_type, data_name, psf_shape=(11, 11)):

    pixel_scale = pixel_scale_from_data_type(data_type=data_type)

    return ccd.load_ccd_data_from_fits(image_path=path + '/data/' + data_name + '/' + data_type + '/image.fits',
                                       psf_path=path + '/data/' + data_name + '/' + data_type + '/psf.fits',
                                       noise_map_path=path + '/data/' + data_name + '/' + data_type + '/noise_map.fits',
                                       pixel_scale=pixel_scale, resized_psf_shape=psf_shape)
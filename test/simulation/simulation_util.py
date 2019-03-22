import os

from autofit.tools import path_util
from autolens.data import ccd

test_path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))


def pixel_scale_from_data_resolution(data_resolution):
    if data_resolution == 'LSST':
        return 0.2
    elif data_resolution == 'Euclid':
        return 0.1
    elif data_resolution == 'HST':
        return 0.05
    elif data_resolution == 'HST_Up':
        return 0.03
    elif data_resolution == 'AO':
        return 0.01
    else:
        raise ValueError('An invalid data resolution was entered - ', data_resolution)


def shape_from_data_resolution(data_resolution):
    if data_resolution == 'LSST':
        return 100, 100
    elif data_resolution == 'Euclid':
        return 150, 150
    elif data_resolution == 'HST':
        return 250, 250
    elif data_resolution == 'HST_Up':
        return 320, 320
    elif data_resolution == 'AO':
        return 750, 750
    else:
        raise ValueError('An invalid data-type was entered - ', data_resolution)


def data_resolution_from_pixel_scale(pixel_scale):
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
        raise ValueError('An invalid pixel-scale was entered - ', pixel_scale)


def load_test_ccd_data(data_type, data_resolution, psf_shape=(11, 11), lens_name=None):
    pixel_scale = pixel_scale_from_data_resolution(data_resolution=data_resolution)

    data_path = path_util.make_and_return_path_from_path_and_folder_names(
        path=test_path, folder_names=['data', data_type, data_resolution])

    return ccd.load_ccd_data_from_fits(image_path=data_path + '/image.fits',
                                       psf_path=data_path + '/psf.fits',
                                       noise_map_path=data_path + '/noise_map.fits',
                                       pixel_scale=pixel_scale, resized_psf_shape=psf_shape, lens_name=lens_name)

import os
import pickle

from autolens.imaging import image
from autolens.imaging import mask
from autolens.imaging import scaled_array
from autolens.lensing import lensing_image

path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


def load_data(name, pixel_scale, psf_shape):
    im = scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=path + 'datas/' + name + '/masked_image', hdu=0,
                                                                        pixel_scale=pixel_scale)
    noise = scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=path + 'datas/' + name + '/noise_map_', hdu=0,
                                                                           pixel_scale=pixel_scale)
    exposure_time = scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=path + 'datas/' + name + '/exposure_time',
                                                                                   hdu=0,
                                                                                   pixel_scale=pixel_scale)
    psf = image.PSF.from_fits_with_scale(file_path=path + 'datas/LSST/psf', hdu=0, pixel_scale=pixel_scale).resized_scaled_array_from_array(psf_shape)

    return im, noise, exposure_time, psf


def setup_class(name, pixel_scale, radius_mask=4.0, psf_shape=(21, 21), sub_grid_size=4):
    def pickle_path():
        return path + 'datas/' + name + '/pickle/r' + str(radius_mask) + '_psf' + str(psf_shape[0]) + '_sub' + \
               str(sub_grid_size)

    if not os.path.isdir(path + 'datas/' + name + '/pickle'):
        os.mkdir(path + 'datas/' + name + '/pickle')

    if not os.path.isfile(pickle_path()):
        return Data(name, pixel_scale, radius_mask, psf_shape, sub_grid_size)
    elif os.path.isfile(pickle_path()):
        with open(pickle_path(), 'rb') as pickle_file:
            thing = pickle.load(file=pickle_file)
        return thing


class Data(object):

    def __init__(self, name, pixel_scale, radius_mask=4.0, psf_shape=(21, 21), sub_grid_size=4):
        def pickle_path():
            return path + 'datas/' + name + '/pickle/r' + str(radius_mask) + '_psf' + str(psf_shape[0]) + '_sub' + \
                   str(sub_grid_size)

        im, noise, exposure_time, psf = load_data(name=name, pixel_scale=pixel_scale, psf_shape=psf_shape)

        im = image.Image(array=im, effective_exposure_time=exposure_time, pixel_scales=pixel_scale, psf=psf,
                         background_noise=noise, poisson_noise=noise)

        ma = mask.Mask.circular(shape=im.shape_arc_seconds, pixel_scale=im.pixel_scales,
                                radius_mask_arcsec=radius_mask)

        self.masked_image = lensing_image.LensingImage(image=im, mask=ma)

        self.grids = mask.GridCollection.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma,
                                                                                     sub_grid_size=sub_grid_size,
                                                                                     psf_shape=psf.shape)

        self.borders = mask.ImagingGridBorders.from_mask(mask=ma, sub_grid_size=sub_grid_size)

        with open(pickle_path(), 'wb') as pickle_file:
            pickle.dump(self, file=pickle_file)

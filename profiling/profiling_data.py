import os
import pickle

from src.imaging import scaled_array
from src.imaging import image
from src.imaging import mask
from src.imaging import masked_image

path =  "{}/".format(os.path.dirname(os.path.realpath(__file__)))

def load_data(name, pixel_scale, psf_shape):
    im = scaled_array.ScaledArray.from_fits(file_path=path + 'data/'+name+'/image', hdu=0, pixel_scale=pixel_scale)
    noise = scaled_array.ScaledArray.from_fits(file_path=path + 'data/'+name+'/noise', hdu=0, pixel_scale=pixel_scale)
    exposure_time = scaled_array.ScaledArray.from_fits(file_path=path + 'data/'+name+'/exposure_time', hdu=0,
                                                       pixel_scale=pixel_scale)
    psf = image.PSF.from_fits(file_path=path + 'data/LSST/psf', hdu=0, pixel_scale=pixel_scale).trim(psf_shape)

    return im, noise, exposure_time, psf

def setup_class(name, pixel_scale, radius_mask=4.0, psf_shape=(21,21), subgrid_size=4):

    def pickle_path():
        return path + 'data/' + name + '/pickle/r' + str(radius_mask) + '_psf' + str(psf_shape[0]) + '_sub' + \
               str(subgrid_size)

    if not os.path.isdir(path + 'data/' + name + '/pickle'):
        os.mkdir(path + 'data/' + name + '/pickle')

    if not os.path.isfile(pickle_path()):
        return Data(name, pixel_scale, radius_mask, psf_shape, subgrid_size)
    elif os.path.isfile(pickle_path()):
        with open(pickle_path(), 'rb') as pickle_file:
             thing=pickle.load(file=pickle_file)
        return thing


class Data(object):

    def __init__(self, name, pixel_scale, radius_mask=4.0, psf_shape=(21, 21), subgrid_size=4):

        def pickle_path():
            return path + 'data/' + name + '/pickle/r' + str(radius_mask) + '_psf' + str(psf_shape[0]) + '_sub' + \
                   str(subgrid_size)

        im, noise, exposure_time, psf = load_data(name=name, pixel_scale=pixel_scale, psf_shape=psf_shape)

        self.image = image.Image(array=im, effective_exposure_time=exposure_time, pixel_scale=pixel_scale, psf=psf,
                         background_noise=noise, poisson_noise=noise)

        self.mask = mask.Mask.circular(shape_arc_seconds=self.image.shape_arc_seconds,
                                       pixel_scale=self.image.pixel_scale, radius_mask=radius_mask)

        self.masked_image = masked_image.MaskedImage(image=self.image, mask=self.mask)

        self.coords = mask.GridCollection.from_mask_subgrid_size_and_blurring_shape(mask=self.mask,
                                                                                    subgrid_size=subgrid_size,
                                                                                    blurring_shape=psf.shape)

        with open(pickle_path(), 'wb') as pickle_file:
            pickle.dump(self, file=pickle_file)

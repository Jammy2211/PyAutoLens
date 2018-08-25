import numpy as np
from astropy.io import fits
import os

def numpy_array_to_fits(array, file_path):

    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(array, new_hdr)
    hdu.writeto(file_path + '.fits')

def numpy_array_from_fits(file_path, hdu):
    hdu_list = fits.open(file_path + '.fits')
    return np.array(hdu_list[hdu].data)

def compute_variances_from_noise(noise):
    """The variances are the signal_to_noise_ratio (standard deviations) squared."""
    return np.square(noise)
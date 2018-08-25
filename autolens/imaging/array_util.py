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

def compute_residuals(observed_image, model_image):
    """ Calculate the residuals between an observed masked_image and a model of that masked_image, \
    by subtracting the model from the data_vector.

    Residuals = (Data - Model).

    Parameters
    ----------
    observed_image : ndarray
        Two dimensional array containing observed masked_image data_vector
    model_image : nddarray
        Two dimensional array containing model_mapper of the observed masked_image
    """
    return np.subtract(observed_image, model_image)

def compute_variances_from_noise(noise):
    """The variances are the signal_to_noise_ratio (standard deviations) squared."""
    return np.square(noise)

def compute_chi_sq_image(observed_image, model_image, noise):
    """ Calculate the chi-squared masked_image of an observed masked_image, model masked_image and signal_to_noise_ratio map.

    The signal_to_noise_ratio map gives the standard deviation of each masked_image pixel, and is converted to variances (by squaring) in the
     chi squared calculation.

     Chi_Sqs = ((Data - Model)/Noise)**2.0 = (Residuals/Noise)**2.0 = (Residuals**2.0/Variances).

    Parameters
    ----------
    observed_image : ndarray
        Two dimensional array containing observed masked_image data_vector
    model_image : nddarray
        Two dimensional array containing model_mapper of the observed masked_image
    noise : nddarray
        Two dimensional array containing the signal_to_noise_ratio (standard deviation) in each observed masked_image pixel.
    """
    residuals = compute_residuals(observed_image, model_image)
    variances = compute_variances_from_noise(noise)
    return np.divide(np.square(residuals), variances)

def compute_likelihood(observed_image, model_image, noise):
    """ Calculate the likelihood masked_image of an observed masked_image, model masked_image and signal_to_noise_ratio map. The likelihood is defined as
    the sum of chi squared values multiplied by 0.5.

    The signal_to_noise_ratio map gives the standard deviation of each masked_image pixel, and is converted to variances (by squaring) in the
    chi squared calculation.

    NOTE1 : For efficiency, this routine uses only numpy routines in one big calculation, thus it is 'unpythonic'.

    Parameters
    ----------
    observed_data : ndarray
        Two dimensional array containing the observed masked_image data_vector
    model_image : nddarray
        Two dimensional array containing the model of the observed masked_image
    noise : nddarray
        Two dimensional array containing the signal_to_noise_ratio (standard deviation) in each observed masked_image pixel
    """
    return -0.5 * np.sum(np.square(np.divide(np.subtract(observed_image, model_image), noise)))


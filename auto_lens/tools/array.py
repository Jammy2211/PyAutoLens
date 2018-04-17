import numpy as np



def compute_residuals(observed_image, model_image):
    """ Calculate the residuals between an observed image and a model of that image, \
    by subtracting the model from the data.

    Residuals = (Data - Model).

    Parameters
    ----------
    observed_image : ndarray
        Two dimensional array containing observed image data
    model_image : nddarray
        Two dimensional array containing model_mapper of the observed image
    """
    return np.subtract(observed_image, model_image)

def compute_variances_from_noise(noise):
    """The variances are the noise (standard deviations) squared."""
    return np.square(noise)

def compute_chi_sq_image(observed_image, model_image, noise):
    """ Calculate the chi-squared image of an observed image, model image and noise map.

    The noise map gives the standard deviation of each image pixel, and is converted to variances (by squaring) in the
     chi squared calculation.

     Chi_Sqs = ((Data - Model)/Noise)**2.0 = (Residuals/Noise)**2.0 = (Residuals**2.0/Variances).

    Parameters
    ----------
    observed_image : ndarray
        Two dimensional array containing observed image data
    model_image : nddarray
        Two dimensional array containing model_mapper of the observed image
    noise : nddarray
        Two dimensional array containing the noise (standard deviation) in each observed image pixel.
    """
    residuals = compute_residuals(observed_image, model_image)
    variances = compute_variances_from_noise(noise)
    return np.divide(np.square(residuals), variances)

def compute_likelihood(observed_image, model_image, noise):
    """ Calculate the likelihood image of an observed image, model image and noise map. The likelihood is defined as
    the sum of chi squared values multiplied by 0.5.

    The noise map gives the standard deviation of each image pixel, and is converted to variances (by squaring) in the
    chi squared calculation.

    NOTE1 : For efficiency, this routine uses only numpy routines in one big calculation, thus it is 'unpythonic'.

    Parameters
    ----------
    observed_data : ndarray
        Two dimensional array containing the observed image data
    model_image : nddarray
        Two dimensional array containing the model of the observed image
    noise : nddarray
        Two dimensional array containing the noise (standard deviation) in each observed image pixel
    """
    return -0.5 * np.sum(np.square(np.divide(np.subtract(observed_image, model_image), noise)))

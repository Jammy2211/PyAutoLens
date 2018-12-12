import numpy as np

def residuals_from_data_mask_and_model_data(data, mask, model_data):
    """Compute the residuals between a list of 1D masked observed datas and model datas, where:

    Residuals = (Data - Model_Data).

    For strong lens imaging, this subtracts the model lens regular from the observed regular within the mask.

    Parameters
    -----------
    data : np.ndarray
        The 1D masked observed data-set.
    model_data : np.ndarray
        The 1D masked model data-set.
    """
    residuals = np.subtract(data, model_data)
    return residuals - residuals * mask

def chi_squareds_from_residuals_and_noise_map(residuals, noise_map):
    """Computes the chi-squared images between a list of 1D masked residuals and noise-maps, where:

    Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    Parameters
    -----------
    residuals : np.ndarray
        The 1D masked residual of the model-data fit to the observed data.
    noise_map : np.ndarray
        The 1D masked noise-map of the observed data.
    """
    return np.square((np.divide(residuals, noise_map)))

def chi_squared_term_from_chi_squareds(chi_squareds):
    """Compute the chi-squared terms of each model's data-set's fit to an observed data-set, by summing the 1D masked
    chi-squared values of the fit.

    Parameters
    ----------
    chi_squareds : np.ndarray
        The 1D masked chi-squared values of the model-data fit to the observed data.
    """
    return np.sum(chi_squareds)

def noise_term_from_mask_and_noise_map(mask, noise_map):
    """Compute the noise-map normalization terms of a list of masked 1D noise-maps, summing the noise vale in every
    pixel as:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise_map : np.ndarray
        List of masked 1D noise-maps.
    """
    masked_noise_map = np.extract(condition=np.logical_not(mask), arr=noise_map)
    return np.sum(np.log(2 * np.pi * masked_noise_map ** 2.0))

def likelihood_from_chi_squared_term_and_noise_term(chi_squared_term, noise_term):
    """Compute the likelihood of each masked 1D model-datas fit to the data, where:

    Likelihood = -0.5*[Chi_Squared_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared_term : float
        The chi-squared term for the model-data fit to the observed data.
    noise_term : float
        The normalization noise-term for the observed data's noise-map.
    """
    return -0.5 * (chi_squared_term + noise_term)
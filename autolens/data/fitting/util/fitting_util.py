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

def chi_squared_from_residuals_and_noise_map(residuals, noise_map):
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

def noise_term_from_noise_map(noise_map):
    """Compute the noise-map normalization terms of a list of masked 1D noise-maps, summing the noise vale in every
    pixel as:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise_map : np.ndarray
        List of masked 1D noise-maps.
    """
    return np.sum(np.log(2 * np.pi * noise_map ** 2.0))

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

def map_arrays_to_scaled_arrays(arrays_, map_to_scaled_arrays):
    """Map a list of masked 1D arrays (followed by an underscore _) to their masked 2D regular, using their
    *map_to_scaled_array* functions.

    Parameters
    -----------
    arrays_ : [ndarray]
        The arrays (e.g. datas_, model_images_, residuals) which are mapped to 2D.
    map_to_scaled_arrays : [func]
        A list of functions which map each regular from 1D to 2D, using their mask.
    """
    return list(map(lambda _array, map_to_scaled_array, : map_to_scaled_array(_array), arrays_, map_to_scaled_arrays))

def map_contributions_to_scaled_arrays(contributions_, map_to_scaled_array):
    """Map a list of masked 1D contribution maps to their masked 2D images, using each images *map_to_scaled_array*
    function.

    The reason this is separate from the *map_arrays_to_scaled_arrays* function is that each regular has a list of
    contribution maps (one for each galaxy). Thus, this function breaks down the mapping of each map and returns the
    2D images in the same data structure (2 lists with indexes [image_index][contribution_map_index].

    Parameters
    -----------
    contributions_ : [[ndarray]]
        Lists of the contribution maps which are mapped to 2D.
    map_to_scaled_arrays : [func]
        A list of functions which map each regular from 1D to 2D, using their mask.
    """
    return list(map(lambda _contribution : map_to_scaled_array(_contribution), contributions_))

def blur_image_including_blurring_region(image_, blurring_image_, convolver):
    """For a list of 1D masked images and 1D blurrings images (the regular regions outside the mask whose light blurs
    into the mask after PSF convolution), use both to computed the blurred regular within the mask via PSF convolution.

     The convolution uses each regular's convolver (*See imaging.convolution*).

    Parameters
    ----------
    image_ : ndarray
        The 1D masked data which is blurred.
    blurring_image : ndarray
        The 1D masked blurring images which are blurred.
    convolver : imaging.convolution.Convolver
        The convolver which perform the convolution and have built into the the PSF kernel.
    """
    return convolver.convolve_image(image_array=image_, blurring_array=blurring_image_)

def likelihood_with_regularization_from_chi_squared_term_regularization_and_noise_term(chi_squared_term,
                                                                                       regularization_term,
                                                                                       noise_term):
    """Compute the likelihood of each masked 1D model-datas fit to the data, including the regularization term which
    comes from an inversion:

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared_term : float
        The chi-squared terms for each model-datas fit to the observed data.
    regularization_term : float
        The regularization terms of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    noise_term : float
        The normalization noise-terms for each observed data's noise-map.
    """
    return -0.5 * (chi_squared_term + regularization_term + noise_term)

def evidence_from_reconstruction_terms(chi_squared_term, regularization_term, log_covariance_regularization_term,
                                       log_regularization_term, noise_term):
    """Compute the evidence of each masked 1D model-datas fit to the data, where an evidence includes a number of
    terms which quantify the complexity of an inversion's reconstruction (see the *inversion* module):

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Log(Covariance_Regularization_Term) -
                       Log(Regularization_Matrix_Term) + Noise_Term]

    Parameters
    ----------
    chi_squared_term : float
        The chi-squared terms for each model-datas fit to the observed data.
    regularization_term : float
        The regularization terms of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    noise_term : float
        The normalization noise-terms for each observed data's noise-map.
    """
    return -0.5 * (chi_squared_term + regularization_term + log_covariance_regularization_term -log_regularization_term
                   + noise_term)

def unmasked_blurred_images_from_fitting_images(fitting_images, unmasked_images_):
    """For a list of fitting images, compute an unmasked blurred images from a list of unmasked unblurred images.
    Unmasked images are used for plotting the results of a model outside a masked region.

    This relies on using a fitting regular's padded_grid, which is grid of coordinates which extends over the entire
    regular as opposed to just the masked region.

    Parameters
    ----------
    fitting_images_ : [fitting.fit_data.FitData]
        The fitting images.
    unmasked_images_ : [ndarray]
        The 1D unmasked images which are blurred.
    """
    return list(map(lambda fitting_image, _unmasked_image :
                    unmasked_model_image_from_fitting_image(fitting_image, _unmasked_image),
                    fitting_images, unmasked_images_))

def unmasked_model_image_from_fitting_image(fitting_image, unmasked_image_):
    """For a fitting regular, compute an unmasked blurred regular from an unmasked unblurred regular. Unmasked
    images are used for plotting the results of a model outside a masked region.

    This relies on using a fitting regular's padded_grid, which is grid of coordinates which extends over the entire
    regular as opposed to just the masked region.

    Parameters
    ----------
    fitting_image_ : fitting.fit_data.FitData
        A fitting_image, whose padded grid is used for PSF convolution.
    unmasked_images_ : [ndarray]
        The 1D unmasked images which are blurred.
    """
    _model_image = fitting_image.padded_grids.regular.convolve_array_1d_with_psf(unmasked_image_,
                                                                               fitting_image.psf)

    return fitting_image.padded_grids.regular.scaled_array_from_array_1d(_model_image)

def contributions_from_fitting_hyper_images_and_hyper_galaxies(fitting_hyper_images, hyper_galaxies):
    """For a list of fitting hyper-images (which includes the hyper model regular and hyper galaxy images) and model
    hyper-galaxies, compute their contribution maps, which are used to compute a scaled-noise map.

    Parameters
    ----------
    fitting_hyper_images : [fitting.fit_data.FitDataHyper]
        The fitting hyper-images.
    hyper_galaxies : [galaxy.Galaxy]
        The hyper-galaxies which represent the model components used to scale the noise, which correspond to
        individual galaxies in the regular.
    """
    return list(map(lambda hyp :
                    contributions_from_hyper_images_and_galaxies(hyper_model_image=hyp.hyper_model_image,
                                                                 hyper_galaxy_images=hyp.hyper_galaxy_images,
                                                                 hyper_galaxies=hyper_galaxies,
                                                                 minimum_values=hyp.hyper_minimum_values),
                fitting_hyper_images))

def contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images, hyper_galaxies, minimum_values):
    """For a fitting hyper-regular, hyper model regular, list of hyper galaxy images and model hyper-galaxies, compute
    their contribution maps, which are used to compute a scaled-noise map. All quantities are masked 1D arrays.

    The reason this is separate from the *contributions_from_fitting_hyper_images_and_hyper_galaxies* function is that
    each hyper regular has a list of hyper galaxy images and associated hyper galaxies (one for each galaxy). Thus,
    this function breaks down the calculation of each 1D masked contribution map and returns them in the same data
    structure (2 lists with indexes [image_index][contribution_map_index].

    Parameters
    ----------
    hyper_model_image : ndarray
        The best-fit model regular to the data (e.g. from a previous analysis phase).
    hyper_galaxy_images : [ndarray]
        The best-fit model regular of each hyper-galaxy to the data (e.g. from a previous analysis phase).
    hyper_galaxies : [galaxy.Galaxy]
        The hyper-galaxies which represent the model components used to scale the noise, which correspond to
        individual galaxies in the regular.
    minimum_values : [float]
        The minimum value of each hyper-images contribution map, which ensure zero's don't impact the scaled noise-map.
    """
    # noinspection PyArgumentList
    return list(map(lambda hyper, galaxy_image, minimum_value:
                    hyper.contributions_from_hyper_images(hyper_model_image, galaxy_image, minimum_value),
                    hyper_galaxies, hyper_galaxy_images, minimum_values))

def scaled_noise_maps_from_fitting_hyper_images_contributions_and_hyper_galaxies(fitting_hyper_images, contributions_,
                                                                                 hyper_galaxies):
    """For a list of fitting hyper-images (which includes the hyper model data and hyper galaxy images),
     contribution maps and model hyper-galaxies, compute their scaled noise-maps.

     This is performed by using each hyper-galaxy's *noise_factor* and *noise_power* parameter in conjunction with the
     unscaled noise-map and contribution map.

    Parameters
    ----------
    fitting_hyper_images : [fitting.fit_data.FitDataHyper]
        The fitting hyper-images.
    contributions_ : [[ndarray]]
        List of each regular's list of 1D masked contribution maps (e.g. one for each hyper-galaxy)
    hyper_galaxies : [galaxy.Galaxy]
        The hyper-galaxies which represent the model components used to scale the noise, which correspond to
        individual galaxies in the regular.
    """
    return list(map(lambda fitting_hyper_image, contribution_ :
                    scaled_noise_map_from_hyper_galaxies_and_contributions(contributions_=contribution_,
                                                                           hyper_galaxies=hyper_galaxies,
                                                                           noise_map_=fitting_hyper_image.noise_map_),
                    fitting_hyper_images, contributions_))

def scaled_noise_map_from_hyper_galaxies_and_contributions(contributions_, hyper_galaxies, noise_map_):
    """For a contribution map and noise-map, use the model hyper galaxies to compute a scaled noise-map.

    Parameters
    -----------
    contributions_ : ndarray
        The regular's list of 1D masked contribution maps (e.g. one for each hyper-galaxy)
    hyper_galaxies : [galaxy.Galaxy]
        The hyper-galaxies which represent the model components used to scale the noise, which correspond to
        individual galaxies in the regular.
    noise_map : imaging.NoiseMap or ndarray
        An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
        second.
    """
    scaled_noise_maps_ = list(map(lambda hyper, contribution_:
                              hyper.scaled_noise_from_contributions(noise_map_, contribution_),
                              hyper_galaxies, contributions_))
    return noise_map_ + sum(scaled_noise_maps_)


def blur_image_including_blurring_region(unblurred_image_1d, blurring_image_1d, convolver):
    """For a 1D masked image and 1D blurring image (the regular regions outside the mask whose light blurs \
    into the mask after PSF convolution), use both to computed the blurred regular within the mask via PSF convolution.

    The convolution uses each image's convolver (*See imaging.convolution*).

    Parameters
    ----------
    unblurred_image_1d : ndarray
        The 1D masked data which is blurred.
    blurring_image_1d : ndarray
        The 1D masked blurring images which are blurred.
    convolver_ : imaging.convolution.Convolver
        The convolver which perform the convolution and have built into the the PSF kernel.
    """
    return convolver.convolve_image(image_array=unblurred_image_1d, blurring_array=blurring_image_1d)

def likelihood_with_regularization_from_chi_squared_term_regularization_and_noise_term(chi_squared_term,
                                                                                       regularization_term,
                                                                                       noise_term):
    """Compute the likelihood of an inversion's fit to the data, including a regularization term which \
    comes from an inversion:

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared_term : float
        The chi-squared term of the inversion's fit to the observed data.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    noise_term : float
        The normalization noise-term for the observed data's noise-map.
    """
    return -0.5 * (chi_squared_term + regularization_term + noise_term)

def evidence_from_reconstruction_terms(chi_squared_term, regularization_term, log_covariance_regularization_term,
                                       log_regularization_term, noise_term):
    """Compute the evidence of an inversion's fit to the data, where the evidence includes a number of \
    terms which quantify the complexity of an inversion's reconstruction (see the *inversion* module):

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Log(Covariance_Regularization_Term) -
                       Log(Regularization_Matrix_Term) + Noise_Term]

    Parameters
    ----------
    chi_squared_term : float
        The chi-squared term of the inversion's fit to the observed data.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    log_covariance_regularization_term : float
        The log of the determinant of the sum of the curvature and regularization matrices.
    log_regularization_term : float
        The log of the determinant o the regularization matrix.
    noise_term : float
        The normalization noise-term for the observed data's noise-map.
    """
    return -0.5 * (chi_squared_term + regularization_term + log_covariance_regularization_term -log_regularization_term
                   + noise_term)

# def unmasked_blurred_images_from_fitting_images(fitting_images, unmasked_images_):
#     """For a list of fitting images, compute an unmasked blurred images from a list of unmasked unblurred images.
#     Unmasked images are used for plotting the results of a model outside a masked region.
#
#     This relies on using a fitting regular's padded_grid, which is grid of coordinates which extends over the entire
#     regular as opposed to just the masked region.
#
#     Parameters
#     ----------
#     fitting_images_ : [fitting.fit_data.FitData]
#         The fitting images.
#     unmasked_images_ : [ndarray]
#         The 1D unmasked images which are blurred.
#     """
#     return list(map(lambda fitting_image, _unmasked_image :
#                     unmasked_model_image_from_psf_padded_grids_and_unmasked_image(fitting_image, _unmasked_image),
#                     fitting_images, unmasked_images_))

# def contributions_from_fitting_hyper_images_and_hyper_galaxies(fitting_hyper_images, hyper_galaxies):
#     """For a list of fitting hyper-images (which includes the hyper model regular and hyper galaxy images) and model
#     hyper-galaxies, compute their contribution maps, which are used to compute a scaled-noise map.
#
#     Parameters
#     ----------
#     fitting_hyper_images : [fitting.fit_data.FitDataHyper]
#         The fitting hyper-images.
#     hyper_galaxies : [galaxy.Galaxy]
#         The hyper-galaxies which represent the model components used to scale the noise, which correspond to
#         individual galaxies in the regular.
#     """
#     return list(map(lambda hyp :
#                     contributions_from_hyper_images_and_galaxies(hyper_model_image=hyp.hyper_model_image,
#                                                                  hyper_galaxy_images=hyp.hyper_galaxy_images,
#                                                                  hyper_galaxies=hyper_galaxies,
#                                                                  minimum_values=hyp.hyper_minimum_values),
#                 fitting_hyper_images))

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

# def scaled_noise_maps_from_fitting_hyper_images_contributions_and_hyper_galaxies(fitting_hyper_images, contributions_,
#                                                                                  hyper_galaxies):
#     """For a list of fitting hyper-images (which includes the hyper model data and hyper galaxy images),
#      contribution maps and model hyper-galaxies, compute their scaled noise-maps.
#
#      This is performed by using each hyper-galaxy's *noise_factor* and *noise_power* parameter in conjunction with the
#      unscaled noise-map and contribution map.
#
#     Parameters
#     ----------
#     fitting_hyper_images : [fitting.fit_data.FitDataHyper]
#         The fitting hyper-images.
#     contributions_ : [[ndarray]]
#         List of each regular's list of 1D masked contribution maps (e.g. one for each hyper-galaxy)
#     hyper_galaxies : [galaxy.Galaxy]
#         The hyper-galaxies which represent the model components used to scale the noise, which correspond to
#         individual galaxies in the regular.
#     """
#     return list(map(lambda fitting_hyper_image, contribution_ :
#                     scaled_noise_map_from_hyper_galaxies_and_contributions(contributions_=contribution_,
#                                                                            hyper_galaxies=hyper_galaxies,
#                                                                            noise_map_1d=fitting_hyper_image.noise_map_1d),
#                     fitting_hyper_images, contributions_))

def scaled_noise_map_from_hyper_galaxies_and_contributions(contributions, hyper_galaxies, noise_map):
    """For a contribution map and noise-map, use the model hyper galaxies to compute a scaled noise-map.

    Parameters
    -----------
    contributions : ndarray
        The regular's list of 1D masked contribution maps (e.g. one for each hyper-galaxy)
    hyper_galaxies : [galaxy.Galaxy]
        The hyper-galaxies which represent the model components used to scale the noise, which correspond to
        individual galaxies in the regular.
    noise_map : imaging.NoiseMap or ndarray
        An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
        second.
    """
    scaled_noise_maps_ = list(map(lambda hyper, contribution_:
                              hyper.scaled_noise_from_contributions(noise_map, contribution_),
                                  hyper_galaxies, contributions))
    return noise_map + sum(scaled_noise_maps_)


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

def unmasked_model_images_of_galaxies_from_lensing_images_and_tracer(lensing_images, tracer):
    return list(map(lambda lensing_image, image_index :
                    unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(lensing_image, tracer, image_index),
                    lensing_images, list(range(tracer.total_images))))

def unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(lensing_image, tracer, image_index):


    padded_model_images_of_galaxies = [[] for _ in range(len(tracer.all_planes))]

    for plane_index, plane in enumerate(tracer.all_planes):
        for galaxy_index in range(len(plane.galaxies)):

            _galaxy_image_plane_image = plane.image_plane_images_of_galaxies_[image_index][galaxy_index]

            galaxy_model_image = fitting_util.unmasked_model_image_from_fitting_image(fitting_image=lensing_image,
                                                                                 unmasked_image_=_galaxy_image_plane_image)

            padded_model_images_of_galaxies[plane_index].append(galaxy_model_image)

    return padded_model_images_of_galaxies
import numpy as np

def blurred_image_1d_from_1d_unblurred_and_blurring_images(unblurred_image_1d, blurring_image_1d, convolver):
    """For a 1D masked image and 1D blurring image (the regular regions outside the masks whose light blurs \
    into the masks after PSF convolution), use both to computed the blurred regular within the masks via PSF convolution.

    The convolution uses each image's convolver (*See imaging.convolution*).

    Parameters
    ----------
    unblurred_image_1d : ndarray
        The 1D masked datas which is blurred.
    blurring_image_1d : ndarray
        The 1D masked blurring unblurred_image_1d which are blurred.
    convolver_ : imaging.convolution.Convolver
        The convolver which perform the convolution and have built into the the PSF kernel.
    """
    return convolver.convolve_image(image_array=unblurred_image_1d, blurring_array=blurring_image_1d)

def blurred_image_from_1d_unblurred_and_blurring_images(unblurred_image_1d, blurring_image_1d, convolver,
                                                        map_to_scaled_array):
    """For a 1D masked image and 1D blurring image (the regular regions outside the masks whose light blurs \
    into the masks after PSF convolution), use both to computed the blurred regular within the masks via PSF convolution.

    This blurred image is then mapped to the 2D array of the masked image datas.

    Parameters
    ----------
    unblurred_image_1d : ndarray
        The 1D masked datas which is blurred.
    blurring_image_1d : ndarray
        The 1D masked blurring unblurred_image_1d which are blurred.
    convolver : imaging.convolution.Convolver
        The convolver which perform the convolution and have built into the the PSF kernel.
    map_to_scaled_array : func
        Function in lensing image which maps a 1D array of datas to its masked 2D array.
    """
    blurred_image_1d = blurred_image_1d_from_1d_unblurred_and_blurring_images(unblurred_image_1d=unblurred_image_1d,
                                                                              blurring_image_1d=blurring_image_1d,
                                                                              convolver=convolver)

    return map_to_scaled_array(array_1d=blurred_image_1d)

def likelihood_with_regularization_from_chi_squared_term_regularization_and_noise_term(chi_squared_term,
                                                                                       regularization_term,
                                                                                       noise_term):
    """Compute the likelihood of an inversion's fit to the datas, including a regularization term which \
    comes from an inversion:

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared_term : float
        The chi-squared term of the inversion's fit to the observed datas.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    noise_term : float
        The normalization noise-term for the observed datas's noise-map.
    """
    return -0.5 * (chi_squared_term + regularization_term + noise_term)

def evidence_from_reconstruction_terms(chi_squared_term, regularization_term, log_covariance_regularization_term,
                                       log_regularization_term, noise_term):
    """Compute the evidence of an inversion's fit to the datas, where the evidence includes a number of \
    terms which quantify the complexity of an inversion's reconstruction (see the *inversion* module):

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Log(Covariance_Regularization_Term) -
                       Log(Regularization_Matrix_Term) + Noise_Term]

    Parameters
    ----------
    chi_squared_term : float
        The chi-squared term of the inversion's fit to the observed datas.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    log_covariance_regularization_term : float
        The log of the determinant of the sum of the curvature and regularization matrices.
    log_regularization_term : float
        The log of the determinant o the regularization matrix.
    noise_term : float
        The normalization noise-term for the observed datas's noise-map.
    """
    return -0.5 * (chi_squared_term + regularization_term + log_covariance_regularization_term -log_regularization_term
                   + noise_term)

def blurred_image_of_planes_from_tracer_and_convolver(tracer, convolver_image, map_to_scaled_array):
    """For a tracer, extract the image-plane image of every plane and blur it with the PSF.

    If none of the galaxies in a plane have a light profie or pixelization (and thus don't have an image) a *None* \
    is used.

    Parameters
    ----------
    tracer : ray_tracing.AbstractTracer
        The tracer, which describes the ray-tracing of the strong lensing configuration.
    convolver_image : data.imaging.convolution.ConvolverImage
        Class which performs the PSF convolution of a masked image in 1D.
    map_to_scaled_arrays : [func]
        A functions which maps a masked image from 1D to 2D.
    """

    blurred_image_of_planes = []

    for plane_index in range(tracer.total_planes):

        # If all entries are zero, there was no light profile / pixeization
        if np.count_nonzero(tracer.image_plane_image_1d_of_planes[plane_index]) > 0:

            blurred_image_of_plane = blurred_image_from_1d_unblurred_and_blurring_images(
                unblurred_image_1d=tracer.image_plane_image_1d_of_planes[plane_index],
                blurring_image_1d=tracer.image_plane_blurring_image_of_planes_1d[plane_index],
                convolver=convolver_image, map_to_scaled_array=map_to_scaled_array)

            blurred_image_of_planes.append(blurred_image_of_plane)

        else:

            blurred_image_of_planes.append(None)

    return blurred_image_of_planes

def unmasked_blurred_image_from_padded_grid_stack_psf_and_unmasked_image(padded_grid_stack, psf, unmasked_image_1d):
    """For a fitting regular, compute an unmasked blurred regular from an unmasked unblurred regular. Unmasked
    images are used for plotting the results of a model outside a masked region.

    This relies on using a fitting regular's padded_grid, which is grid of coordinates which extends over the entire
    regular as opposed to just the masked region.

    Parameters
    ----------
    fitting_image_ : fitting.fitting_data.FittingImage
        A padded_grid_stack, whose padded grid is used for PSF convolution.
    unmasked_images_ : [ndarray]
        The 1D unmasked images which are blurred.
    """
    blurred_image_1d = padded_grid_stack.regular.convolve_array_1d_with_psf(padded_array_1d=unmasked_image_1d,
                                                                            psf=psf)

    return padded_grid_stack.regular.scaled_array_from_array_1d(array_1d=blurred_image_1d)

def unmasked_blurred_image_of_galaxies_from_padded_grid_stack_psf_and_tracer(padded_grid_stack, psf, tracer):

    unmasked_blurred_images_of_galaxies = [[] for _ in range(len(tracer.all_planes))]

    for plane_index, plane in enumerate(tracer.all_planes):
        for galaxy_index in range(len(plane.galaxies)):

            image_plane_image_1d_of_galaxy = plane.image_plane_image_1d_of_galaxies[galaxy_index]

            blurred_galaxy_image = unmasked_blurred_image_from_padded_grid_stack_psf_and_unmasked_image(
                padded_grid_stack=padded_grid_stack, psf=psf, unmasked_image_1d=image_plane_image_1d_of_galaxy)

            unmasked_blurred_images_of_galaxies[plane_index].append(blurred_galaxy_image)

    return unmasked_blurred_images_of_galaxies

# def contributions_from_fitting_hyper_images_and_hyper_galaxies(fitting_hyper_images, hyper_galaxies):
#     """For a list of fitting hyper_galaxy-unblurred_image_1d (which includes the hyper_galaxy model regular and hyper_galaxy galaxy unblurred_image_1d) and model
#     hyper_galaxy-galaxies, compute their contribution maps, which are used to compute a scaled-noise map.
#
#     Parameters
#     ----------
#     fitting_hyper_images : [fitting.fit_data.FitDataHyper]
#         The fitting hyper_galaxy-unblurred_image_1d.
#     hyper_galaxies : [galaxy.Galaxy]
#         The hyper_galaxy-galaxies which represent the model components used to scale the noise, which correspond to
#         individual galaxies in the regular.
#     """
#     return list(map(lambda hyp :
#                     contributions_from_hyper_images_and_galaxies(hyper_model_image=hyp.hyper_model_image,
#                                                                  hyper_galaxy_images=hyp.hyper_galaxy_images,
#                                                                  hyper_galaxies=hyper_galaxies,
#                                                                  minimum_values=hyp.hyper_minimum_values),
#                 fitting_hyper_images))

def contributions_from_hyper_images_and_galaxies(hyper_model_image_1d, hyper_galaxy_images_1d, hyper_galaxies,
                                                 hyper_minimum_values):
    """For a fitting hyper_galaxy-regular, hyper_galaxy model regular, list of hyper_galaxy galaxy unblurred_image_1d and model hyper_galaxy-galaxies, compute
    their contribution maps, which are used to compute a scaled-noise map. All quantities are masked 1D arrays.

    The reason this is separate from the *contributions_from_fitting_hyper_images_and_hyper_galaxies* function is that
    each hyper_galaxy regular has a list of hyper_galaxy galaxy unblurred_image_1d and associated hyper_galaxy galaxies (one for each galaxy). Thus,
    this function breaks down the calculation of each 1D masked contribution map and returns them in the same datas
    structure (2 lists with indexes [image_index][contribution_map_index].

    Parameters
    ----------
    hyper_model_image_1d : ndarray
        The best-fit model regular to the datas (e.g. from a previous analysis phase).
    hyper_galaxy_images_1d : [ndarray]
        The best-fit model regular of each hyper_galaxy-galaxy to the datas (e.g. from a previous analysis phase).
    hyper_galaxies : [galaxy.Galaxy]
        The hyper_galaxy-galaxies which represent the model components used to scale the noise, which correspond to
        individual galaxies in the regular.
    hyper_minimum_values : [float]
        The minimum value of each hyper_galaxy-unblurred_image_1d contribution map, which ensure zero's don't impact the scaled noise-map.
    """
    # noinspection PyArgumentList
    return list(map(lambda hyper_galaxy, hyper_galaxy_image_1d, hyper_minimum_value:
                    hyper_galaxy.contributions_from_hyper_images(hyper_model_image=hyper_model_image_1d,
                                                                 hyper_galaxy_image=hyper_galaxy_image_1d,
                                                                 hyper_minimum_value=hyper_minimum_value),
                    hyper_galaxies, hyper_galaxy_images_1d, hyper_minimum_values))

# def scaled_noise_maps_from_fitting_hyper_images_contributions_and_hyper_galaxies(fitting_hyper_images, contributions_1d,
#                                                                                  hyper_galaxies):
#     """For a list of fitting hyper_galaxy-unblurred_image_1d (which includes the hyper_galaxy model datas and hyper_galaxy galaxy unblurred_image_1d),
#      contribution maps and model hyper_galaxy-galaxies, compute their scaled noise-maps.
#
#      This is performed by using each hyper_galaxy-galaxy's *noise_factor* and *noise_power* parameter in conjunction with the
#      unscaled noise-map and contribution map.
#
#     Parameters
#     ----------
#     fitting_hyper_images : [fitting.fit_data.FitDataHyper]
#         The fitting hyper_galaxy-unblurred_image_1d.
#     contributions_1d : [[ndarray]]
#         List of each regular's list of 1D masked contribution maps (e.g. one for each hyper_galaxy-galaxy)
#     hyper_galaxies : [galaxy.Galaxy]
#         The hyper_galaxy-galaxies which represent the model components used to scale the noise, which correspond to
#         individual galaxies in the regular.
#     """
#     return list(map(lambda fitting_hyper_image, contribution_1d :
#                     scaled_noise_map_from_hyper_galaxies_and_contributions(contributions_1d=contribution_1d,
#                                                                            hyper_galaxies=hyper_galaxies,
#                                                                            noise_map_1d=fitting_hyper_image.noise_map_1d),
#                     fitting_hyper_images, contributions_1d))

def scaled_noise_map_from_hyper_galaxies_and_contributions(contributions_1d, hyper_galaxies, noise_map_1d):
    """For a contribution map and noise-map, use the model hyper_galaxy galaxies to compute a scaled noise-map.

    Parameters
    -----------
    contributions_1d : ndarray
        The regular's list of 1D masked contribution maps (e.g. one for each hyper_galaxy-galaxy)
    hyper_galaxies : [galaxy.Galaxy]
        The hyper_galaxy-galaxies which represent the model components used to scale the noise, which correspond to
        individual galaxies in the regular.
    noise_map_1d : imaging.NoiseMap or ndarray
        An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
        second.
    """
    scaled_noise_maps_1d = list(map(lambda hyper_galaxy, contribution_1d:
                              hyper_galaxy.scaled_noise_from_contributions(noise_map=noise_map_1d,
                                                                           contributions=contribution_1d),
                              hyper_galaxies, contributions_1d))
    return noise_map_1d + sum(scaled_noise_maps_1d)


def map_contributions_to_scaled_arrays(contributions_, map_to_scaled_array):
    """Map a list of masked 1D contribution maps to their masked 2D unblurred_image_1d, using each unblurred_image_1d *map_to_scaled_array*
    function.

    The reason this is separate from the *map_arrays_to_scaled_arrays* function is that each regular has a list of
    contribution maps (one for each galaxy). Thus, this function breaks down the mapping of each map and returns the
    2D unblurred_image_1d in the same datas structure (2 lists with indexes [image_index][contribution_map_index].

    Parameters
    -----------
    contributions_ : [[ndarray]]
        Lists of the contribution maps which are mapped to 2D.
    map_to_scaled_arrays : [func]
        A list of functions which map each regular from 1D to 2D, using their masks.
    """
    return list(map(lambda _contribution : map_to_scaled_array(_contribution), contributions_))

# def unmasked_blurred_images_from_fitting_images(fitting_images, unmasked_images_):
#     """For a list of fitting unblurred_image_1d, compute an unmasked blurred unblurred_image_1d from a list of unmasked unblurred unblurred_image_1d.
#     Unmasked unblurred_image_1d are used for plotting the results of a model outside a masked region.
#
#     This relies on using a fitting regular's padded_grid, which is grid of coordinates which extends over the entire
#     regular as opposed to just the masked region.
#
#     Parameters
#     ----------
#     fitting_images_ : [fitting.fit_data.FitData]
#         The fitting unblurred_image_1d.
#     unmasked_images_ : [ndarray]
#         The 1D unmasked unblurred_image_1d which are blurred.
#     """
#     return list(map(lambda fitting_image, _unmasked_image :
#                     unmasked_model_image_from_psf_padded_grids_and_unmasked_image(fitting_image, _unmasked_image),
#                     fitting_images, unmasked_images_))

# def unmasked_model_images_of_galaxies_from_lensing_images_and_tracer(lensing_images, tracer):
#     return list(map(lambda lensing_image, image_index :
#                     unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(lensing_image, tracer, image_index),
#                     lensing_images, list(range(tracer.total_grid_stacks))))

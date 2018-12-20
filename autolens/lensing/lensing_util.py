import numpy as np

def ordered_redshifts_from_galaxies(galaxies):
    """Given a list of galaxies (with redshifts), return a list of the in ascending order.

    If two or more galaxies have the same redshift that redshift is not double counted.

    Parameters
    -----------
    galaxies : [Galaxy]
        The list of galaxies in the ray-tracing calculation.
    """
    ordered_galaxies = sorted(galaxies, key=lambda galaxy: galaxy.redshift, reverse=False)

    # Ideally we'd extract the planes_red_Shfit order from the list above. However, I dont know how to extract it
    # Using a list of class attributes so make a list of redshifts for now.

    galaxy_redshifts = list(map(lambda galaxy: galaxy.redshift, ordered_galaxies))
    return [redshift for i, redshift in enumerate(galaxy_redshifts) if redshift not in galaxy_redshifts[:i]]

def galaxies_in_redshift_ordered_lists_from_galaxies(galaxies, ordered_redshifts):
    """Given a list of galaxies (with redshifts), return a list of the galaxies where each entry contains a list \
    of galaxies at the same redshift in ascending redshift order.

    Parameters
    -----------
    galaxies : [Galaxy]
        The list of galaxies in the ray-tracing calculation.
    """
    ordered_galaxies = sorted(galaxies, key=lambda galaxy: galaxy.redshift, reverse=False)

    galaxies_in_redshift_ordered_lists = []

    for (index, redshift) in enumerate(ordered_redshifts):

        galaxies_in_redshift_ordered_lists.append(list(map(lambda galaxy:
                                                            galaxy if galaxy.redshift == redshift else None,
                                                            ordered_galaxies)))

        galaxies_in_redshift_ordered_lists[index] = list(filter(None, galaxies_in_redshift_ordered_lists[index]))

    return galaxies_in_redshift_ordered_lists

def scaling_factor_between_redshifts_for_cosmology(z1, z2, z_final, cosmology):

    angular_diameter_distance_between_z1_z2 = cosmology.angular_diameter_distance_z1z2(z1=z1, z2=z2).to('kpc').value
    angular_diameter_distance_to_z_final = cosmology.angular_diameter_distance(z=z_final).to('kpc').value
    angular_diameter_distance_of_z2_to_earth = cosmology.angular_diameter_distance(z=z2).to('kpc').value
    angular_diameter_distance_between_z2_z_final = \
        cosmology.angular_diameter_distance_z1z2(z1=z1, z2=z_final).to('kpc').value

    return (angular_diameter_distance_between_z1_z2 * angular_diameter_distance_to_z_final) / \
           (angular_diameter_distance_of_z2_to_earth * angular_diameter_distance_between_z2_z_final)

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

def likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(chi_squared,
                                                                  regularization_term, noise_normalization):
    """Compute the likelihood of an inversion's fit to the datas, including a regularization term which \
    comes from an inversion:

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared : float
        The chi-squared term of the inversion's fit to the observed datas.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    noise_normalization : float
        The normalization noise_map-term for the observed datas's noise-map.
    """
    return -0.5 * (chi_squared + regularization_term + noise_normalization)

def evidence_from_inversion_terms(chi_squared, regularization_term, log_curvature_regularization_term,
                                  log_regularization_term, noise_normalization):
    """Compute the evidence of an inversion's fit to the datas, where the evidence includes a number of \
    terms which quantify the complexity of an inversion's reconstruction (see the *inversion* module):

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Log(Covariance_Regularization_Term) -
                       Log(Regularization_Matrix_Term) + Noise_Term]

    Parameters
    ----------
    chi_squared : float
        The chi-squared term of the inversion's fit to the observed datas.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    log_curvature_regularization_term : float
        The log of the determinant of the sum of the curvature and regularization matrices.
    log_regularization_term : float
        The log of the determinant o the regularization matrix.
    noise_normalization : float
        The normalization noise_map-term for the observed datas's noise-map.
    """
    return -0.5 * (chi_squared + regularization_term + log_curvature_regularization_term - log_regularization_term
                   + noise_normalization)

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

    unmasked_blurred_images_of_galaxies = [[] for _ in range(len(tracer.planes))]

    for plane_index, plane in enumerate(tracer.planes):
        for galaxy_index in range(len(plane.galaxies)):

            image_plane_image_1d_of_galaxy = plane.image_plane_image_1d_of_galaxies[galaxy_index]

            blurred_galaxy_image = unmasked_blurred_image_from_padded_grid_stack_psf_and_unmasked_image(
                padded_grid_stack=padded_grid_stack, psf=psf, unmasked_image_1d=image_plane_image_1d_of_galaxy)

            unmasked_blurred_images_of_galaxies[plane_index].append(blurred_galaxy_image)

    return unmasked_blurred_images_of_galaxies

def contribution_maps_1d_from_hyper_images_and_galaxies(hyper_model_image_1d, hyper_galaxy_images_1d, hyper_galaxies,
                                                        hyper_minimum_values):
    """For a fitting hyper_galaxy_image, hyper_galaxy model regular, list of hyper_galaxy galaxy unblurred_image_1d and model hyper_galaxy-galaxies, compute
    their contribution maps, which are used to compute a scaled-noise_map map. All quantities are masked 1D arrays.

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
        The hyper_galaxy-galaxies which represent the model components used to scale the noise_map, which correspond to
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

def contribution_maps_from_hyper_images_and_galaxies(hyper_model_image_1d, hyper_galaxy_images_1d, hyper_galaxies,
                                                     hyper_minimum_values, map_to_scaled_array):
    """For a fitting hyper_galaxy_image, hyper_galaxy model regular, list of hyper_galaxy galaxy unblurred_image_1d and model hyper_galaxy-galaxies, compute
    their contribution maps, which are used to compute a scaled-noise_map map. All quantities are masked 1D arrays.

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
        The hyper_galaxy-galaxies which represent the model components used to scale the noise_map, which correspond to
        individual galaxies in the regular.
    hyper_minimum_values : [float]
        The minimum value of each hyper_galaxy-unblurred_image_1d contribution map, which ensure zero's don't impact the scaled noise-map.
    """

    contributions_maps_1d = contribution_maps_1d_from_hyper_images_and_galaxies(hyper_model_image_1d=hyper_model_image_1d,
                                                                           hyper_galaxy_images_1d=hyper_galaxy_images_1d,
                                                                           hyper_galaxies=hyper_galaxies,
                                                                           hyper_minimum_values=hyper_minimum_values)

    return list(map(lambda contribution_map_1d : map_to_scaled_array(array_1d=contribution_map_1d),
                    contributions_maps_1d))


def scaled_noise_map_from_hyper_galaxies_and_contribution_maps(contribution_maps, hyper_galaxies, noise_map):
    """For a contribution map and noise-map, use the model hyper_galaxy galaxies to compute a scaled noise-map.

    Parameters
    -----------
    contribution_maps : ndarray
        The regular's list of 1D masked contribution maps (e.g. one for each hyper_galaxy-galaxy)
    hyper_galaxies : [galaxy.Galaxy]
        The hyper_galaxy-galaxies which represent the model components used to scale the noise_map, which correspond to
        individual galaxies in the regular.
    noise_map : imaging.NoiseMap or ndarray
        An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
        second.
    """
    scaled_noise_maps = list(map(lambda hyper_galaxy, contribution_map:
                              hyper_galaxy.scaled_noise_from_contributions(noise_map=noise_map,
                                                                           contributions=contribution_map),
                                    hyper_galaxies, contribution_maps))
    return noise_map + sum(scaled_noise_maps)